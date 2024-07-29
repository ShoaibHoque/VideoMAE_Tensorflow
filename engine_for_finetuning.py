import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples, training=True)
    loss = criterion(target, outputs)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: tf.keras.Model, criterion, data_loader: Iterable, optimizer: optimizers.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0, model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, num_training_steps_per_epoch=None,
                    update_freq=None):
    model.trainable = True
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.int64)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = tf.cast(samples, tf.float16)
            loss, output = train_class_batch(model, samples, targets, criterion)
        else:
            with tf.GradientTape() as tape:
                loss, output = train_class_batch(model, samples, targets, criterion)

        loss_value = loss.numpy()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss = loss / update_freq
            tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # Assuming that loss_scaler is a gradient scaler
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss = loss / update_freq
            grads = tape.gradient(loss, model.trainable_variables)
            loss_scaler(grads, optimizer, clip_grad=max_norm,
                        parameters=model.trainable_variables, create_graph=is_second_order,
                        update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        if mixup_fn is None:
            class_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), targets), tf.float32))
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validation_one_epoch(data_loader, model):
    criterion = losses.SparseCategoricalCrossentropy(from_logits=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.trainable = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = tf.convert_to_tensor(videos, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.int64)

        # compute output
        with tf.GradientTape() as tape:
            output = model(videos, training=False)
            loss = criterion(target, output)

        acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), target), tf.float32))
        acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(target, output, 5), tf.float32))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1.numpy(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.numpy(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def final_test(data_loader, model, file):
    criterion = losses.SparseCategoricalCrossentropy(from_logits=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.trainable = False
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = tf.convert_to_tensor(videos, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.int64)

        # compute output
        with tf.GradientTape() as tape:
            output = model(videos, training=False)
            loss = criterion(target, output)

        for i in range(output.shape[0]):
            string = "{} {} {} {} {}\n".format(ids[i].numpy(), \
                                                str(output.numpy()[i].tolist()), \
                                                str(int(target.numpy()[i])), \
                                                str(int(chunk_nb[i].numpy())), \
                                                str(int(split_nb[i].numpy())))
            final_result.append(string)

        acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), target), tf.float32))
        acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(target, output, 5), tf.float32))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1.numpy(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.numpy(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1.numpy(), acc5.numpy()))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
