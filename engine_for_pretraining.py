import math
import sys
from typing import Iterable
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
import numpy as np
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def train_one_epoch(model: tf.keras.Model, data_loader: Iterable, optimizer: optimizers.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.trainable = True
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = losses.MeanSquaredError()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = tf.convert_to_tensor(videos, dtype=tf.float32)
        bool_masked_pos = tf.convert_to_tensor(bool_masked_pos, dtype=tf.bool)

        # calculate the predict label
        mean = tf.constant(IMAGENET_DEFAULT_MEAN, dtype=tf.float32)[None, :, None, None, None]
        std = tf.constant(IMAGENET_DEFAULT_STD, dtype=tf.float32)[None, :, None, None, None]
        unnorm_videos = videos * std + mean  # in [0, 1]

        if normlize_target:
            videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
            videos_norm = (videos_squeeze - tf.reduce_mean(videos_squeeze, axis=-2, keepdims=True)) / (tf.math.reduce_variance(videos_squeeze, axis=-2, keepdims=True) ** 0.5 + 1e-6)
            videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
        else:
            videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

        B, _, C = videos_patch.shape
        labels = tf.boolean_mask(videos_patch, bool_masked_pos).reshape(B, -1, C)

        with tf.GradientTape() as tape:
            outputs = model(videos, training=True)
            loss = loss_func(labels, outputs)

        loss_value = loss.numpy()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        metric_logger.update(loss=loss_value)
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

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
