# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model
from einops import rearrange
import tensorflow_addons as tfa
from decord import VideoReader, cpu

# Custom utility functions
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from transforms import *
from masking_generator import TubeMaskingGenerator

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = tf.keras.Sequential([
            self.train_augmentation,
            Stack(roll=False),
            ToTensor(),
            GroupNormalize(self.input_mean, self.input_std)
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()

def get_model(args):
    print(f"Creating model: {args.model}")
    model = load_model(args.model_path)
    return model

def main(args):
    print(args)

    device = args.device
    model = get_model(args)
    patch_size = (16, 16)  # This needs to match your model's patch size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    frame_id_list = list(np.arange(0, 32, 2) + 60)
    video_data = vr.get_batch(frame_id_list).asnumpy()
    print(video_data.shape)
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid in range(len(frame_id_list))]

    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None))  # T*C,H,W
    img = tf.reshape(img, (args.num_frames, 3) + img.shape[-2:])
    img = tf.transpose(img, perm=[1, 0, 2, 3])  # T*C,H,W -> T,C,H,W -> C,T,H,W
    bool_masked_pos = tf.convert_to_tensor(bool_masked_pos, dtype=tf.bool)

    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    bool_masked_pos = tf.expand_dims(bool_masked_pos, axis=0)

    outputs = model([img, bool_masked_pos])

    # Save original video
    mean = tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=tf.float32)[None, :, None, None, None]
    std = tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=tf.float32)[None, :, None, None, None]
    ori_img = img * std + mean  # in [0, 1]
    imgs = [tf.keras.preprocessing.image.array_to_img(ori_img[0, :, vid, :, :]) for vid in range(len(frame_id_list))]
    for id, im in enumerate(imgs):
        im.save(f"{args.save_path}/ori_img{id}.jpg")

    img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
    img_norm = (img_squeeze - tf.reduce_mean(img_squeeze, axis=-2, keepdims=True)) / (tf.math.reduce_std(img_squeeze, axis=-2, keepdims=True) + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    img_patch = tf.tensor_scatter_nd_update(img_patch, bool_masked_pos, outputs)

    # Make mask
    mask = tf.ones_like(img_patch)
    mask = tf.tensor_scatter_nd_update(mask, bool_masked_pos, tf.zeros_like(img_patch))
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

    # Save reconstruction video
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    rec_img = rec_img * (tf.math.reduce_std(img_squeeze, axis=-2, keepdims=True) + 1e-6) + tf.reduce_mean(img_squeeze, axis=-2, keepdims=True)
    rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
    imgs = [tf.keras.preprocessing.image.array_to_img(rec_img[0, :, vid, :, :]) for vid in range(len(frame_id_list))]
    for id, im in enumerate(imgs):
        im.save(f"{args.save_path}/rec_img{id}.jpg")

    # Save masked video 
    img_mask = rec_img * mask
    imgs = [tf.keras.preprocessing.image.array_to_img(img_mask[0, :, vid, :, :]) for vid in range(len(frame_id_list))]
    for id, im in enumerate(imgs):
        im.save(f"{args.save_path}/mask_img{id}.jpg")

if __name__ == '__main__':
    opts = get_args()
    main(opts)
