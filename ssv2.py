import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import warnings
import pandas as pd
from decord import VideoReader, cpu
from tensorflow.keras.utils import Sequence
from tensorflow.image import ResizeMethod

class SSVideoClsDataset(Sequence):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 crop_size=224, short_side_size=256, new_height=256,
                 new_width=340, keep_aspect_ratio=True, num_segment=1,
                 num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if mode == 'validation':
            self.data_transform = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.Resizing(self.short_side_size, self.short_side_size, interpolation='bilinear'),
                tf.keras.layers.CenterCrop(self.crop_size, self.crop_size),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = tf.keras.layers.experimental.preprocessing.Resizing(self.short_side_size, self.short_side_size, interpolation='bilinear')
            self.data_transform = tf.keras.Sequential([
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.test_num_crop - 1)
            temporal_start = chunk_nb # 0/1
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::2, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::2, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            tf.keras.preprocessing.image.array_to_img(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [tf.keras.preprocessing.image.img_to_array(img) for img in buffer]
        buffer = tf.stack(buffer) # T H W C

        buffer = tf.image.per_image_standardization(buffer)

        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = tfa.image.random_cutout(
                buffer, (args.reprob, args.reprob), constant_values=0
            )
            buffer = erase_transform

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = []
            tick = len(vr) / float(self.num_segment)
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                               [int(tick * x) for x in range(self.num_segment)]))
            while len(all_index) < (self.num_segment * self.test_num_segment):
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index)))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224,
                     random_horizontal_flip=True, inverse_uniform_sampling=False,
                     aspect_ratio=None, scale=None, motion_shift=False):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, bottom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames = tf.image.resize_with_crop_or_pad(
                frames, min_scale, max_scale)
            frames = tf.image.random_crop(frames, size=[crop_size, crop_size, frames.shape[-1]])
        else:
            frames = tf.image.random_crop(frames, size=[crop_size, crop_size, frames.shape[-1]])
        if random_horizontal_flip:
            frames = tf.image.random_flip_left_right(frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expected to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames = tf.image.resize_with_crop_or_pad(
            frames, min_scale, max_scale)
        frames = tf.image.central_crop(frames, crop_size)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == tf.uint8:
        tensor = tf.cast(tensor, tf.float32)
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = tf.constant(mean)
    if type(std) == list:
        std = tf.constant(std)
    tensor = (tensor - mean) / std
    return tensor
