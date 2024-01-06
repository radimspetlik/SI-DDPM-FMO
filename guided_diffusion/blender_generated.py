# MIT License
#
# Copyright (c) 2023 Radim Spetlik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import logging
import random
import math
import socket

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py

import conf_mgt
from conf_mgt.conf_base import get_path_subconfig_name
from guided_diffusion.train_util import HostnameFilter

logger = logging.getLogger()
logger.addFilter(HostnameFilter())


class BlenderDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        raise RuntimeError("Do not use abstract dataset!")

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    @staticmethod
    def compose(fg_0_1, bg_0_1, fg_alpha_0_1):
        fg_0_1 = np.power(fg_0_1, 2.2)
        bg_0_1 = np.power(bg_0_1, 2.2)
        blend = (1.0 - fg_alpha_0_1) * bg_0_1 + fg_alpha_0_1 * fg_0_1
        blend = np.power(blend, 1 / 2.2)

        return blend

    @staticmethod
    def from_255_to_m1p1(img):
        img = img.copy()
        for cycle_id in range(math.ceil(img.shape[0] / 3)):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img[idx_from:idx_to] = img[idx_from:idx_to].astype(np.float32) / 127.5 - 1

        return img

    def resize_clip_to_0_255(self, img_0_255):
        if self.transform is None:
            w = self.config['model']['input_width']
            h = self.config['model']['input_height']
            img_resized_0_255 = cv2.resize(img_0_255, (w, h))
        else:
            pil_image = Image.fromarray(img_0_255)
            img_resized_0_255 = np.array(self.transform(pil_image))

        img_resized_0_255 = img_resized_0_255.astype(np.float32)
        img_resized_0_255[img_resized_0_255 < 0] = 0
        img_resized_0_255[img_resized_0_255 > 255] = 255

        return img_resized_0_255

    @staticmethod
    def convert_to_0mean_1std_chw(img_chw, min_value=0.0, max_value=255.0):
        img_chw = img_chw.copy()
        for cycle_id in range(img_chw.shape[0] // 3):
            idx_from = 3 * cycle_id
            idx_to = 3 * (cycle_id + 1)
            img_chw[idx_from:idx_to] -= min_value
            img_chw[idx_from:idx_to] /= max_value
            img_chw[idx_from:idx_to] -= BlenderDataset.mean
            img_chw[idx_from:idx_to] /= BlenderDataset.std

        return img_chw

    @staticmethod
    def unnormalize_loss(loss):
        loss = loss * BlenderDataset.std.mean()
        loss *= 255.0
        return loss

    @staticmethod
    def get_limits(usr_mask_0_255, axis=0):
        usr_mask_0_255_min_x = usr_mask_0_255[..., 0].sum(axis=axis)
        usr_mask_0_255_min_x = np.argwhere(usr_mask_0_255_min_x > 0).flatten()
        min_x = usr_mask_0_255_min_x.min()
        max_x = usr_mask_0_255_min_x.max()

        return min_x, max_x

    @staticmethod
    def estimate_transparency_masks(fmo_mask_resized_0_255, low_threshold=1, high_threshold=200):
        """
        Given fmo_mask_resized and thresholds, finds a training mask (a mask where the fmo_mask_resized values are between the low_threshold
        and the high_threshold) and the no_chance_mask where there is almost none signal from the background image left.
        Args:
            fmo_mask_resized_0_255: mask of an FMO (with transparency values)
            low_threshold:
            high_threshold:

        Returns: (tuple) mask_0_1, no_chance_fmo_mask_0_1

        """
        assert len(fmo_mask_resized_0_255.shape) == 3
        mask_0_1 = np.all(
            np.concatenate(
                (low_threshold < fmo_mask_resized_0_255[..., :1],
                 fmo_mask_resized_0_255[..., :1] < high_threshold),
                axis=-1
            ), axis=-1
        )[..., np.newaxis]
        mask_0_1 = np.stack((mask_0_1, mask_0_1, mask_0_1), axis=-1)

        no_chance_fmo_mask_0_1 = fmo_mask_resized_0_255[..., :1] >= high_threshold
        no_chance_fmo_mask_0_1 = np.stack(
            (no_chance_fmo_mask_0_1, no_chance_fmo_mask_0_1, no_chance_fmo_mask_0_1),
            axis=-1)

        return mask_0_1, no_chance_fmo_mask_0_1

    def collect_timestep_images_and_masks(self, timestep_image_extraction_function,
                                          GT_imgs, timestep_paths):
        timestep_images_m1p1 = []
        alpha_channels_0_1_1c = []
        alpha_channels_0_1 = []
        tsr_steps = self.config['temporal_super_resolution_steps']
        step = int(tsr_steps / self.config['temporal_super_resolution_mini_steps'])
        for selected_range in [range(a, b) for a, b in zip(range(0, tsr_steps, step),
                                                           range(step, tsr_steps + step, step))]:
            selected_alpha_channels_0_1_1c = []
            selected_timestep_images_0_255 = []
            for selected_idx in selected_range:
                timestep_image_0_255 = \
                    timestep_image_extraction_function(GT_imgs, timestep_paths, selected_idx)
                if timestep_image_0_255.dtype == np.uint16:
                    timestep_image_0_255 = timestep_image_0_255 * (255.0 / 65535.0)
                alpha_channel_0_1_1c = timestep_image_0_255[..., -1:] / 255.0
                alpha_channels_0_1_1c.append(alpha_channel_0_1_1c)
                selected_alpha_channels_0_1_1c.append(alpha_channel_0_1_1c)
                selected_timestep_images_0_255.append(timestep_image_0_255)

            selected_alpha_channels_0_1_1c = \
                np.stack(selected_alpha_channels_0_1_1c, axis=0).mean(0)
            alpha_channels_0_1.append(selected_alpha_channels_0_1_1c)
            selected_timestep_images_0_255 = \
                np.stack(selected_timestep_images_0_255, axis=0).mean(0)
            timestep_image_0_255 = selected_timestep_images_0_255.copy()
            # timestep_image_0_255[..., :3] = \
            #     timestep_image_0_255[..., :3] * selected_alpha_channels_0_1_1c
            # selected_timestep_images_0_255[..., :3] * selected_alpha_channels_0_1_1c \
            # + bg_img_0_255 * (1.0 - selected_alpha_channels_0_1_1c)
            timestep_image_resized = self.preprocess_image(timestep_image_0_255.astype(np.uint8))
            timestep_image_resized = timestep_image_resized.transpose((2, 0, 1)).astype(np.float32)
            timestep_image_resized_m1p1 = self.from_255_to_m1p1(timestep_image_resized)
            timestep_images_m1p1.append(timestep_image_resized_m1p1)

        mask_bool_1c = np.concatenate(alpha_channels_0_1_1c, axis=-1).mean(axis=-1, keepdims=True)
        mask_bool_1c[mask_bool_1c > 0] = 255.0
        mask_bool_1c = self.preprocess_image(mask_bool_1c)
        mask_bool_1c = mask_bool_1c.transpose((2, 0, 1)).astype(np.float32)
        mask_bool_1c_0_1 = mask_bool_1c / 255.0

        for alpha_channel_idx in range(len(alpha_channels_0_1)):
            alpha_channels_0_1[alpha_channel_idx] = \
                self.preprocess_image(alpha_channels_0_1[alpha_channel_idx])
        alpha_channels_0_1 = np.concatenate(alpha_channels_0_1, axis=-1) * 255.0
        alpha_channels_0_1 = alpha_channels_0_1.transpose((2, 0, 1)).astype(np.float32) / 255.0

        return alpha_channels_0_1, mask_bool_1c_0_1, timestep_images_m1p1

    def preprocess_image(self, target_img):
        if self.transform is None:
            wh = self.ds_config['image_size']
            target_img_resized = cv2.resize(target_img, (wh, wh))
        else:
            pil_image = Image.fromarray(target_img)
            target_img_resized = np.array(self.transform(pil_image))

        if len(target_img_resized.shape) == 2:
            target_img_resized = target_img_resized[..., np.newaxis]

        return target_img_resized[..., ::-1]

    def preprocess_img_blurry_and_bg_img(self, imgs):
        imgs_out = []
        for img in imgs:
            img_resized_0_255 = self.preprocess_image(img)
            img_resized_0_255 = img_resized_0_255.transpose((2, 0, 1)).astype(np.float32)
            imgs_out.append(self.from_255_to_m1p1(img_resized_0_255))

        return imgs_out


class BlenderShapeNetGeneratedDataset(BlenderDataset):
    def __init__(self, config,
                 ds_config: conf_mgt.Default_Conf, cache_dir: str = None,
                 world_size: int = 1,
                 global_rank: int = 0, transform=None, is_trn=False):
        self.config = config
        self.ds_config = ds_config
        self.cache_dir = cache_dir
        self.transform = transform
        self.is_trn = is_trn

        paths_subconfig = get_path_subconfig_name(self.ds_config)

        class_dirs = []
        for dir_record in os.scandir(self.ds_config['paths'][paths_subconfig]['data_dir']):
            class_dirs.append(dir_record.path)

        self.class_dirs = class_dirs

        self.world_size = world_size

        logger.info('< preparing %s dataset >', 'TRN' if self.is_trn else 'VAL')

        data_tuples = []
        for class_dir in tqdm(class_dirs):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                if os.path.isdir(filepath):
                    continue

                stem, ext = os.path.splitext(filename)
                if ext == '.log':
                    continue

                success = True
                bg_path = os.path.join(class_dir, 'GT', stem, 'bgr.png')
                if not os.path.isfile(bg_path):
                    logger.warning('< missing bg for %s >', filepath)
                    success = False

                timestep_paths = []
                for i in range(self.config['temporal_super_resolution_steps']):
                    timestep_paths.append(
                        os.path.join(class_dir, 'GT', stem, f'image-{i + 1:06d}.png'))
                    if not os.path.isfile(timestep_paths[-1]):
                        logger.warning('< missing %d.timestep for %s >', i + 1, filepath)
                        success = False

                if not success:
                    continue

                data_tuples.append((filepath, bg_path, timestep_paths))

                if 'debug' in ds_config and ds_config['debug'] and len(data_tuples) > 50:
                    break

        self.data_tuples = []
        for data_tuple_idx in range(global_rank, len(data_tuples), world_size):
            self.data_tuples.append(data_tuples[data_tuple_idx])

        logger.info('< there is %d samples in the %s dataset >', len(self.data_tuples),
                    'TRN' if self.is_trn else 'VAL')
        print('< there is %d samples in the %s dataset >' % (
            len(self.data_tuples), 'TRN' if self.is_trn else 'VAL'))

    def __getitem__(self, index) -> T_co:
        image_filepath, bg_filepath, timestep_paths = self.data_tuples[index]

        img_0_255 = cv2.imread(image_filepath)
        bg_img_0_255 = cv2.imread(bg_filepath)

        bg_img_m1p1, img_resized_m1p1 = \
            self.preprocess_img_blurry_and_bg_img([bg_img_0_255, img_0_255])

        def timestep_image_extraction(_, timestep_paths, selected_idx):
            return cv2.imread(timestep_paths[selected_idx], cv2.IMREAD_UNCHANGED)

        alpha_channels_0_1, mask_bool_1c_m1p1, timestep_images_m1p1 = \
            self.collect_timestep_images_and_masks(timestep_image_extraction,
                                                   None, timestep_paths)

        return (img_resized_m1p1,
                bg_img_m1p1,
                alpha_channels_0_1,
                timestep_images_m1p1,
                [],
                {})

    def __len__(self):
        return len(self.data_tuples)

    @staticmethod
    def unnormalize_loss(loss):
        loss = loss * BlenderShapeNetGeneratedDataset.std.mean()
        loss *= 255.0
        return loss


class BlenderDatasetHdf5(BlenderDataset):
    def __init__(self):
        self._hdf5_file = None
        self.dataset_filepath = None
        self.ds_config = {'paths': None}

    @property
    def hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.dataset_filepath, 'r')

        return self._hdf5_file


class BlenderGeneratedDatasetHdf5(BlenderDatasetHdf5):
    def __init__(self, config,
                 ds_config: conf_mgt.Default_Conf, world_size: int = 1, global_rank: int = 0,
                 transform=None, is_trn: bool = False, dataset_size: int = 5000):
        super().__init__()
        self.config = config
        self.ds_config = ds_config
        self.transform = transform
        self.is_trn = is_trn
        self.dataset_filepath = ds_config['paths'][get_path_subconfig_name(self.ds_config)][
            'data_filepath']
        self.dataset_size = dataset_size
        self._samples_requested = 0

        logger.info('< dataset_filepath: %s >', self.dataset_filepath)

        self.world_size = world_size
        self.global_rank = global_rank

        logger.info('< preparing %s dataset >', 'TRN' if self.is_trn else 'VAL')

        self.data_tuples = self.sample_tuples()

        self._hdf5_file = None
        logger.info('< there is %d samples in the %s dataset >', len(self.data_tuples),
                    'TRN' if self.is_trn else 'VAL')

    def sample_tuples(self):
        sample_size_per_class = self.dataset_size // len(self.hdf5_file)
        data_tuples = []
        for class_name in tqdm(self.hdf5_file):
            valid_filenames = list(self.hdf5_file[class_name])
            filename_list = random.choices(valid_filenames, k=sample_size_per_class)
            if not self.is_trn:
                filename_list = valid_filenames

            for filename_id, filename in enumerate(filename_list):
                data_tuples.append((class_name, filename))

            if 'debug' in self.config and self.config['debug'] and len(data_tuples) > 50:
                break

        world_data_tuples = []
        for data_tuple_idx in range(self.global_rank, len(data_tuples), self.world_size):
            world_data_tuples.append(data_tuples[data_tuple_idx])

        return world_data_tuples

    def __getitem__(self, index) -> T_co:
        class_name, filename = self.data_tuples[index]

        filenames = list(self.hdf5_file[class_name])
        second_bg_filename = filenames[np.random.randint(0, len(filenames))]
        img_0_255 = np.array(self.hdf5_file[class_name][filename]['im'], dtype=np.float32)
        bg_img_0_255 = np.array(self.hdf5_file[class_name][filename]['bgr'], dtype=np.float32)
        bg2_img_0_255 = np.array(self.hdf5_file[class_name][second_bg_filename]['bgr'],
                                 dtype=np.float32)

        bg_img_m1p1, bg2_img_m1p1, img_resized_m1p1 = \
            self.preprocess_img_blurry_and_bg_img([bg_img_0_255, bg2_img_0_255, img_0_255])

        def timestep_image_extraction(GT_imgs, timestep_paths, selected_idx):
            return np.array(GT_imgs[timestep_paths[selected_idx]])

        GT_imgs = self.hdf5_file[class_name][filename]['GT']
        timestep_paths = list(GT_imgs.keys())

        alpha_channels_0_1, mask_bool_1c_0_1, timestep_images_m1p1 = \
            self.collect_timestep_images_and_masks(timestep_image_extraction,
                                                   GT_imgs, timestep_paths)

        composed_fg_0_1 = (np.mean(np.stack(timestep_images_m1p1, axis=0), axis=0) + 1.0) / 2.0
        imgs_resized_m1p1 = \
            [BlenderDataset.compose(composed_fg_0_1[1:], (bg_img_m1p1 + 1.0) / 2.0,
                                    composed_fg_0_1[:1]),
             BlenderDataset.compose(composed_fg_0_1[1:], (bg2_img_m1p1 + 1.0) / 2.0,
                                    composed_fg_0_1[:1])
             ]
        imgs_resized_m1p1 = [img * 2.0 - 1.0 for img in imgs_resized_m1p1]
        for k in range(len(timestep_images_m1p1)):
            alpha = alpha_channels_0_1[k:k+1]
            # timestep_fmo_0_2 = (1.0 - alpha) * (bg_img_m1p1 + 1.0) + alpha * (timestep_images_m1p1[k][1:] + 1.0)
            timestep_fmo_0_1 = BlenderDataset.compose((timestep_images_m1p1[k][1:] + 1.0) / 2.0,
                                                      (bg_img_m1p1 + 1.0) / 2.0,
                                                      alpha)
            timestep_images_m1p1[k][1:] = ((timestep_fmo_0_1 * 2.0) - (imgs_resized_m1p1[0] + 1.0)) / 2.0


        self._samples_requested += 1
        if self._samples_requested == self.dataset_size and self.is_trn:
            self.shuffle()

        return (imgs_resized_m1p1,
                bg_img_m1p1,
                mask_bool_1c_0_1,
                timestep_images_m1p1,
                [],
                {})

    def __len__(self):
        return len(self.data_tuples)

    def shuffle(self):
        self.data_tuples = self.sample_tuples()
        self._hdf5_file = None
        self._samples_requested = 0
