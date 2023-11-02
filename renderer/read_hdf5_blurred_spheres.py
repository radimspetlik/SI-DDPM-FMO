import os

import h5py
import numpy as np

from matplotlib import pyplot as plt


def read_spheres_3(dataset_name):
    with h5py.File(os.path.join(dataset_dir, dataset_name), 'r') as hdf5_file:
        for group in hdf5_file['sphere']:
            for object in hdf5_file['sphere'][group]:
                # ['GT', 'bgr', 'bgr_med', 'im']
                #
                print(f'{group} {object}')
                print(hdf5_file['sphere'][group][object].keys())
                object_group = hdf5_file['sphere'][group][object]
                im_0_1 = np.array(object_group['im'])

                # plt.imshow(im_0_1)
                # plt.show()
                for gt_image_key in object_group['GT'].keys():
                    im_0_1 = np.array(object_group['GT'][gt_image_key])
                    plt.imshow(im_0_1)
                    plt.show()
                    exit(0)


def read_spheres(dataset_name):
    with h5py.File(os.path.join(dataset_dir, dataset_name), 'r') as hdf5_file:
        for group in hdf5_file['sphere']:
            # ['GT', 'bgr', 'bgr_med', 'im']
            print(f'{group}')
            print(hdf5_file['sphere'][group].keys())
            object_group = hdf5_file['sphere'][group]
            im_0_1 = np.array(object_group['im'])

            # plt.imshow(im_0_1)
            # plt.show()
            for gt_image_key in object_group['GT'].keys():
                im_0_1 = np.array(object_group['GT'][gt_image_key])
                plt.imshow(im_0_1)
                plt.show()
                exit(0)


if __name__ == '__main__':
    dataset_dir = os.path.join('/', os.sep, 'mnt', 'lascar', 'rozumden', 'dataset')
    dataset_name = 'BlurredSpheres3Frames20.hdf5'
    # read_spheres_3(dataset_name)

    dataset_name = 'BlurredSpheres20.hdf5'
    read_spheres(dataset_name)
