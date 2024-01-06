import os
import random
from datetime import datetime

import socket

import torch

import conf_mgt
from benchmark.benchmark_loader import *
from benchmark.loaders_helpers import *
import argparse

from guided_diffusion import dist_util
from guided_diffusion.blender_generated import BlenderDataset
from guided_diffusion.script_util import create_model_and_diffusion, select_args, \
    model_and_diffusion_defaults
from utils import yamlread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

g_data_dir = '.'

g_dataset_dir = os.path.join(g_data_dir, 'datasets')


def renders2traj(renders, device):
    masks = renders[:, :, 0]
    sumx = torch.sum(masks, -2)
    sumy = torch.sum(masks, -1)
    cenx = torch.sum(sumy * torch.arange(1, sumy.shape[-1] + 1)[None, None].float().to(device),
                     -1) / torch.sum(sumy, -1)
    ceny = torch.sum(sumx * torch.arange(1, sumx.shape[-1] + 1)[None, None].float().to(device),
                     -1) / torch.sum(sumx, -1)
    est_traj = torch.cat((cenx.unsqueeze(-1), ceny.unsqueeze(-1)), -1)
    return est_traj


def parse_bool(arg):
    return bool(int(arg))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tbd_path", default=os.path.join(g_dataset_dir, 'TbD'), required=False)
    parser.add_argument("--tbd3d_path", default=os.path.join(g_dataset_dir, 'TbD-3D'),
                        required=False)
    parser.add_argument("--falling_path", default=os.path.join(g_dataset_dir, 'falling_objects'),
                        required=False)
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--visualization_path",
                        default=os.path.join('.', 'visualization'), required=False)
    parser.add_argument("--save_visualization", default=True, required=False)
    parser.add_argument("--add_traj", action='store_true', default=True)
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument("--visualize", type=parse_bool, default=True, required=False)

    return parser.parse_args()


def apply_color_map(grayscale_image):
    from matplotlib import cm
    cm = cm.get_cmap('viridis')
    return cm(grayscale_image)[..., :3] * 255


def create_difference_image_and_colorbar(source, target, max_val=30):
    diff = np.abs(source.astype(np.float32) - target.astype(np.float32))
    diff = np.mean(diff, axis=-1, keepdims=True)
    diff = np.concatenate([diff, diff, diff], axis=-1)
    max_val_constant = (255. / float(max_val))
    diff *= max_val_constant  # max je 30, cokoliv pod je relativne
    diff[diff > 255] = 255
    diff_max = diff.max() + 10 ** -7
    diff /= diff_max

    colorbar = apply_color_map(
        np.repeat(np.linspace(255, 0, diff.shape[0], dtype=np.uint8).reshape(-1, 1), 20, axis=1))
    colorbar = np.rot90(colorbar).copy()
    cv2.putText(colorbar, '{:.1f}'.format(diff_max / max_val_constant), (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return apply_color_map(diff[..., -1])[..., ::-1], np.fliplr(np.flipud(np.rot90(colorbar)))[...,
                                                      ::-1]


def main():
    args = parse_args()

    sidefmo_args = vars(args)
    for arg_name, arg_value in sidefmo_args.items():
        logger.info('%s="%s"', arg_name, str(arg_value))

    gpu_id = 0
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(sidefmo_args.get('conf_path')))
    conf.update({'is_tst': True})

    torch.manual_seed(0 + conf.seed)
    random.seed(0 + conf.seed)
    np.random.seed(0 + conf.seed)

    experiment_dir = os.path.join('.', 'output')
    model_filename = 'siddpmfmo.pt'
    model_name, _ = os.path.splitext(model_filename)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    file_handler = logging.FileHandler(
        os.path.join(experiment_dir,
                     'fmo_benchmark_' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S-%f')
                     + '_' + model_name
                     + '.log'))
    logger.addHandler(file_handler)

    g_resolution_x = g_resolution_y = conf.image_size
    t_start = conf.diffusion_steps - 1
    t_end = 0

    model, diffusion = create_model_and_diffusion(
        conf=conf,
        **select_args(conf, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(os.path.join('.', 'models', model_filename), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return model(x, t, y if conf.class_cond else None, gt=gt)

    def deblur_siddpmfmo(I_0_1, bbox_tight, nsplits, radius, kk, ff):
        bbox = extend_bbox(bbox_tight.copy(), 4 * np.max(radius), g_resolution_y / g_resolution_x,
                           I_0_1.shape)
        im_crop_0_1 = crop_resize(I_0_1, bbox, (g_resolution_x, g_resolution_y))
        img_blurry_0_1 = torch.tensor(im_crop_0_1).to(device).unsqueeze(0).float()
        with torch.no_grad():
            batch_size = img_blurry_0_1.shape[0]
            img_blurry_m1p1 = img_blurry_0_1.permute((0, 3, 1, 2)) * 2.0 - 1.0

            steps = int(conf.temporal_super_resolution_mini_steps / nsplits)

            previous_level = [img_blurry_m1p1.to(device)]
            for level in range(1):
                current_level = []
                for sample_id in range(2 ** level):
                    model_kwargs = {"gt": previous_level[sample_id],
                                    "x_km1": None,
                                    "bg_med": None}

                    sample_fn = (
                        diffusion.p_sample_single
                    )

                    result = sample_fn(
                        model_fn,
                        (batch_size, conf.out_channels, conf.image_size, conf.image_size),
                        clip_denoised=conf.clip_denoised,
                        model_kwargs=model_kwargs,
                        cond_fn=None,
                        device=device,
                        progress=conf.show_progress,
                        t_start=t_start,
                        t_end=t_end
                    )

                    samples = torch.split(result['sample'], 4, dim=1) # 96 = 24 * 4
                    samples = [(samples[k] + samples[k + 1] + samples[k + 2]) / 3.0
                               if steps == 3 else
                               (samples[k] + samples[k + 1]) / 2.0
                               for k in range(0, len(samples), steps)]
                    current_level.extend(samples)
                alphas_concat = 0
                for elem in current_level:
                    alphas_concat += (elem[:, :1] + 1.0) / 2.0
                alphas_concat = alphas_concat / len(current_level)

            result['sample'] = torch.cat(current_level, dim=1)

            rgba_bg_m1p1 = result['sample'][0].permute((1, 2, 0))
            rgba_bg_0_1 = (rgba_bg_m1p1 + 1.0) / 2.0

            rgba_0_1 = rgba_bg_0_1.reshape(list(rgba_bg_0_1.shape[:2]) + [nsplits, 4])
            rgba_0_1 = rgba_0_1.permute((0, 1, 3, 2))

            # shape 256x256x4x8
            alphas_concat = alphas_concat.permute((2, 3, 0, 1))

            rest, sequence_name = os.path.split(ff)
            rest, _ = os.path.split(rest)
            _, dataset_name = os.path.split(rest)
            mask_dir = os.path.join(args.visualization_path, dataset_name + '_eval', sequence_name,
                                    'masks')
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)
            img_dir = os.path.join(args.visualization_path, dataset_name + '_eval',
                                   sequence_name, 'imgs')
            if not os.path.isdir(img_dir):
                os.makedirs(img_dir)
            for render_idx in range(rgba_0_1.shape[-1]):
                alpha_np = rgba_0_1[..., :1, render_idx].cpu().detach().numpy()
                alpha = np.clip(alpha_np * 255, 0, 255).astype(np.uint8)
                alpha = rev_crop_resize(
                    np.concatenate((alpha, alpha, alpha), axis=-1)[..., np.newaxis], bbox, I_0_1)
                cv2.imwrite(os.path.join(mask_dir, f'{kk:04d}mask_{render_idx:02d}.png'), alpha[..., 0])

            bg_0_1 = img_blurry_0_1[0]

            # est_hs_crop_0_1 = \
            #     rgba2hs_residual(alphas_concat,
            #                      (rgba_0_1 * 2.0) - 1.0,
            #                      bg_0_1).data.cpu().detach().numpy()
            est_hs_crop_0_1 = rgba2hs(rgba_0_1, bg_0_1).data.cpu().detach().numpy()
            est_hs_crop_0_1 = est_hs_crop_0_1.clip(0, 1.0)
            est_hs_0_1 = rev_crop_resize(est_hs_crop_0_1, bbox, I_0_1)
            est_hs_0_1 = np.clip(est_hs_0_1, 0, 1)
            for render_idx in range(est_hs_0_1.shape[-1]):
                cv2.imwrite(os.path.join(img_dir, f'{kk:04d}img_{render_idx:02d}.png'),
                            np.clip(est_hs_0_1[..., ::-1, render_idx] * 255, 0, 255).astype(np.uint8))

            rgba_0_1_to_traj = rgba_0_1[np.newaxis].permute((0, 4, 3, 1, 2))
            est_traj = renders2traj(rgba_0_1_to_traj, device)[0].T.cpu()
            est_traj = rev_crop_resize_traj(est_traj, bbox, (g_resolution_x, g_resolution_y))

        return est_hs_0_1, est_traj

    args.method_name = 'SI-DDPM-FMO'
    run_benchmark(args, deblur_siddpmfmo)


if __name__ == "__main__":
    main()
