import os
import random
from datetime import datetime

import socket

import torch
import torchvision

import conf_mgt
from benchmark.benchmark_loader import *
from benchmark.loaders_helpers import *
import argparse

from defmo.encoder import EncoderCNN
from defmo.rendering import RenderingCNN

from fmo_deblurring_benchmark import parse_args, renders2traj
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, select_args, \
    model_and_diffusion_defaults
from utils import yamlread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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
    model_filename = 'baseline.pt'
    model_name, _ = os.path.splitext(model_filename)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    file_handler = logging.FileHandler(
        os.path.join(experiment_dir,
                     'baseline_dddpm_defmo_benchmark_' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S-%f')
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

    encoder = EncoderCNN()
    rendering = RenderingCNN()

    if torch.cuda.is_available():
        encoder.load_state_dict(torch.load(os.path.join('.', 'models', 'encoder_best.pt')))
        rendering.load_state_dict(
            torch.load(os.path.join('.', 'models', 'rendering_best.pt')))
    else:
        encoder.load_state_dict(torch.load(os.path.join('.', 'models', 'encoder_best.pt'),
                                           map_location=torch.device('cpu')))
        rendering.load_state_dict(
            torch.load(os.path.join('.', 'models', 'rendering_best.pt'),
                       map_location=torch.device('cpu')))

    encoder = encoder.to(device)
    rendering = rendering.to(device)

    encoder.train(False)
    rendering.train(False)

    def get_transform():
        return torchvision.transforms.ToTensor()

    preprocess = get_transform()

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return model(x, t, y if conf.class_cond else None, gt=gt)

    def deblur_sidefmo(I_0_1, bbox_tight, nsplits, radius, kk, ff):
        bbox = extend_bbox(bbox_tight.copy(), 4 * np.max(radius), g_resolution_y / g_resolution_x,
                           I_0_1.shape)
        im_crop_0_1 = crop_resize(I_0_1, bbox, (g_resolution_x, g_resolution_y))
        img_blurry_0_1 = torch.tensor(im_crop_0_1).to(device).unsqueeze(0).float()
        with torch.no_grad():
            batch_size = img_blurry_0_1.shape[0]
            img_blurry_m1p1 = img_blurry_0_1.permute((0, 3, 1, 2)) * 2.0 - 1.0

            model_kwargs = {"gt": img_blurry_m1p1.to(device),
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

            bg_m1p1 = torch.split(result['sample'], 3, dim=1)  #

            im_crop_0_1_t = preprocess(im_crop_0_1).to(device)
            bgr_crop_0_1 = (bg_m1p1[0][0] + 1.0) / 2.0

            input_batch = torch.cat((im_crop_0_1_t, bgr_crop_0_1), 0).to(
                device).unsqueeze(0).float()

            latent = encoder(input_batch)
            times = torch.linspace(0, 1, nsplits * multi_f + 1).to(device)
            renders = rendering(latent, times[None])
            renders = renders[:, :-1].reshape(1, nsplits, multi_f, 4, g_resolution_y,
                                              g_resolution_x).mean(2)  # add small motion blur

            bgr_crop = bgr_crop_0_1.cpu().numpy().transpose((1, 2, 0))
            rgba_0_1 = renders[0].clone().permute((2, 3, 1, 0))
            renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2, 3, 1, 0)

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
                alpha_np = rgba_0_1[..., -1:, render_idx].cpu().detach().numpy()
                alpha = np.clip(alpha_np * 255, 0, 255).astype(np.uint8)
                alpha = rev_crop_resize(
                    np.concatenate((alpha, alpha, alpha), axis=-1)[..., np.newaxis], bbox, I_0_1)
                cv2.imwrite(os.path.join(mask_dir, f'{kk:04d}mask_{render_idx:02d}.png'),
                            alpha[..., 0])

            def rgba2hs(rgba, bgr):
                return rgba[:, :, :3] * rgba[:, :, 3:] + bgr[:, :, :, None] * (1 - rgba[:, :, 3:])

            est_hs_crop = rgba2hs(renders_rgba, bgr_crop)
            est_hs_0_1 = rev_crop_resize(est_hs_crop, bbox, I_0_1)

            for render_idx in range(est_hs_0_1.shape[-1]):
                cv2.imwrite(os.path.join(img_dir, f'{kk:04d}img_{render_idx:02d}.png'),
                            np.clip(est_hs_0_1[..., ::-1, render_idx] * 255, 0, 255).astype(np.uint8))

            est_traj = renders2traj(renders, device)[0].T.cpu()
            est_traj = rev_crop_resize_traj(est_traj, bbox, (g_resolution_x, g_resolution_y))

        return est_hs_0_1, est_traj

    args.method_name = 'BaselineDDPMDeFMO'
    run_benchmark(args, deblur_sidefmo)


if __name__ == "__main__":
    multi_f = 5  ## simulate small motion blur
    main()
