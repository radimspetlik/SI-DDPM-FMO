"""
Train a diffusion model on images.
"""

import argparse
import os
from multiprocessing import set_start_method
from shutil import copyfile

import conf_mgt
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_args,
)
from guided_diffusion.train_util import TrainLoop
from utils import yamlread


def main(conf: conf_mgt.Default_Conf):
    global_rank = dist_util.setup_dist()
    experiment_dir = os.getenv("EXPERIMENT_DIR") if os.getenv("EXPERIMENT_DIR") is not None else './experiments/debug/'

    logger.configure(dir=experiment_dir, global_rank=global_rank)
    logger.log(f'experiment_dir: {experiment_dir}')
    logger.log("creating model and diffusion...")

    conf_path = args.get('conf_path')
    if os.path.isdir(experiment_dir) and os.path.isfile(conf_path):
        _, conf_filename = os.path.split(conf_path)
        copyfile(conf_path, os.path.join(experiment_dir, conf_filename))

    model, diffusion = create_model_and_diffusion(conf=conf,
                                                  **select_args(
                                                      conf,
                                                      model_and_diffusion_defaults().keys()
                                                  ))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(conf.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    train_name = conf.get_default_train_name()
    data = conf.get_dataloader(dsName=train_name)

    logger.log("training...")
    TrainLoop(model=model,
              diffusion=diffusion,
              data=data,
              schedule_sampler=schedule_sampler,
              conf=conf
              ).run_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))

    set_start_method('forkserver', force=True)

    main(conf_arg)
