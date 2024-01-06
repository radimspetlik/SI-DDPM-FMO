#!/usr/bin/bash

SCRIPTS_DIR="${HOME}/SI-DDPM-FMO/"

cd "${SCRIPTS_DIR}" || exit

source "$(pipenv --venv)/bin/activate"

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/image_train.py --conf_path confs/siddpmfmo.yml