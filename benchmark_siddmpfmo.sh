#!/usr/bin/bash

SCRIPTS_DIR="${HOME}/SI-DDPM-FMO/"

cd "${SCRIPTS_DIR}" || exit

source "$(pipenv --venv)/bin/activate"

python fmo_deblurring_benchmark.py --conf_path confs/siddpmfmo.yml