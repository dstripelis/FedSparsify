#!/bin/bash

source global_variables.sh
cd "${PROJECT_HOME}"
export PYTHONPATH=.

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda

"${PROJECT_CONDA_ENVIRONMENT}"/python3 "${PROJECT_HOME}"/fedpurgemerge_main/fashion_mnist_main.py
#"${PROJECT_CONDA_ENVIRONMENT}"/python3 "${PROJECT_HOME}"/fedpurgemerge_main/prunefl/fashion_mnist_main.py
