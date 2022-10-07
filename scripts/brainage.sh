PROJECT_HOME=/data/stripeli/projectmetis/
cd $PROJECT_HOME
export PYTHONPATH=.

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda

/data/stripeli/metiscondaenvtf2/bin/python3 $PROJECT_HOME/simulatedFL/fedpurgemerge_main/brainage_main.py

