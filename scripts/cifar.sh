PROJECT_HOME=/data/stripeli/projectmetis/
cd $PROJECT_HOME
export PYTHONPATH=.

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda

#/data/stripeli/metiscondaenvtf2/bin/python3 /data/stripeli/projectmetis/simulatedFL/fedinit_main/cifar_main.py
#/data/stripeli/metiscondaenvtf2/bin/python3 /data/stripeli/projectmetis/simulatedFL/fedpurgemerge_main/cifar10_main_cnn.py
/data/stripeli/metiscondaenvtf2/bin/python3 /data/stripeli/projectmetis/simulatedFL/fedpurgemerge_main/cifar_main.py
#/data/stripeli/metiscondaenvtf2/bin/python3 /data/stripeli/projectmetis/simulatedFL/fedpurgemerge_main/prunefl/cifar_main.py

