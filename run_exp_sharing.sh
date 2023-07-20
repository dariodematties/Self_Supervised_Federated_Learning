#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 720
#COBALT -n 2
#COBALT -A MultiActiveAI

export OMP_NUM_THREADS=24

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

EXP_DIRECTORY=/home/srajani/fed_ssl/Self_Supervised_Federated_Learning/
EXP_SCRIPT=src/experiments_sharing.py

CONTAINER=/lus/theta-fs0/software/thetagpu/nvidia-containers/pytorch/pytorch-22.04-py3.simg
MPI_BASE=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/

mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE --pwd $EXP_DIRECTORY $CONTAINER python $EXP_SCRIPT --supervision --dataset=cifar --model=resnet --local_ep=5