# DEFT: Exploiting Gradient Norm Difference between Model Layers for Scalable Gradient Sparsification

## Description
> ***DEFT*** is abbreviation of 'Distributed Execution of Fragmented Top-k'. DEFT partitions the gradient sparsification task into sub tasks and distributes them to workers. Benchmarks include: 1) image classification using CNNs, 2) language modelling using LSTMs `cnn-lstm`, and 3) recommendation using NCF `ncf`.

## How to setup

To install the necessary dependencies, create Conda environment using `environment.yml` by running the following commands. ***Note***: check compatibility of each package version such as `cudatoolkit` and `cudnn` with your device, e.g., NVIDIA Tesla V100 GPU is compatible.

```bash
$ conda env create --file environment.yml
$ conda activate deft
$ python -m spacy download en
$ conda deactivate deft
```

## How to start

The scripts to run code are written for SLURM workload manager. The source code supports distributed training with **multi-node and multi-GPU**. In `run.sh`, you can specify *model*, *dataset*, ***reducer***, and *world_size*.

### Overview of shell scripts

 - If you use **SLURM**, use `pararun` and modify it for your configuration. The script `pararun` executes `run.sh` in parallel. The script `run.sh` includes setup for distributed training.
 - If you do not use SLURM, you do not need to use `pararun`. Instead, run `run.sh` on your nodes, then rendezvous of pytorch allows processes are connected.

### CNN and LSTM benchmarks

 - If you use **SLURM**, use following command.
```bash
$ sbatch pararun
```
 - If you do not use SLURM, use following command on each node.
```bash
$ hostip=<ip> port=<port> mpirun -np <world_size> run.sh
```

### Neural Collaborative Filtering (NCF) benchmarks

#### 1. Prepare dataset

 - To download dataset, use following command.
```bash
$ ./prepare_dataset.sh
```

#### 2. Run training script

 - If you use **SLURM**, use following command.
```bash
$ sbatch pararun
```
 - If you do not use SLURM, use following command on each node.
```bash
$ hostip=<ip> port=<port> mpirun -np <world_size> run.sh
```

## Publication

If you use this code, please cite the following [**\[Paper\]**](https://arxiv.org/abs/2307.03500):
- **DEFT: Exploiting Gradient Norm Difference between Model Layers for Scalable Gradient Sparsification**. Daegun Yoon, Sangyoon Oh. ***ICPP 2023***, Aug. 2023.

```
@article{yoon2023deft,
  title={DEFT: Exploiting Gradient Norm Difference between Model Layers for Scalable Gradient Sparsification},
  author={Yoon, Daegun and Oh, Sangyoon},
  journal={arXiv preprint arXiv:2307.03500},
  year={2023}
}
```

## Contact

If you have any questions about this project, contact me by one of the followings:
- slashxp@naver.com
- kljp@ajou.ac.kr
