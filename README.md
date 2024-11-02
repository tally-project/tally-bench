# tally-bench

This repository contains the benchmark scripts for our paper "Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads". The structure of the repo is follows: 

## Artifact Evalution
We provide instructions for artifact evalution. The expected output of this evalution will look similar to [here](https://github.com/tally-project/tally-bench/tree/master/tally_results).

The required machine for this evalution is a Linux server with 1 NVIDIA A100 GPU. We demonstrate how to run the experiments inside docker containers.

[This folder](https://github.com/tally-project/tally-bench/tree/master/docker) provides several Dockerfiles. Specifically, `Dockerfile.base` illustrates the environment needed to build Tally and its dependencies. `Dockerfile.bench` prepares the benchmark environment, including the datasets and compiled models.

We provide a pre-built docker image named 'wzhao18/tally:bench'

## Steps

Install and launch docker if not already installed.
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
dockerd &
```

Pull docker image - this image is about 120GB of storage
```bash
docker pull wzhao18/tally:bench
```

Launch container with volumn. The results will be saved to the local directory `tally_results`.
```bash
mkdir tally_results
docker run -it --shm-size=64g -v ${PWD}/tally_results:/home/tally-bench/tally_results wzhao18/tally:bench /bin/bash
```

Inside the container, run the following commands to run the benchmark and generate the plots.
Note, the experiments need to be run in two steps due to the need to change the GPU mode (Time-Slicing requires DEFAULT while MPS requires EXCLUSIVE_PROCESS), which requires sudo permission.
```bash
sudo nvidia-smi -i 0 -c DEFAULT
./scripts/run_bench.sh > tally_results/bench-step1.log 2>&1

sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
./scripts/run_bench.sh > tally_results/bench-step2.log 2>&1

python3 ./scripts/plot_results_micro.py
```





