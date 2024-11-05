# tally-bench

This repository contains the benchmark scripts for our paper *"Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads."* The repository structure is organized as follows:

- `docker`: Contains Dockerfiles for building the Docker containers needed for the runtime environment.
- `python`: Holds benchmark scripts and configuration files.
- `scripts`: Includes scripts to run the experiments presented in the paper and to generate plots.
- `tally`: Contains the source code of Tally.
- `tally_results_demo`: Provides example figures generated during the artifact evaluation process.

## Required Hardware
To run these experiments, you'll need:
- An x86-64 Linux host with at least 85 GB of RAM.
- 200 GB of free disk space.
- An NVIDIA A100 GPU with 40 GB of memory.

Our test flow was conducted on a Google Cloud `a2-highgpu-1g` instance, which we recommend for verified performance.

## Installation and Setup

### Step 1: Install Docker

If Docker is not already installed, run the following commands:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
dockerd &
```

### Step 2: Clone the Repository
Clone this repository and navigate into it:
```
git clone https://github.com/tally-project/tally-bench.git
cd tally-bench
```

### Step 3: Pull the Docker Image
The Docker image required for the experiments is about 130 GB. Pull it by running:
```bash
docker pull wzhao18/tally:bench
```

### Step 4: Set Up Results Directory and Launch Docker Container
Create a directory to store the results and start the Docker container:
```bash
mkdir tally_results
docker run -it --shm-size=64g -v ${PWD}/tally_results:/home/tally-bench/tally_results wzhao18/tally:bench /bin/bash
```

## Running the Experiments
The experiments need to be run in two steps due to GPU mode requirements. Time-Slicing requires `DEFAULT` mode, while MPS requires `EXCLUSIVE_PROCESS`. These settings require sudo permissions.
1. Set GPU to `DEFAULT` mode and run the first set of experiments:
```bash
# On host machine
sudo nvidia-smi -i 0 -c DEFAULT

# In Docker container
./scripts/run_bench.sh > tally_results/b1.log 2>&1
```
2. 
```bash
# On host machine
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# In Docker container
./scripts/run_bench.sh > tally_results/b2.log 2>&1
```

## Parsing Results and Generating Plots
After the experiments complete, you can parse the results and generate plots. This can be done within the Docker container or on the host machine (which will require `pandas` and `matplotlib` installations if done locally).

Run the following commands:
```bash
python3 ./scripts/parse_results.py
python3 ./scripts/plot_results_micro.py
```
Both the tabular results and plots will be saved in the `tally_results` directory.
