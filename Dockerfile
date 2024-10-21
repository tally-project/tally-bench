FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git \
        wget \
        libacl1-dev \
        libncurses5-dev \
        pkg-config \
        zlib1g \
        g++-10 \
        sudo \
        libssl-dev \
        vim \
        libfreeimage-dev

RUN apt remove python3.8 -y && \
    apt autoremove -y && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3.11-dev && \
    cp /usr/bin/python3.11 /usr/bin/python3

RUN cd /home && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

RUN pip3 install pyyaml typing_extensions

RUN rm /usr/bin/g++ && \
    rm /usr/bin/gcc && \
    ln -s /usr/bin/g++-10 /usr/bin/g++ && \
    ln -s /usr/bin/gcc-10 /usr/bin/gcc

RUN cd /home && \
    mkdir /opt/cmake && \
    wget https://cmake.org/files/v3.27/cmake-3.27.0-linux-x86_64.sh && \
    sh cmake-3.27.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

RUN cd /home && \
    git clone https://github.com/tally-project/pytorch.git && \
    cd pytorch && \
    git checkout v2.2.0-tally && \
    git submodule update --init --recursive

RUN cd /home/pytorch && \
    USE_CUDNN=0 MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="8.0"  \
        python3 setup.py develop \
        --install-dir $(python3 -c "import site; print(site.getsitepackages()[0])")

RUN cd /home && \ 
    git clone https://github.com/pytorch/vision.git && \
    cd vision && \
    git checkout v0.17.2 && \
    USE_CUDNN=0 MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="8.0"  \
        python3 setup.py develop \
        --install-dir $(python3 -c "import site; print(site.getsitepackages()[0])")

RUN apt install -y zlib1g-dev

RUN cd /home && \ 
    git clone https://github.com/tally-project/triton.git && \
    cd triton && \
    git checkout v2.1.0-tally && \
    cd python && \
    pip3 install cmake && \
    pip3 install -v -e .

RUN cd /home && \ 
    git clone https://github.com/tally-project/hidet.git && \
    cd hidet && \
    mkdir build && \
    cd build && \
    cp ../config.cmake .  && \
    cmake .. && \
    make -j4 && \
    cd .. && \
    pip3 install -e .

RUN cd /home && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz && \
    tar xvf boost_1_80_0.tar.gz

RUN cd /home/boost_1_80_0 && \
    ./bootstrap.sh --prefix=/usr/ && \
    ./b2 install && \
    cd /home && \
    rm boost_1_80_0.tar.gz && \
    rm -rf boost_1_80_0

RUN rm /usr/bin/python3.8* && \
    rm /usr/bin/python3 && \
    cp /usr/bin/python3.11 /usr/bin/python3

RUN cd /home && \
    git clone https://github.com/facebook/folly && \
    cd folly && \
    git checkout 6d79e8b && \
    sudo ./build/fbcode_builder/getdeps.py install-system-deps --recursive && \
    python3 ./build/fbcode_builder/getdeps.py --allow-system-packages build --no-tests --install-prefix /usr/local

RUN cp /usr/include/cudnn*.h /usr/local/cuda/include && \
    cp -P /usr/lib/$(uname -m)-linux-gnu/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

WORKDIR /home/tally-bench

COPY . .

RUN mkdir /etc/iceoryx && \
    cp tally/config/roudi_config.toml /etc/iceoryx/roudi

RUN cd tally/third_party && \
    cp cudnn-frontend /usr/local/cuda -r
 
RUN cd tally/third_party/nccl && make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80"
RUN cd tally && mkdir -p build && cd build && cmake .. && make -j