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
        g++-10

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3.11 && \
    cp /usr/bin/python3.11 /usr/bin/python3

RUN cd /home && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

RUN rm /usr/bin/g++ && \
    ln -s /usr/bin/g++-10 /usr/bin/g++

RUN cd /home && \
    mkdir /opt/cmake && \
    wget https://cmake.org/files/v3.27/cmake-3.27.0-linux-x86_64.sh && \
    sh cmake-3.27.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

RUN cd /home && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz && \
    tar xvf boost_1_80_0.tar.gz && \
    cd boost_1_80_0 && \
    ./bootstrap.sh --prefix=/usr/ && \
    ./b2 install && \
    cd /home && \
    rm boost_1_80_0.tar.gz && \
    rm -rf boost_1_80_0

RUN cp /usr/include/cudnn*.h /usr/local/cuda/include && \
    cp -P /usr/lib/$(uname -m)-linux-gnu/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

WORKDIR /home/tally-bench

COPY . .

RUN cd tally/third_party && \
    cp cudnn-frontend /usr/local/cuda -r
 
RUN cd tally/third_party/nccl && make -j src.build
# RUN cd tally/third_party/nccl/ext-net/example && make
# RUN cd tally && mkdir -p build && cd build && cmake .. && make -j