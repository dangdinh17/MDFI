#!/usr/bin/env bash

# CUDA_HOME=/usr/local/cuda-10.2 \
# CUDNN_INCLUDE_DIR=/usr/local/cuda-10.2/include \
# CUDNN_LIB_DIR=/usr/local/cuda-10.2/lib64 \

export CUDA_HOME=/work/u9564043/cuda-11.7
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include
export CUDNN_LIB_DIR=$CUDA_HOME/lib64

# GCC
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

# (optional nhưng nên có)
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
# CONDA_PREFIX=$(python -c "import sys; print(sys.prefix)")

# # Thiết lập trỏ vào nội bộ môi trường ovqe
# export CUDA_HOME=$CONDA_PREFIX
# export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/include
# export CUDNN_LIB_DIR=$CONDA_PREFIX/lib
# # export PATH=$CONDA_PREFIX/bin:$PATH

# export CC=/usr/bin/gcc-11
# export CXX=/usr/bin/g++-11

python setup.py build_ext --inplace

if [ -d "build" ]; then
    rm -r build
fi
