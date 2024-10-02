#!/bin/bash

N_JOBS=$1
if [ -z $N_JOBS ]; then
    N_JOBS=6
fi

#bazel build --local_ram --config=dbg resources=4096 --subcommands //tensorflow/tools/pip_package:build_pip_package --verbose_failures
# bazel build -j 6 --config=dbg //tensorflow/tools/pip_package:build_pip_package #--verbose_failures
bazel build -j $N_JOBS   //tensorflow/tools/pip_package:build_pip_package --verbose_failures
if [ $? -ne 0 ]; then
    exit 1
fi

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
if [ $? -ne 0 ]; then
    exit 1
fi

# determine the numpy version that used to build
np_version=`python -c "import numpy as np; print(np.__version__)"`

pip install --force-reinstall  /tmp/tensorflow_pkg/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl && pip install numpy==$np_version
