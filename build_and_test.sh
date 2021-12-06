#!/bin/bash

./build_and_install.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# rund tests
cd ../../messing_around
python3 test_plugin.py

cd ../dependencies/tensorflow