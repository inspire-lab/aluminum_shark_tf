#!/bin/bash

./build_and_install.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# run tests
python3 test_plugin.py
