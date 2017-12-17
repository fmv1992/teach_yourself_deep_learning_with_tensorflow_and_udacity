#! /bin/bash

export PYTHONPATH=$PYTHONPATH:$(readlink -f ../../)
echo $PYTHONPATH
../../virtual_environment/google_dl/bin/python3 ./1_notmnist.py
