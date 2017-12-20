#! /bin/bash

# Add 'google_course_library' to python path.
export PYTHONPATH=$PYTHONPATH:$(readlink -f ../../)

../../virtual_environment/google_dl/bin/python3 ./4_convolutions.py
