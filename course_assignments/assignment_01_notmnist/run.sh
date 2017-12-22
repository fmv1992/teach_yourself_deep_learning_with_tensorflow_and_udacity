#! /bin/bash

# Add 'google_course_library' to python path.
export PYTHONPATH=$PYTHONPATH:$(readlink -f ../../)

cd $(dirname $0)

# echo $VE_ROOT
# echo $PYTHON3_VE
# eval "$PYTHON3_VE -c \"print(99*'-')\""

eval "$PYTHON3_VE ./1_notmnist.py"
