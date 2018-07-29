#! /bin/bash

# Important: it has to be the first line.
cd $(dirname $0)

# Add 'google_course_library' to python path.
export PYTHONPATH=$PYTHONPATH:$(readlink -f ../../)
echo $PYTHONPATH

eval "$PYTHON3_VE ./5_word2vec.py"
