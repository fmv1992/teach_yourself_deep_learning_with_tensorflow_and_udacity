#! /bin/bash


cd $(dirname $0)

export PYTHON3_VE=$VE_ROOT/google_dl/bin/python3

bash ./assignment_01_notmnist/run.sh

exit 1

bash ./assignment_02_stochastic_gradient_descent/run.sh

bash ./assignment_03_regularization/run.sh
