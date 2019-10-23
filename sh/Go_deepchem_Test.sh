#!/bin/bash
#
# https://deepchem.io/
# https://github.com/deepchem/deepchem
###
# Karst: https://kb.iu.edu/d/bezu
# Modules: https://kb.iu.edu/d/bcwy
###
#
cwd=$(pwd)
#
#
###
module list
module avail anaconda3
#
conda info
conda env list
#
###
# Activate DeepChem env.
source activate deepchem_env
#
conda info
conda list
#
###
# Test RDKit
${cwd}/python/rdk_fptest.py
#
###
# Test DeepChem...


###
# Deactivate DeepChem env.
source activate root
#
