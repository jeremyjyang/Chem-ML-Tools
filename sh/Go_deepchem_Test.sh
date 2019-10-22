#!/bin/sh
#
# https://deepchem.io/
# https://github.com/deepchem/deepchem
###
# Karst: https://kb.iu.edu/d/bezu
# Modules: https://kb.iu.edu/d/bcwy
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
# Deactivate DeepChem env.
source activate root
#
