#!/bin/bash
# FILENAME: jobscript.sh

# Loads the required environment configuration
module purge
module load anaconda/2024.02-py311
module use /depot/leifur/etc/modules
module load conda-env/gilbreth_env_3_11-py3.11.7

# Change directory to the original directory
cd $SLURM_SUBMIT_DIR

# Run the python script
python3 env_model_function_rombo_multiple.py

