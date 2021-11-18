#!/bin/bash
#SBATCH -p class --gres=gpu:1 -A sxg125_csds438
#SBATCH -c 8
#SBATCH --mem=25gb

module swap intel gcc

module load cuda/11.2 python/3.8.6

nvidia-smi

export PYTHONUSERBASE=$HOME/.usr/local/python/3.8.6

# git clone https://github.com/jchen42703/g2net_ml_dl.git
# Change this to wherever you cloned g2net_ml_dl into
# For me, it was home
export SCRIPTS_DIR="${HOME}/final_project"

echo "g2net_ml_dl repo directory located in: ${SCRIPTS_DIR}/g2net_ml_dl"

cd $SCRIPTS_DIR/g2net_ml_dl/python

pip3 install --user .
pip3 install --user torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "job directory: $PFSDIR"

# changing to job directory
cd $PFSDIR

python3 "${SCRIPTS_DIR}/g2net_ml_dl/python/scripts/prep_data_csvs.py" --yml_path="${SCRIPTS_DIR}/g2net_ml_dl/python/scripts/prep_data.yml"
python3 "${SCRIPTS_DIR}/g2net_ml_dl/python/scripts/train_minirocket.py" --yml_path="${SCRIPTS_DIR}/g2net_ml_dl/python/scripts/train_minirocket.yml"
