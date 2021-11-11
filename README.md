# g2net_ml_dl

# Workflows

## Installing the library

```
cd python
pip install .
```

Then you can import it into whatever package you need. This makes it convenient to use outside of just this repo.

- Downside: If you need to make changes to the library, you must reinstall the package for the new changes to work

## Full Workflow

```
git clone -b YOUR_BRANCH_NAME https://github.com/jchen42703/g2net_ml_dl
cd g2net_ml_dl/python
pip install .

# Then run your scripts from here or just import stuff from the library ^^.
```

If you're debugging a lot, I recommend:

```
import os
os.chdir("python")
import g2net
os.chdir("..")
os.getcwd()
```

A simple kernel restart will update your changes, so you won't need to reinstall the library with `pip install .`



## Using the Neural network on HPC

Move all of the files to related to the network onto HPC and then find the environment.yaml file.
There is an error in the environment.yaml file and you will need to vim into it and comment out
```
  - pytorch=1.9.0=py3.7_cuda11.1_cudnn8.0.5_0

```
And another to:
```
    - timm==0.4.12
```
To run it and set up the environement run the following commands:
```
module load miniconda3/4.9.2
conda env create -n kumaconda -f=environment.yaml
bash
conda activate kumaconda
conda install pytorch=1.9.0=py3.7_cuda11.1_cudnn8.0.5_0
```

Prep network with
```
python prep_data.py
```

Make sure that kuma_utils is up to date. In particular, the files:
```
kuma_utils/torch/modules/pooling.py
kuma_utils/torch/trainer.py
kuma_utils/torch/clip_grad.py
```
You may need to change the batch size and hardware in config.py:
```
HW_CFG = {
    'RTX3090': (16, 128, 1, 24), # CPU cores, RAM amount, GPU count, GPU RAM total
    'A100': (9, 60, 2, 40),
}

...

class Nspec23arch3(Nspec23):
    name = 'nspec_23_arch_3'
    model_params = Nspec23.model_params.copy()
    model_params['spec_params'] = dict(
        base_filters=128,
        kernel_sizes=(64, 16, 4),
    )
    batch_size = 16
    model_params['model_name'] = 'tf_efficientnet_b6_ns'
    transforms = Nspec22aug1.transforms.copy()


```

To train the network: 

```
python train.py --config Nspec23arch3 --gpu 0 1 --progress_bar

```

For output, we will use g2net-submission.ipynb, put it will need to be edited as our version will
be only using one part of this scheme. More specifically, we will have to edit 
```
prediction_list = [
    RESULT_DIR/'pseudo_12',
    RESULT_DIR/'pseudo_seq_04',
    RESULT_DIR/'pseudo_13',
    RESULT_DIR/'pseudo_14',
    RESULT_DIR/'pseudo_seq_07',
    RESULT_DIR/'pseudo_16',
    RESULT_DIR/'pseudo_17',
    RESULT_DIR/'pseudo_18',
    RESULT_DIR/'pseudo_19',
    RESULT_DIR/'pseudo_21',
    RESULT_DIR/'pseudo_23',
    # 
    RESULT_DIR/'pseudo_24',
    RESULT_DIR/'pseudo_10',
    RESULT_DIR/'pseudo_26',
    RESULT_DIR/'pseudo_25',
    RESULT_DIR/'pseudo_07',
    RESULT_DIR/'pseudo_22',
    RESULT_DIR/'pseudo_15',
    RESULT_DIR/'pseudo_seq_03',
    RESULT_DIR/'pseudo_06',
]
```
since we will not have these files, but only one of them.
