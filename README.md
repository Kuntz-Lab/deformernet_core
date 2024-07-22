# DeformerNet and Dense Predictor's architectures + training scripts

## Overview

Related documents:
- Public docs
    - [Data collection in Isaac Gym](https://github.com/Utah-ARMLab/deformernet_isaacgym)

## Prerequisite

1. Pull the repository.
```bash
git clone git@github.com:Utah-ARMLab/deformernet_core.git
```

**NOTE**: Make sure you have NVIDIA GPU(s) to train the models. 


## Single-arm DeformerNet
1. Process data.
```bash
cd single_deformernet
python3 process_data_single.py --obj_category {your_category}
```

2. Train.
```bash
python3 single_trainer_all_objects.py
```


## Bimanual DeformerNet
1. Process data.
```bash
cd bimanual_deformernet
python3 process_data_bimanual.py --obj_category {your_category}
```

2. Train.
```bash
python3 bimanual_trainer_all_objects.py
```


## Single-arm Dense Predictor
1. Process data.
```bash
cd learn_manipulation_points
python3 process_data_dense_predictor_single.py --obj_category {your_category}
```

2. Train.
```bash
python3 single_dense_predictor_trainer.py
```


## Bimanual Dense Predictor
1. Process data.
```bash
cd learn_manipulation_points
python3 process_data_dense_predictor_bimanual.py --obj_category {your_category}
```

2. Train.
```bash
python3 bimanual_dense_predictor_trainer.py
```


## Code format
Please run the following command to format the repo with the [black](https://github.com/psf/black) formatter.

```bash
bash format.sh
```
