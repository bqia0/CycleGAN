# CycleGAN
IE498 Final Project

[CycleGAN](https://junyanz.github.io/CycleGAN/) implementation using PyTorch

## Usage


### Datasets
You can download a dataset with `datasets/download_dataset.sh` (from the [official CycleGAN repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))
```
bash ./download_dataset.sh [name of dataset]
```

### Training
```
python ./main.py --train --dataset [name of dataset]
```

### Testing
You can generate a few test examples using the below command.
```
python ./main.py --test --dataset [name of dataset] --test_samples [number of test samples]
```
