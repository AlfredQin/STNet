# STNet
This repository is an official implementation of the paper [A Spatial-Temporal Deformable Attention based
Framework for Breast Lesion Detection in Videos]. (MICCAI-2023)
 ![STNet](./figs/paper1755_fig_overal_arcitecture.png)

## Abstract
Detecting breast lesion in videos is crucial for computer-
aided diagnosis. Existing video-based breast lesion detection approaches
typically perform temporal feature aggregation of deep backbone fea-
tures based on the self-attention operation. We argue that such a strat-
egy struggles to effectively perform deep feature aggregation and ig-
nores the useful local information. To tackle these issues, we propose a
spatial-temporal deformable attention based framework, named STNet.
Our STNet introduces a spatial-temporal deformable attention module
to perform local spatial-temporal feature fusion. The spatial-temporal
deformable attention module enables deep feature aggregation in each
stage of both encoder and decoder. To further accelerate the detection
speed, we introduce an encoder feature shuffle strategy for multi-frame
prediction during inference. In our encoder feature shuffle strategy, we
share the backbone and encoder features, and shuffle encoder features
for decoder to generate the predictions of multiple frames. The exper-
iments on the public breast lesion ultrasound video dataset show that
our STNet obtains a state-of-the-art detection performance, while oper-
ating twice as fast inference speed. 

## Usage
## Installation
### Requirements
* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n STNet python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate STNet
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Dataset preparation

The dataset is provided by [CVA-Net](http://arxiv.org/abs/2207.00141), which is available for only non-commercial use in
research or educational purpose. 
As long as you use the database for these purposes, you can edit or process images and annotations in this database.
Please contact the authors of [CVA-Net](http://arxiv.org/abs/2207.00141) for getting the access to the dataset.
```
code_root/
└── miccai_buv/
      ├── rawframes/
      ├── train.json
      └── val.json
```

### Training

#### Training on single node

For example, the command for training CVA-NET on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/configs.sh
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> CVA-Net 8 configs/configs.sh
```

### Testing
We provide a trained model on the validation set. You can download it from
[here](https://drive.google.com/file/d/1G7CaP9R_fx96-iCx5K7Py_cW-4I0bwNU/view?usp=sharing) and put it in `./checkpoints/`.
Then you can test it by running the following command:

```bash
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/test.sh 
```

## Notes
The code of this repository is built on https://github.com/jhl-Det/CVA-Net. We thank the authors of
[CVA-Net](http://arxiv.org/abs/2207.00141) for their great work.