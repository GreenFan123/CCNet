### Requirements

To install PyTorch==0.4.0 or 0.4.1, please refer to https://github.com/pytorch/pytorch#installation.   
4 x 12G GPUs (_e.g._ TITAN XP)  
Python 3.6   
gcc (GCC) 4.8.5  
CUDA 8.0  

### Compiling

Some parts of **InPlace-ABN** and **Criss-Cross Attention** have native CUDA implementations, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py

cd ../cc_attention
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model

Plesae download cityscapes dataset and unzip the dataset into `YOUR_CS_PATH`.

Please download open source imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

### Training and Evaluation
Training script.
```bash
python train.py --data-dir ${YOUR_CS_PATH} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 1e-2 --input-size 769,769 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --recurrence 2
``` 

【**Recommend**】You can also open the OHEM flag to reduce the performance gap between val and test set.
```bash
python train.py --data-dir ${YOUR_CS_PATH} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 1e-2 --input-size 769,769 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000
``` 

Evaluation script.
```bash
python evaluate.py --data-dir ${YOUR_CS_PATH} --restore-from snapshots/CS_scenes_60000.pth --gpu 0 --recurrence 2
``` 

All in one.
```bash
./run_local.sh YOUR_CS_PATH
``` 
