A higher performance [PyTorch](http://pytorch.org/) implementation of [RefineDet++: Single-Shot Refinement Neural Network for Object Detection](http://www.cbsr.ia.ac.cn/users/sfzhang/files/TCSVT_RefineDet++.pdf ).

### Table of Contents
- <a href='#major features'>Major features</a>
- <a href='#performance'>Performance</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-refinedet'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Major features
- Original RefineDet model.
- Align Convolution module proposed in RefineDet++ is implemented by DeformConv with accurately calculated offset rather than learned, which is a good solution for the feature misalignment problem among 1.5 stage object detection methods.
- Multi-scale test, which is modified from the original caffe implementation of RefineDet.
- VGG backbone with bn layers to make the training more stable.
- ResNet and ResNeXt backbone, which has been implemented fully, but the training is hard to converge for slightly large learning rate, sting working on this.

## Performance

#### SSDD (remote ship detection dataset of Radar images)

##### COCO AP 

| Arch | Our PyTorch Version |
|:-:|:-:|:-:|:-:|
| RefineDet++512 ms| 66.21% | 
| RefineDet++512 | 62.94% | 
| RefineDet512 | 62.57% | 
ms: multi scale test, we report the best results among many times run, so the results are convincing.

#### VOC2007 Test

##### mAP 

| Arch | Paper | Our PyTorch Version |
|:-:|:-:|:-:|:-:|
| RefineDet++512 | 82.5% | TODO |

## Installation
We use [MMDetection v2](https://mmdetection.readthedocs.io/) as our environment, since we need the Deformable Convolution implementation of it. Our results are derived with the latest version of MMDetection, other version is obscure.

## Datasets
We currently use SSDD in our lab, which is a remote ship detection dataset widely used. It can be found at my another repository (https://github.com/HaoIrving/SSDD_coco.git).

run following commands to settle the dataset:
```Shell
cd data
mkdir SSDD
cd SSDD
git clone https://github.com/HaoIrving/SSDD_coco.git
# or put SSDD_coco in other place, then ln -s /absolute/path/to/SSDD_coco /absolute/path/to/SANet/data/SSDD/SSDD_coco
```

<!-- 
### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
``` -->

## Training RefineDet++
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `RefineDet.PyTorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RefineDet320++ or RefineDet512++.

```Shell
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_512vggbn/ --ngpu 4  --model 512_vggbn --batch_size 16 

```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train_refinedet.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_coco.py --prefix weights/align_4e3_512vggbn  --model 512_vggbn
```

You can specify the parameters listed in the `eval_refinedet_coco.py` file by flagging them or manually changing them.  

## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [ ] Support for ResNet and ResNeXt backbone.

## References
- [Original RefineDet Implementation (CAFFE)](https://github.com/sfzhang15/RefineDet)
- Our code is built based on [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch) and [MMDetection](https://github.com/open-mmlab/mmdetection), many thanks to them.
