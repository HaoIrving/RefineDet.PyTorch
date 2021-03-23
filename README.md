Impletation of SANet: Semantic Attention-Based Network for Inshore SAR ship detection, ICDIP 2021.

### Table of Contents
<!-- - <a href='#performance'>Performance</a> -->
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

<!-- ## Performance

#### VOC2007 Test

##### mAP (*Single Scale Test*)

| Arch | Paper | Caffe Version | Our PyTorch Version |
|:-:|:-:|:-:|:-:|
| RefineDet320 | 80.0% | 79.52% | 79.81% |
| RefineDet512 | 81.8% | 81.85% | 80.50% | -->

## Installation
- CUDA 10.1
- Cudnn 7
- pip install visdom
- Install SOLO environment:
```shell
conda create -n solo python=3.7 -y
conda activate solo
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
git clone https://github.com/WXinlong/SOLO.git
cd SOLO
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
python setup.py develop # if build failed, try 'rm -r build', then use this command
```

## Datasets
### SSDD
```Shell
cd data
mkdir SSDD
cd SSDD
git clone https://github.com/HaoIrving/SSDD_coco.git
# or put SSDD_coco in other place, then ln -s /absolute/path/to/SSDD_coco /absolute/path/to/SANet/data/SSDD/SSDD_coco
```

## Training

```Shell
python train_sanet.py --num_workers 12 --lr 2e-3 --save_folder weights/tmp/ --batch_size 16
```

## Evaluation
To evaluate a trained network:

```Shell
python eval_sanet_coco.py --prefix weights/tmp
```
the code will filter out the best model among the last 20 models.

## Demo
To visualize the detected ships:

```Shell
python demo_sanet_coco.py --prefix weights/tmp --trained_model RefineDet512_COCO_epoches_280.pth

# if you want the detected picture be saved: 
python demo_sanet_coco.py --prefix weights/tmp --trained_model RefineDet512_COCO_epoches_280.pth --save_detected
```

