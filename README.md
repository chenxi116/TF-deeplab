# TF-deeplab

This is a Tensorflow implementation of [DeepLab](http://liangchiehchen.com/projects/DeepLab.html). 

Currently it only supports testing the ResNet 101 version by converting the caffemodel provided by Jay. Although supporting training should be quick and easy. 

Note that the current version is not multi-scale, i.e. only uses the original resolution branch and discarding all layers of 0.5 and 0.75 resolution.

The `caffemodel2npy.py` is modified from [here](https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/loadcaffe.py), and the `deeplab_model.py` is modified from [here](https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py).

## Example Usage
- Download the prototxt and caffemodel [provided by Jay](http://liangchiehchen.com/projects/DeepLabv2_resnet.html)
- Convert caffemodel to npy file
```bash
python caffemodel2npy.py deploy.prototxt ../deeplab/ResNet101/init.caffemodel ./model/ResNet101_init.npy
python caffemodel2npy.py deploy.prototxt ../deeplab/ResNet101/train_iter_20000.caffemodel ./model/ResNet101_train.npy
python caffemodel2npy.py deploy.prototxt ../deeplab/ResNet101/train2_iter_20000.caffemodel ./model/ResNet101_train2.npy
```
- Convert npy file to tfmodel
```bash
python npy2tfmodel.py 0 ./model/ResNet101_init.npy ./model/ResNet101_init.tfmodel
python npy2tfmodel.py 0 ./model/ResNet101_train.npy ./model/ResNet101_train.tfmodel
python npy2tfmodel.py 0 ./model/ResNet101_train2.npy ./model/ResNet101_train2.tfmodel
```
- Test on a single image
```bash
python deeplab_main.py 0 single
```
- Test on the PASCAL VOC2012 validation set (you will also want to look at the `matlab` folder after you run the following command)
```bash
python deeplab_main.py 0 pascal
```

## Performance

The converted DeepLab ResNet 101 model achieves mean IOU of 73.296% on the validation set of PASCAL VOC2012. Again, this is only with the original resolution branch, which is likely to be the reason for the performance gap (according to the [paper](https://arxiv.org/pdf/1606.00915.pdf) this number should be around 75%).

## TODO

- Incorporating 0.5 and 0.75 resolution
- Training code
