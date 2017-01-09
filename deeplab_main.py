import tensorflow as tf
import numpy as np
import deeplab_model
from PIL import Image
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import pdb

# sample usage:
# python deeplab_main.py 0 single

def process_im(imname, mu):
  im = np.array(Image.open(imname), dtype=np.float32)
  if im.ndim == 3:
    if im.shape[2] == 4:
      im = im[:, :, 0:3]
    im = im[:,:,::-1]
  else:
    im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
  im -= mu
  im = np.expand_dims(im, axis=0)
  return im

if __name__ == "__main__":

  caffe_root = '/media/Work_HD/cxliu/tools/caffe/'
  mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
  mu = mu.mean(1).mean(1)
  pretrained_model = './model/ResNet101_init.tfmodel'

  model = deeplab_model.DeepLab()

  snapshot_restorer = tf.train.Saver()
  sess = tf.Session()
  snapshot_restorer.restore(sess, pretrained_model)

  if sys.argv[2] == 'single':
    im = process_im('example/2007_000129.jpg', mu)
    pred = sess.run(model.up, feed_dict={
              model.images  : im,
              model.H       : np.shape(im)[1],
              model.W       : np.shape(im)[2],
              model.labels  : np.zeros((1, 21)) # dummy
          })
    pred = np.argmax(pred, axis=3).squeeze().astype(np.uint8)
    seg = Image.fromarray(pred)
    seg.save('example/2007_000129.png')

  elif sys.argv[2] == 'pascal':
    pascal_dir = '/media/Work_HD/cxliu/datasets/VOCdevkit/VOC2012/JPEGImages/'
    list_dir = '/media/Work_HD/cxliu/projects/deeplab/list/'
    save_dir = 'example/val/'
    lines = np.loadtxt(list_dir + 'val_id.txt', dtype=str)
    for i, line in enumerate(lines):
      imname = line
      im = process_im(pascal_dir + imname + '.jpg', mu)
      pred = sess.run(model.up, feed_dict={
                model.images  : im,
                model.H       : np.shape(im)[1],
                model.W       : np.shape(im)[2],
                model.labels  : np.zeros((1, 21)) # dummy
            })
      pred = np.argmax(pred, axis=3).squeeze().astype(np.uint8)
      seg = Image.fromarray(pred)
      seg.save('example/val/' + imname + '.png')
      print('processing %d/%d' % (i + 1, len(lines)))