'''
Read images from input dir, resize it and store as TFRecord or Images.

@author : srijith

'''
from __future__ import print_function

import os
import sys
import argparse
import utils
import tensorflow as tf
import random

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example_proto(image_buffer, label, height, width):
  '''
  Converts to example proto from TF

  Arguments:
  image_buffer : Image data
  label : class label
  height 
  width

  Returns:
  Example proto file
  '''
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
 
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      #'colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      #'channels': _int64_feature(channels),
      'label': _bytes_feature(tf.compat.as_bytes(label)),
      'format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image': _bytes_feature(image_buffer)}))

  return example

def process_images(img_obj,center_crop, size):
  '''
  Decode the image as JPEG , center crop if enabled
  
  Arguments:
  img_obj : Image data
  center_crop : center crop enabled or not
  size : Image width and height
  '''
  with tf.gfile.FastGFile(img_obj.image_path, 'rb') as f:
    img_data = f.read()
  
  #if center_crop:
  #  img_data  = tf.image.resize_image_with_crop_or_pad(img_data, size, size)
  #else:
  #  img_data = tf.image.resize_images(img_data, [size, size])

  return img_data
  

def process(data, mode, out_dir, argv):
  '''
  Process the image data list and store it as TFRecord
  '''
  print("Process dataset length : %d " %(len(data)))
  writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, mode +".tfrecord"))
  for i in range(len(data)):
    img_obj = data[i]
    resized_img = process_images(img_obj,argv.center_crop, argv.image_size)
    example = convert_to_example_proto(resized_img, img_obj.name, argv.image_size, argv.image_size)
    writer.write(example.SerializeToString())

  writer.close()

  

def main(argv):
  if not os.path.isdir(argv.out_dir):
    os.makedirs(argv.out_dir)
  
  data = utils.get_dataset(argv.data_dir)
  split = int(round(argv.split_ratio * len(data)))
  random.seed(argv.seed)
  print("Full Dataset length : %d " %(len(data))) 
  print("Split lenght : %d " %(split))
  if argv.split_type == "FULL":
    process(data, 'full', argv.out_dir, argv)
  elif argv.split_type == "TRAIN_TEST":
    train_dir = os.path.join(argv.out_dir, "train")
    test_dir = os.path.join(argv.out_dir, "test")
    if not os.path.isdir(train_dir):
      os.makedirs(train_dir)
    if not os.path.isdir(test_dir):
      os.makedirs(test_dir)
    train_set, test_set  = data[split:], data[:split]
    print("Train length : %d " %(len(train_set)))
    print("Test length : %d " %(len(test_set))) 
    process(train_set, 'train', train_dir, argv)
    process(test_set, 'test', test_dir, argv)

def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  # input parameters
  parser.add_argument('--data_dir', type=str, help='Path to the directory contains train images', default='./data/input')
  parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=264)
  parser.add_argument('--seed', type=int,help='Random seed.', default=666)
  
  # Pre-processing parameters
  parser.add_argument('--center_crop', help='Performs center cropping/pad of training images.', action='store_true')

  # Output parameters
  parser.add_argument('--out_dir', type=str, help='Path to the output directory', default='./data/output')
  parser.add_argument('--out_format', type=str, help='Whether to write as TFRecord or Normal image', choices=['Normal','TFRecord'],default='TFRecord')
  parser.add_argument('--split_ratio', type=float, help='Test/Val split ratio',default=0.2)
  parser.add_argument('--split_type', type=str, choices=['TRAIN_TEST','FULL'], help='How to split the data', default='FULL')

  return parser.parse_args(argv)


if __name__ == "__main__":
  main(parse_arguments(sys.argv[1:]))


