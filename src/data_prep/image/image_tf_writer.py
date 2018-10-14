from __future__ import print_function

import tensorflow as tf
import os
import sys
import utils
import argparse

class ImageTFWriter:
  """
  Write an image as TFRecord. Contains helper functions to do
  Pre-processing and writing as TFRecord
  """
  def __init__(self, args, filenames, labels):
    tf.reset_default_graph()
    self.sess = tf.Session()
    self.args = args
    self.filenames = filenames
    self.labels = labels
    assert len(filenames) == len(labels), "Filenames and labels should have same length"
    print("Full Dataset length : %d " %(len(filenames)))
    if not os.path.isdir(self.args.out_dir):
      os.makedirs(self.args.out_dir)

  def _int64_feature(self, value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
      value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


  def _bytes_feature(self, value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


  def convert_to_example_proto(self, image_buffer, label, img_size):
    """  
    Convert Image features using TF example proto

    Arguments:
      image_buffer: Image data
      label: class label
      img_size: Height and Width of image

    Returns:
      Example proto representation of image data
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'height': self._int64_feature(img_size),
      'width': self._int64_feature(img_size),
      'colorspace': self._bytes_feature(tf.compat.as_bytes(colorspace)),
      'channels': self._int64_feature(channels),
      'label': self._bytes_feature(tf.compat.as_bytes(label)),
      'format': self._bytes_feature(tf.compat.as_bytes(image_format)),
      'image': self._bytes_feature(tf.compat.as_bytes(image_buffer.tostring()))}))

    return example


  def write_tf_records(self): 
    if self.args.split_type == "FULL":
      # create a tf.data.Dataset from full filenames,labels list
      dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
      if not os.path.isdir(os.path.join(self.args.out_dir, "full")):
        os.makedirs(os.path.join(self.args.out_dir, "full"))
      
      self.process(dataset, 'full', os.path.join(self.args.out_dir, "full"))
      
    elif self.args.split_type == "TRAIN_TEST":
      split = int(round(self.args.split_ratio * len(self.filenames)))
      if not os.path.isdir(os.path.join(self.args.out_dir, "train")):
        os.makedirs(os.path.join(self.args.out_dir, "train"))
      if not os.path.isdir(os.path.join(self.args.out_dir, "test")):
        os.makedirs(os.path.join(self.args.out_dir, "test"))

      # create train and test tf.data.Dataset from filenames and labels list
      train_file_set, test_file_set = filenames[split:], filenames[:split]
      train_label_set, test_label_set = labels[split:], labels[:split]

      train_dataset = tf.data.Dataset.from_tensor_slices((train_file_set, train_label_set))
      test_dataset = tf.data.Dataset.from_tensor_slices((test_file_set, test_label_set))

      self.process(train_dataset, 'train', os.path.join(self.args.out_dir, "train"))
      self.process(test_dataset, 'test', os.path.join(self.args.out_dir, "test"))

  def read_convert_image(self, img_path, label, args):
    """
    Read image from disk and crop/resize based on args

    Arguments:
      img_path : Path to read image 
      label : Image class name
      args : arguments to the program

    Returns:
      image_data : cropped/resized image
      label : Image class name
    """
    image_data = tf.read_file(img_path)
    image_data = tf.image.decode_jpeg(image_data, channels=3)

    if args.center_crop:
      image_data  = tf.image.resize_image_with_crop_or_pad(image_data, args.crop_size, args.crop_size)

    if args.resize:
      image_data = tf.image.resize_images(image_data, [args.image_size, args.image_size])

    return image_data, label

  def process(self, dataset, mode, out_dir):
    """
    Read images , pre-process it and write as TFRecord

    Arguments:
      dataset : tf.data.Dataset object containng image filenames,label to process
      mode : which mode train/test or full ?
      out_dir : where to write the TFRecord file

    """
    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, mode +".tfrecord"))
    
    # lambda function which takes each filename, label from dataset and process it    
    process_fn = lambda img, lbl: self.read_convert_image(img, lbl, self.args)

    # dataset operations
    dataset = dataset.shuffle(self.args.batch_size).map(process_fn, num_parallel_calls=self.args.no_threads).apply(tf.contrib.data.ignore_errors()).batch(self.args.batch_size).prefetch(10)
    
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()
    init_op = iterator.initializer

    self.sess.run(init_op)
    while True:
      try:
        imgs, lbls = self.sess.run([image, label])
        for img,lbl in zip(imgs,lbls):
          example_proto = self.convert_to_example_proto(img, lbl, img.shape[1])
          writer.write(example_proto.SerializeToString())
      except tf.errors.OutOfRangeError:
        print("End of training dataset.")
        break
    writer.close()


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  # input parameters
  parser.add_argument('--data_dir', type=str, help='Path to the directory contains train images', default='./data/input')
  parser.add_argument('--seed', type=int, help='Random seed.', default=666)
  parser.add_argument('--batch_size', type=int, help='Batch size to fetch from dataset', default=50)
  parser.add_argument('--no_threads', type=int, help='Number of parllel threads to run in dataset', default=5)

  # Pre-processing parameters
  parser.add_argument('--center_crop', help='Performs center cropping/pad of training images.', action='store_true')
  parser.add_argument('--crop_size', type=int, help='Crop size', default=300)
  parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=150)
  parser.add_argument('--resize', help='Resize image based on image_size', action='store_true')

  # Output parameters
  parser.add_argument('--out_dir', type=str, help='Path to the output directory', default='./data/output')
  parser.add_argument('--split_ratio', type=float, help='Test/Val split ratio',default=0.2)
  parser.add_argument('--split_type', type=str, choices=['TRAIN_TEST','FULL'], help='How to split the data', default='FULL')

  return parser.parse_args(argv)


if __name__ == "__main__":

  args = parse_arguments(sys.argv[1:])
  # create a filenames, labels list
  filenames, labels = utils.get_images_labels(args.data_dir)
  
  # create ImageTFWriter object and process
  image_writer = ImageTFWriter(args, filenames, labels)
  image_writer.write_tf_records()

