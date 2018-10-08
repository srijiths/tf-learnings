''' 
Utilitiy methods

@author : srijith

'''
from __future__ import print_function
import os

dataset = []

# class to represent an Image object. image label and path is stored
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_path):
        self.name = name
        self.image_path = image_path

    def __str__(self):
        return self.name + ', ' + str(len(self.image_path)) + ' images'

    def __len__(self):
        return len(self.image_path)


def get_image_paths(class_dir, label):
  '''
  Get Image paths from the filesystem

  Arguments : 
    class_dir : Directory containing input images
  
  Returns :
    A list containing absolute image paths
  '''
  image_paths = []
  if os.path.isdir(class_dir):
    images = os.listdir(class_dir)
    for img in images:
      dataset.append(ImageClass(label, os.path.join(class_dir, img)))
  

def get_dataset(path):
  '''
  Get the image dataset from the given path and prepare train data

  Arguments:
    path : Directory containing all input images

  Returns:
    A list containing ImageClass objects 
  '''
  path_exp = os.path.expanduser(path)
  classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
  classes.sort()
  no_classes = len(classes)
  for i in range(no_classes):
    class_name = classes[i]
    class_dir = os.path.join(path_exp, class_name)
    get_image_paths(class_dir,class_name)
  return dataset

