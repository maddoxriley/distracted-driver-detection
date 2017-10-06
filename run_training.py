"""
Author: Maddox Riley

Baded on an independent project from course: CS63, Artificial Intelligence
at Swarthmore College with Prof. Bryce Wiedenbeck
"""

from sys import argv
import os.path
import time
from PIL import Image
import csv
import operator
import random

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout
from keras.layers.core import Flatten
from keras.utils import to_categorical

import numpy as np
import math

def import_image(filename, cols, rows):
    """
    Imports a single image file for
    conversion to a numpy array
    @param filename: the path/name of the image file to be imported
    @param cols: the horizontal resolution of the image
    @param rows: the vertical resolution of the image
    @return: the image represented as a numpy array where each
             array element is a sub-array of R, G, B values (0-255)
    """

    if not(os.path.exists(filename)):
        print "Error: image ", filename, " not found"
        exit()

    image = Image.open(filename)
    image_arry = np.array(image.getdata()).reshape(rows, cols, 3)

    return image_arry

def import_label_data(filename):
    """
    Read the data, including category labels, for all images in the
    training set from the provided .csv
    @param filename: the path/name of the .csv file containing the label data
                     associated with each training image
    @return:  the shuffled label data, represented as a list where each
              element is a list of the form [subject, classname (category), filename]
    """
    label_data = []

    if not(os.path.exists(filename)):
        print "Error: Label .csv file not found"
        exit()

    with open(filename, 'rb') as labels_file:
        labelreader = csv.DictReader(labels_file)
        for row in labelreader:
            label_data.append([row['subject'], row['classname'], row['img']])

    random.shuffle(label_data)
    return label_data


def split_label_data(data, sublist_size):
  """
  Takes in the complete imported training image label data
  and splits it into sublists of a specified size. If the number
  of images is not evenly divisible by the sublist size, the last
  sublist will contain the remainder of the images.

  Splitting the image data in this way avoids swamping the main memory
  of the computer used to run neural net training by allowing the net to be
  trained on one smaller subset of the images at a time, rather than decompressing
  all images and loading them into memory (which wouldn't work)

  @param data: all training labels as imported from the supplied .csv
  @param sublist_size: the integer size of each label sublist
                       (excluding the last sublist if there is a remainder)
  @return: a list of sublists containing training image label data
  """
  num_sublists = int(math.ceil(len(data)/(sublist_size*1.0)))
  label_sublists = []

  for i in range(num_sublists):
      label_sublists.append(data[i*sublist_size:(i+1)*sublist_size])

  return label_sublists



def import_image_subset(label_subset, import_path, cols, rows):
    """
    Imports a single subset of the training images as a large
    numpy array
    @param label_subset: a list containing label data for one subset of
                         the training images
    @param import_path: the path to the directory containing the train images
    @params cols, rows: the resolution of the train images
    @return: a numpy array containing all imported images
            (which are themselves numpy arrays) numpyin a training subset
    """

    if not(os.path.exists(import_path)):
       print "Error: train image directory not found"
       exit()

    image_subset = []
    num_images = len(label_subset)
    import_progress = 0

    print "Loading images..."

    for label in label_subset:
        #training images are stored in subfolders according to their
        #category labels. Use each image's category label to determine the
        #sub-path to that image
        filename = import_path + "/" + label_subset[import_progress][1] \
                   + "/" + label_subset[import_progress][2]
        image = import_image(filename, cols, rows)
        image_subset.append(image)
        import_progress += 1
        #print "Imported image ", import_progress, " of ", num_images

    return np.array(image_subset)



def build_nnet():
    """
    Constructs a keras neural network topology using two convolutional layers with
    16 feature detectors. A pooling layer condenses the output from the convolutional
    layers, and a dense layer provides the network's output (10 output nodes for the
    10 image categories)

    @return: the complied network topology

    """
    net = Sequential()

    net.add(Conv2D(16, (8, 8), strides = (4, 4), activation='relu', \
                input_shape = (480, 640, 3)))
    net.add(Conv2D(16, (8, 8), strides = (4, 4), activation='relu'))

    net.add(MaxPooling2D(pool_size=(2, 2), strides=None, \
                padding='valid', data_format=None))


    net.add(Flatten())


    net.add(Dense(10, activation='softmax'))


    net.summary()

    net.compile(optimizer="RMSprop", loss="categorical_crossentropy", \
                metrics=['accuracy'])

    return net




def run_training(label_subsets, import_path, net, cols, rows, num_epochs):
    """
    Performs training on the neural net once it has been complied.
    Saves learned network weights at the end of training on each train image subset

    @param label_subsets: list containing the label data for all train images
    @param import_path: path to the directory containing train images
    @param net: the compiled keras neural network
    @params cols, rows: the resolution of the train images
    @param num_epochs: the number of training epochs to be used for each train subset.
                       3-5 epochs appears to work well
    @return: None
    """

    num_subsets = len(label_subsets)

    start = time.time()
    accuracy_totals = 0.0
    for sub in range(num_subsets):
        print "Training on subset", sub+1, " of ", num_subsets
        label_data = label_subsets[sub]

        #Extract just the labels (0-9) from the image data
        y_data = []
        for item in label_data:
          y_data.append(int(item[1][1]))


        #Import images, represented as numpy arrays, from disk
        x_data = import_image_subset(label_data, import_path, cols, rows)


        #Split into train and test examples for cross-validation
        train_examples = int(len(y_data) * 0.70)
        test_examples = len(y_data) - train_examples

        x_train = x_data[:train_examples]
        x_test = x_data[train_examples:]

        y_train = y_data[:train_examples]
        y_test = y_data[train_examples:]


        #Normalize values within the arrays representing images
        x_max = x_train.max()
        x_min = x_train.min()
        num_categories = 10

        x_train_images = \
          (x_train.reshape(train_examples, 480, 640, 3) - x_min) / float(x_max - x_min)

        x_test_images = \
          (x_test.reshape(test_examples, 480, 640, 3) - x_min) / float(x_max - x_min)


        y_train_vectors = to_categorical(y_train, num_categories)
        y_test_vectors = to_categorical(y_test, num_categories)



        history = net.fit(x_train_images, y_train_vectors, \
            verbose=1, validation_data=(x_test_images, y_test_vectors), epochs=num_epochs)

        loss, accuracy = net.evaluate(x_test_images, y_test_vectors, verbose = 0)
        print "validation accuracy: {}%".format(accuracy*100)
        accuracy_totals += accuracy
        print "average val accuracy: {}%".format((accuracy_totals*100)/(sub+1))

        net.save_weights("net_weights.h5")

    end = time.time()
    print "Completed training in ", (end-start)/60, " minutes"

def main():

    cmd_args = "\nUsage: run_training.py <label .csv> <training image directory>\
    <training sublist size> <training epochs>\n"

    if len(argv) != 5:
        print cmd_args
        exit()


    net = build_nnet()

    label_data = import_label_data(argv[1])

    sublist_size = int(argv[3])

    label_subsets = split_label_data(label_data, sublist_size)

    import_path = argv[2]

    cols = 640
    rows = 480

    num_epochs = int(argv[4])

    run_training(label_subsets, import_path, net, cols, rows, num_epochs)


if __name__ == '__main__':
    main()
