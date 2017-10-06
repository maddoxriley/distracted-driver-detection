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
import h5py

import run_training as train


def predict_image(filename, net):
    """
    Imports a single image, converts it to numpy array
    format (using code from run_training.py), and makes a
    prediction for the category of distracted driving behavior
    for that image using the neural net and its learned weights
    @param filename: the path to the image
    @param net: the compiled neural net (with learned weights)
    @return prediction: the predicted category, an int from 0-9
    """

    image_arry = train.import_image(filename, 640, 480)

    #Normalize image array values
    image_max = image_arry.max()
    image_min = image_arry.min()

    final_image = (image_arry.reshape(1, 480, 640, 3) - image_min) / float(image_max - image_min)

    result = net.predict(final_image)

    #The output node with the highest value corresponds to the predicted category
    prediction = result.argmax()

    return prediction




def main():

    categories = ["c0: safe driving",\
                  "c1: texting - right",\
                  "c2: talking on the phone - right",\
                  "c3: texting - left",\
                  "c4: talking on the phone - left",\
                  "c5: operating the radio",\
                  "c6: drinking",\
                  "c7: reaching behind",\
                  "c8: hair and makeup",\
                  "c9: talking to passenger"]

    if len(argv) != 2:
        print "Usage: classify_image.py <image path>"
        exit()

    net = train.build_nnet()

    net.load_weights("net_weights.h5")

    path = argv[1]

    #Check that path/filename is valid
    if not(os.path.exists(path)):
        print "Error: file or directory not found"
        exit()


    #If the user-supplied path is to a single image,
    #return a prediction for that image only
    if os.path.isfile(path):
       #Check that file is a .jpg
       if not(path.endswith(".jpg") or path.endswith(".jpeg")):
           print "Error: file is not a JPEG image"
           exit()
       else:
           result = predict_image(path, net)
           print "\n", categories[result], " predicted for image ", path, "\n"

    #If the path is to a directory, make predictions for all images found
    #in that directory
    if os.path.isdir(path):
       for item in os.listdir(path):
           if item.endswith(".jpg") or item.endswith(".jpeg"):
              result = predict_image(path + "/" + item, net)
              print "\n", categories[result], " predicted for image ", item, "\n"
              continue
           else:
               continue















if __name__ == '__main__':
    main()
