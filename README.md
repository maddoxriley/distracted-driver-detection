This project uses the Python library Keras to build
a convolutional neural network that can be trained to classify
images of distracted drivers behind the wheel. It is based on a project I 
developed independently for the course CS63: Artificial Intelligence at 
Swarthmore College with Professor Bryce Wiedenbeck.

The source of the
data for this project is a machine learning competition posted by
State Farm on kaggle.com:

https://www.kaggle.com/c/state-farm-distracted-driver-detection

The images are 640x480 JPEGs of many different drivers behind the wheel, all
engaging in behaviors that are categorized as follows:

c0: safe driving
c1: texting - right
c2: talking on the phone - right
c3: texting - left
c4: talking on the phone - left
c5: operating the radio
c6: drinking
c7: reaching behind
c8: hair and makeup
c9: talking to passenger

The script run_training.py trains a neural net using the training data provided
as part of the competition. It takes 4 command line arguments:

1. The path to a .csv file with label data for the training images, provided with
    the competition materials

2. The directory containing the training images. The script expects the training
   images to be grouped in subfolders according to category c0-c9, as they are
   in the provided competition materials.

3. An integer representing the size of training image subsets to be used during
   training. The reasoning behind this is that the training images must be represented
   as numpy arrays for processing by the neural network. Since this representation
   of images is quite large, loading hundreds of images at a time for training would
   quickly swamp the main memory of the computer being used. Instead, training is
   performed on sublists of a certain size one at a time, with the
   size being specified by this argument. As a rule of thumb, a subset size of
   150 for each 8 GB of RAM in the computer used should be safe, though this could
   vary depending on the system.

4. The number of epochs to use for each training subset. This should be large
   enough for the network to meaningfully learn on the training data, but small
   enough to avoid overfitting for the training data. A vale of 5 seems to work well.


The script classify_image.py classifies images from the test set after training
has been completed. It compiles the same network topology used during training
and applies the learned weights, stored in the file "net_weights.h5". The script expects
the file to have this name and be stored in the same directory as the script. The weights file
included in the repository contains the weights from training so that the classification
script can be used immediately. However, running the training script will overwrite
these learned weights, unless the included "net_weights.h5" file is moved or renamed.

classify_image.py takes a single command line argument, which is the path to the image(s)
to be classified. If the path is a single image file, that image alone will be classified.
If the path is a directory, the script will classify all the images found in that directory.
# distracted-driver-detection
