from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import Text
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import talos

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)

import logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.models import load_model, Sequential
from keras.layers import Dense

import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

version = 3

def makeDir():
    try:
        os.mkdir(work_dir)
        os.mkdir(work_dir_A)
        os.mkdir(work_dir_B)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        os.mkdir(train_pos_dir)
        os.mkdir(train_neg_dir)
        os.mkdir(validation_pos_dir)
        os.mkdir(validation_neg_dir)
        os.mkdir(test_pos_dir)
        os.mkdir(test_neg_dir)
        print("All directories have been generated")
    
    except:
        pass

    try:
        os.mkdir(r"C:\Users\Raghu Katragadda\Documents\School\Senior Year\KSU Research\ALL\v" + str(version))
    except:
        pass

base_dir = "C:/Users/Raghu Katragadda/Documents/School/Senior Year/KSU Research/ALL_IDB/ALL_IDB2/img"
work_dir = "C:/Users/Raghu Katragadda/Documents/School/Senior Year/KSU Research/ALL_IDB/ALL_IDB2/work"

base_dir_A = os.path.dirname("C:/Users/Raghu Katragadda/Documents/School/Senior Year/KSU Research/ALL_IDB/ALL_IDB2/img")
base_dir_B = os.path.dirname("C:/Users/Raghu Katragadda/Documents/School/Senior Year/KSU Research/ALL_IDB/ALL_IDB2/img")

work_dir_A = "C:/Users/Raghu Katragadda/Documents/School/Senior Year/KSU Research/ALL_IDB/ALL_IDB2/work/A/"
work_dir_B = "C:/Users/Raghu Katragadda/Documents/School/Senior Year/KSU Research/ALL_IDB/ALL_IDB2/work/B/"

train_dir = os.path.join(work_dir, 'train')

validation_dir = os.path.join(work_dir, 'validation')

test_dir = os.path.join(work_dir, 'test')

train_pos_dir = os.path.join(train_dir, 'pos')
train_neg_dir = os.path.join(train_dir, 'neg')

validation_pos_dir = os.path.join(validation_dir, 'pos')
validation_neg_dir = os.path.join(validation_dir, 'neg')

test_pos_dir = os.path.join(test_dir, 'pos')
test_neg_dir = os.path.join(test_dir, 'neg')

makeDir()


'''
i = 0
      
for i in range(1, 131): 
    dst ="pos" + str(i) + ".jpg"
    

    if(i < 10):
        source = base_dir_A + '/img/Im00' + str(i) + "_1.tif"
    elif(i<100):
        source = base_dir_A + '/img/Im0' + str(i) + "_1.tif"
    else:
        source = base_dir_A + '/img/Im' + str(i) + "_1.tif"

    src = source
    dst = work_dir_A + dst
    print(dst)
        
    
    shutil.copy(src, dst) 
    i += 1


       
j = 0
      
for j in range(131, 261): 
    dst ="neg" + str(j-130) + ".jpg"

    if(j < 10):
        source2 = base_dir_A + '/img/Im00' + str(j) + "_0.tif"
    elif(j<99):
        source2 = base_dir_A + '/img/Im0' + str(j) + "_0.tif"
    else:
        source2 = base_dir_A + '/img/Im' + str(j) + "_0.tif"

    src = source2
    dst = work_dir_B + dst 
        
    
    shutil.copy(src, dst) 
    j += 1       

print("Images for both categories have been copied to working directories, renamed to A & B + num")

fnames = ['pos{}.jpg'.format(i) for i in range(1, 91)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(train_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(91, 121)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(validation_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(121, 131)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(test_pos_dir, fname)
    shutil.copyfile(src, dst)
    


fnames = ['neg{}.jpg'.format(i) for i in range(1, 91)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(train_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(91, 121)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(validation_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(121, 131)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(test_neg_dir, fname)
    shutil.copyfile(src, dst)

'''

numPos = len(os.listdir(base_dir_A))
numNeg = len(os.listdir(base_dir_B))

total_train = numPos + numNeg

#This is where I add all of my augmentation
train_image_generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20)


p = {
    'activation':'relu',
    'optimizer': 'Adam',
    'losses': 'binary_crossentropy',
    'batch_size': 15,
    'epochs': 7,
    'steps_per_epoch': 130
    }



def CNN(params):
    train_generator = train_image_generator.flow_from_directory(
                                                           train_dir,
                                                           target_size=(257, 257),
                                                           batch_size = params['batch_size'],
                                                           class_mode='binary')
    

    validation_generator = train_image_generator.flow_from_directory(
                                                           validation_dir,
                                                           target_size=(257, 257),
                                                           batch_size = params['batch_size'],
                                                           class_mode='binary')
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(16, (3, 3), activation = 'relu',
                            input_shape=(257, 257, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = params['activation']))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = params['activation']))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = params['activation']))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation = params['activation']))
    model.add(layers.Dense(1, activation = 'sigmoid')) ###FIX THIS, YOU MIGHT BE SCREWING IT UP WITH THE 1 INTSTEAD OF 2 AND THE SIGMOID INSTAED OF SOFTMAX. TEST!!!!!!
    model.summary()

    model.compile(loss = params['losses'],
                optimizer = params['optimizer'], 
                metrics=['acc', tf.keras.metrics.AUC(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    
    checkpoint_path = r"C:\Users\Raghu Katragadda\Documents\School\Senior Year\KSU Research\ALL\v" + str(version) + r"\cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    history = model.fit_generator(
            train_generator,
            steps_per_epoch = params['steps_per_epoch'],
            epochs = params['epochs'],
            validation_data=validation_generator)
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    print("Accuracy: " + str(acc) + "\nValidation Accuracy: " + str(val_acc) + "\nLoss: " + str(loss) + "\nValidation Loss: " + str(val_loss) + "\nEpochs: " + str(epochs))
    f = open(r"C:\Users\Raghu Katragadda\Documents\School\Senior Year\KSU Research\ALL\v" + str(version) + r"\values-v" + str(version) + '.txt', 'w')
    f.write("Accuracy: " + str(acc) + "\nValidation Accuracy: " + str(val_acc) + "\nLoss: " + str(loss) + "\nValidation Loss: " + str(val_loss) + "\nEpochs: " + str(epochs))
    f.close

    model.save(r"C:\Users\Raghu Katragadda\Documents\School\Senior Year\KSU Research\ALL\v" + str(version) + r"\model-v" + str(version))



def predict(path):
    class_names = ['Negative', 'Positive']
    model = tf.keras.models.load_model(r"C:\Users\Raghu Katragadda\Documents\School\Senior Year\KSU Research\ALL\v" + str(version) + r"\model-v" + str(version))
    image_file = path

    img = keras.preprocessing.image.load_img(
        image_file, target_size=(257, 257)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    pred_string = str(predictions[0])
    pred_string = pred_string.replace("[", "")
    pred_string = pred_string.replace("]", "")
    score = float(pred_string) * 100
    if(score > 50):
        prediction = "Positive"
    elif(score < 50):
        prediction = "Negative"
        score = 100 - score
    
    print("I am " + str(score) + r"% confident" + " that the sample is " + prediction)

#CNN(p)
#predict(r"C:\Users\Raghu Katragadda\Documents\School\Senior Year\KSU Research\ALL_IDB\ALL_IDB2\work\validation\neg\neg115.jpg")