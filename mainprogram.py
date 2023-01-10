from keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model


#
# folders = glob.glob('/home/webtunix/Downloads/Augmetation25New/*')
# imagenames_list=[]
# labels=[]
#
# imagenames__list = []
# count=0
# for folder in folders:
#
#     for f in glob.glob(folder+'/*'):
#         imagenames_list.append(f)
#         labels.append(count)
#
#     count+=1
#
# read_images = []
# Tensor_input=[]
# for image in imagenames_list:
#     read_images.append(cv2.resize(cv2.imread(image),(64,40)))
#
# Labels=tf.keras.utils.to_categorical(
#     labels,
#     num_classes=None)
# print(Labels.shape)
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(read_images, Labels, test_size=0.10)
#
#
#
# img_width, img_height = 40, 64
# from keras.models import Model
#
# from keras.layers import *
#
# epochs = 350
# batch_size = 32
#
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)
#
#

rnn_size=12
rnn_size1=12
input_shapes=(224,224,3)
input_p = tf.keras.Input(shape=input_shapes)

# input = Input(shape=input_shapes)

x=tf.keras.layers.Convolution2D(64, (3, 3), strides=(2, 2),dilation_rate=(1, 1),activation='relu',use_bias=True,kernel_regularizer=l2(0.0002))(input_p)
x=tf.keras.layers.Convolution2D(256, (3, 3), strides=(2, 2),dilation_rate=(1, 1),activation='relu',use_bias=True,kernel_regularizer=l2(0.0002))(x)
x=tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(x)
x=tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid',)(x)
x=tf.keras.layers.Dropout(0.25)(x)

conv_shape = x.get_shape()
x = tf.keras.layers.Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

# convolution_shape=x.get_shape()
# x=tf.keras.layers.Reshape(target_shape=(int(convolution_shape[1]), int(convolution_shape[2]*convolution_shape[3])))(x)
#


gru_1 = tf.keras.layers.GRU(rnn_size, return_sequences=True, name='gru1')(x)
gru_1a = tf.keras.layers.GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru1_a')(x)
gru1_merged = tf.keras.layers.Add()([gru_1, gru_1a])

gru_2 = tf.keras.layers.GRU(rnn_size, return_sequences=True, name='gru2')(gru1_merged)
gru_2a = tf.keras.layers.GRU(rnn_size, return_sequences=True, go_backwards=True,  name='gru2_a')(gru1_merged)
x = tf.keras.layers.Concatenate(axis=1, name='DYNN/output')([gru_2, gru_2a])



gru_3 = tf.keras.layers.GRU(rnn_size, return_sequences=True,  name='gru3')(x)
gru_3b = tf.keras.layers.GRU(rnn_size, return_sequences=True, go_backwards=True,  name='gru_3b')(x)
gru3_merged = tf.keras.layers.Add()([gru_3, gru_3b])

gru_4 = tf.keras.layers.GRU(rnn_size, return_sequences=True,  name='gru4')(gru3_merged)
gru_4b = tf.keras.layers.GRU(rnn_size, return_sequences=True, go_backwards=True,  name='gru4_b')(gru3_merged)
x = tf.keras.layers.Concatenate(axis=1, name='DYNN2/output')([gru_4, gru_4b])

x=tf.keras.layers.Dropout(0.5)(x)

x=tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256)(x)

model=tf.keras.Model(inputs=input_p, outputs=x)
from keras.optimizers import SGD,  Adam, Adadelta
#
# adam = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])


# model.summary()
#
import numpy as np
first_input=cv2.resize(cv2.imread('/home/aj/Desktop/kiritii/left/014/01_L.bmp'),(224,224))

input_image = np.reshape(np.array(first_input, dtype=np.float32), [1, 224, 224,3]) / 255

outputs=model(input_image)
print(type(outputs))