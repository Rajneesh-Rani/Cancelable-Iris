import glob
import os

import tensorflow as tf
# build the lightgbm model
import lightgbm as lgb
clf_L = lgb.LGBMClassifier()
folders = glob.glob('database/final/*')
import numpy as np
imagenames_list=[]
labels=[]

imagenames__list = []
count=0
for folder in folders:

    for f in glob.glob(folder+'/*'):
        imagenames_list.append(f)
        labels.append(count)

    count+=1

read_images = []
Tensor_input=[]
for image in imagenames_list:
    read_images.append(np.load(image))

Labels=tf.keras.utils.to_categorical(
    labels,
    num_classes=None)
print(Labels.shape)

# print(read_images)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(read_images, Labels, test_size=0.10)



train_images = np.array(X_train)
test_images = np.array(X_test)


train_images = train_images.reshape(train_images.shape[0], 16,256, 1)
test_images = test_images.reshape(test_images.shape[0],16,256, 1)
#





print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))




input_shape=(16, 256, 1)
input = tf.keras.Input(shape=input_shape)

x=tf.keras.layers.Flatten()(input)
x=tf.keras.layers.Dense(500)(x)
#
x=tf.keras.layers.Dense(208, activation='softmax')(x)
model=tf.keras.Model(inputs=input,outputs=x)
#
#





model.summary()
testing = False
epochs = 300
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            patience=7,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
model.fit(train_images, y_train, epochs=epochs, validation_data = (test_images, y_test),callbacks=[earlyStopping,learning_rate_reduction])
test_loss, test_acc = model.evaluate(test_images, y_test)
print('\nTest accuracy: {}'.format(test_acc))


import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
if os.path.isdir(export_path):
  print('\nAlready saved a model, cleaning up\n')
  import shutil

  shutil.rmtree(export_path)


tf.saved_model.save(model, export_path)
print('\nSaved model:')



