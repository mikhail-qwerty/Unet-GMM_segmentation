from IO import*
from data import* 
from Unet_models import*

import tensorflow as tf 
from glob import glob
from sklearn.model_selection import train_test_split

# find availiable GPUs and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# set mixed precision (32/16 bits) for weights calculation
tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

#------------------------------------------------------------------------
# set parameters for model training 

model = '2D' # choose 2D or 3D U-net to train
im_path = '/home/m_fokin/NN/Unet_3D_2D/data/data_patches/2D_256/images/' # path to images
label_path = '/home/m_fokin/NN/Unet_3D_2D/data/data_patches/2D_256/labels/' # path to masks
data_format = '.tif' # format of images and masks

batch_size = 10 
epochs = 15
input_size = (256, 256, 1)
# ------------------------------------------------------------------------

# data preparation
im_list = sorted(glob(f'{im_path}*{data_format}'))
label_list = sorted(glob(f'{label_path}*{data_format}'))

print(f'{im_path}*{data_format}')
print(f'{label_path}*{data_format}')

# split dataset to training and validation
im_train_list, im_test_list, label_train_list, label_test_list  = train_test_split(im_list, label_list, train_size=0.8)

# define steps per epoch according to the number of images and batch_size
steps_per_epoch = len(im_train_list) // batch_size
validation_steps = len(im_test_list) // batch_size

# initialize train and validation data generators 
train_gen = dataGenerator_3D(im_list = im_train_list, label_list = label_train_list, batch_size = batch_size)
validation_gen = dataGenerator_3D(im_list = im_test_list, label_list = label_test_list, batch_size = batch_size)
# define and train UNet_3D model

if (model == '3D'):
    model = UNet_3D(input_size = input_size)
if (model == '2D'):
    model = UNet_2D(input_size = input_size)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
model_checkpoint = ModelCheckpoint('./UNet_weights_2D.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
history = model.fit(train_gen,steps_per_epoch=steps_per_epoch,validation_data = validation_gen, 
                    validation_steps = validation_steps, epochs=epochs,callbacks=[model_checkpoint])