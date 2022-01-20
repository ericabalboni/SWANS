# Nicola Dinsdale 2018
# Applying the 3d segmentation to the HarP dataset
################################################################################
# Import dependencies
from keras.models import Model, Sequential
from keras.layers import Dropout, Conv3D, Input, MaxPooling3D, UpSampling3D, Flatten, Dense
from keras.layers.merge import concatenate
import keras.backend as K
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, binary_crossentropy

# Custom blocks
from spatialTransformer3d import SpatialTransformer3D
from functions3d import createElipsoid2

# Other
import numpy as np

weight_decay = 1e-4


################################################################################
# Define the loss function


def dice_coef(y_true, y_pred, smooth=1):
    """
    :param y_true: true labels
    :param y_pred: predicted labels
    :param smooth: smoothing factor to prevent division by one
    :return: dice loss val, float
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    smooth = 1.0
    iflat = K.flatten(y_true)
    tflat = K.flatten(y_pred)
    intersection = K.sum(iflat * tflat)
    return 1 - ((2. * intersection + smooth) / (K.sum(iflat) + K.sum(tflat) + smooth))

def loss(y_true, y_pred):
    alpha = 0.6
    loss =  dice_loss(y_true, y_pred) + alpha *  binary_crossentropy(y_true, y_pred)
    return loss

# Define the model to learn the transformation
def get_mini_net3d(input_dim, num_output_classes):
    img_input = Input(shape=input_dim, name='image_slice')
    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    ds1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x1)

    x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    ds2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x2)

    x4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds2)
    x4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x4)

    us2 = concatenate([UpSampling3D(size=(2, 2, 2))(x4), x2])
    x6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(us2)

    us3 = concatenate([UpSampling3D(size=(2, 2, 2))(x6), x1])
    x7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(us3)

    dense_prediction = Conv3D(
                             num_output_classes,
                             (3, 3, 3),
                             padding='same',
                             activation='linear',
                             kernel_initializer='orthogonal',
                             kernel_regularizer=l2(weight_decay),
                             bias_regularizer=l2(weight_decay))(x7)

    encoder = Model(inputs=img_input, outputs=dense_prediction)
    transformation = SpatialTransformer3D(def_net=encoder,
                                          initial_map=ellipse,
                                          output_size=(height, width, depth, batch_size),
                                          input_shape=input_dim)(img_input)


    model = Model(inputs=img_input, outputs=transformation)
    model.compile(optimizer=Adam(1e-4), loss=[loss], metrics=['accuracy', dice_coef])

    return model


def get_mini_net3d_aux(input_dim, num_output_classes):
    img_input = Input(shape=input_dim, name='image_slice')
    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    ds1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x1)

    x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    ds2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x2)

    x4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds2)
    x4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x4)

    us2 = concatenate([UpSampling3D(size=(2, 2, 2))(x4), x2])
    x6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(us2)

    us3 = concatenate([UpSampling3D(size=(2, 2, 2))(x6), x1])
    x7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(us3)

    dense_prediction = Conv3D(
        num_output_classes,
        (3, 3, 3),
        padding='same',
        activation='linear',
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        bias_regularizer=l2(weight_decay))(x7)

    encoder = Model(inputs=img_input, outputs=dense_prediction)

    transformation = SpatialTransformer3D(def_net=encoder,
                                          initial_map=ellipse,
                                          output_size=(height, width, depth, batch_size),
                                          input_shape=input_dim)(img_input)

    flat = Flatten()(x4)
    dense1 = Dense(300, activation='relu')(flat)
    dropout = Dropout(0.25, noise_shape=None, seed=None)(dense1)
    classification = Dense(3, activation='softmax', name='Classification')(dropout)

    # Define the model
    model = Model(inputs=img_input, outputs=[transformation, classification])
    model.compile(optimizer=Adam(1e-4), loss=[dice_coef_loss, 'categorical_crossentropy'], metrics=['accuracy', dice_coef])
    return model


################################################################################
################################################################################

# Create the input data
im_size = (32, 64, 64)     

testingSlices = np.load('harp_testing_3d_larger.npy')
testingTrueSeg = np.load('harp_testing_labels_larger.npy')
trainingSlices = np.load('harp_training_3d_larger.npy')
trainingTrueSeg = np.load('harp_training_labels_larger.npy')

trainClasses = np.load('harp_training_classes_larger.npy')
testClasses = np.load('harp_testing_classes_larger.npy')


print('Creating Initial Ellipse')
ellipse = createElipsoid2(im_size, scale=8, center=[7, 10, 10])
ellipse = np.reshape(ellipse, (32, 64, 64, 1))
ellipse = tf.convert_to_tensor(ellipse)

# ###############################################################################
# ########################### Create the model ##################################

# Model parameters
height = 32
width = 64
depth = 64
batch_size = len(trainingSlices)
input_shape = (height, width, depth, 1)
output_shape = (height, width, depth, 3)
num_output_classes = 3
num_output_classes_aux = 3

model = Sequential()
model = get_mini_net3d(input_shape, num_output_classes)

print('\n***** ***** ***** ***** ***** ***** ')
print('*** Spatial transformer summary *** ')
print('***** ***** ***** ***** ***** ***** ')
model.summary()

# Train the model
print('Creating the model fit', flush=True)
class_weights = {'Classification': {0: 1, 1: 1.14, 2: 1.3}}
loss = model.fit(trainingSlices,
                 trainingTrueSeg,
                 validation_split=0.05,
                 epochs=100,
                 batch_size=2,
                 verbose=2)


# Save the results
print('Saving the weights')
model.save('harp_weights_3d.h5')


