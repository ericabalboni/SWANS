from keras.models import Model
from keras.layers import Conv3D, Input, MaxPooling3D, UpSampling3D
from keras.layers.merge import concatenate
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

# Custom blocks
from spatialTransformer3d import SpatialTransformer3D
from readh5 import readh5


import nibabel as nib
import numpy as np
import os


weight_decay = 1e-4


#Loss function and scores
def dice_coef(y_true, y_pred, smooth=1):
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
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



# Model to learn the transformation
def deeper_net3d(input_dim, num_output_classes):
    img_input = Input(shape=input_dim, name='image_slice')
    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    ds1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x1)

    x2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    ds2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x2)

    x3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(ds2)
    ds3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x3)

    x4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds3)
    x4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x4)

    us1 = concatenate([UpSampling3D(size=(2, 2, 2))(x4), x3])
    x5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(us1)

    us2 = concatenate([UpSampling3D(size=(2, 2, 2))(x5), x2])
    x6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(us2)

    us3 = concatenate([UpSampling3D(size=(2, 2, 2))(x6), x1])
    x7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(us3)

    dense_prediction_2 = Conv3D(
        num_output_classes,
        (3, 3, 3),
        padding='same',
        activation='linear',
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        bias_regularizer=l2(weight_decay),
        name = 'dense_pred_2')(x7)
    
    #3 warp maps to apply to the ellipse
    encoder = Model(inputs=img_input, outputs=dense_prediction_2)
    
    #transformation for the ellipse
    transformation = SpatialTransformer3D(def_net=encoder,
                                          output_size=(32, 64, 64, 3),
                                          input_shape=input_dim)(img_input)
                                          
    
    
    
    
    model = Model(inputs=img_input, outputs=transformation)
    model.compile(optimizer=Adam(1e-4),
                 loss=[loss],
                 metrics=['accuracy', dice_coef])
    
    
    return model, encoder






def predicting(Folder, name, trained_model): 
    #model parameters
    height = 32
    width = 64
    depth = 64
    batch_size = 1
    input_shape = (height, width, depth, 1)

	
    num_output_classes = 3
    
	
    #call the model
    model, encoder = deeper_net3d(input_shape, num_output_classes)
    
    
        
    #input data
    folderStore_t1 = nib.load(os.path.join(Folder,name+'_to_std_fnirt.nii.gz'))
    fileStore_t1 = folderStore_t1.get_fdata()    
    t1_merge1= np.array(fileStore_t1)
    if t1_merge1.ndim==3:
        t1_merge=np.expand_dims(t1_merge1,axis=0)
    else:
        t1_merge=t1_merge1.transpose(3,0,1,2)
    brain_crop = np.zeros((2*len(t1_merge),32,64,64))
    brain_Lcrop = np.zeros((len(t1_merge),32,64,64))
    brain_Rcrop = np.zeros((len(t1_merge),32,64,64))
    testingSlices = np.ones((2*batch_size*len(t1_merge),32,64,64,1))
    
    for i in range(len(t1_merge)):
        brain_Lcrop[i,:,:,:] = t1_merge[i,100: 132, 76: 140, 40: 104]
        brain_Lcrop[i,:,:,:] = brain_Lcrop[i,:,:,:]/np.percentile(brain_Lcrop[i,:,:,:], 99)
        

    for i in range(len(t1_merge)):
        brain_Rcrop[i,:,:,:] = t1_merge[i,50:82, 76: 140, 40: 104]
        brain_Rcrop[i,:,:,:] = brain_Rcrop[i,:,:,:]/np.percentile(brain_Rcrop[i,:,:,:], 99)
        
    for i in range(len(t1_merge)):
        brain_crop[2*i,:,:,:] = brain_Lcrop[i,:,:,:]
        brain_crop[2*i+1,:,:,:] = brain_Rcrop[i,:,:,:]
        
       
    for i in range(2*len(t1_merge)):
        for j in range(batch_size):
            testingSlices[batch_size*i+j,:,:,:,0]=brain_crop[i,:,:,:]
       
       
    #loading weights
    final_weights = readh5(trained_model)
        
    #layers of the model
    model.summary()
        
    #passing the weights to the model
    print('Loading the weights')
    
    model.set_weights(final_weights)
      
    #testing
    predict = model.predict(testingSlices, batch_size=batch_size, verbose =1)
    
    #output data to be compared
    predict_readable =np.zeros((len(predict),32,64,64))    
    
    predict_readable=predict[:,:,:,:,0]
    
    length = len(predict)//batch_size    
    final_length = length//2
    
    predict_L = np.zeros((final_length, 32, 64, 64))
    predict_R = np.zeros((final_length, 32, 64, 64))

    
    
    for i in range(length):
        predict_readable[i,:,:,:] = predict_readable[batch_size*i,:,:,:]
        
    for i in range(final_length):
        predict_L[i,:,:,:] = predict_readable[2*i,:,:,:]
        predict_R[i,:,:,:] = predict_readable[2*i+1,:,:,:]
        
    predict_L = predict_L.transpose(1,2,3,0)
    predict_R = predict_R.transpose(1,2,3,0)
    
    
    
    return predict_L, predict_R, brain_Lcrop, brain_Rcrop




















