import numpy as np
import h5py
def readh5(trained_model):   
    
    if trained_model=='locked' :  
        hdf1 = h5py.File('./models/harp_weights_3d_crossentropy_working.h5', 'r')
        final_weights=[np.array(hdf1['model_weights/spatial_transformer3d_1/block1_conv1/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block1_conv1/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block2_conv1/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block2_conv1/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block3_conv1/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block3_conv1/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block4_conv1/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block4_conv1/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block4_conv3/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/block4_conv3/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/us1_conv2/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/us1_conv2/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/us2_conv2/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/us2_conv2/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/us3_conv2/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/us3_conv2/bias:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/conv3d_1/kernel:0']),
                     np.array(hdf1['model_weights/spatial_transformer3d_1/conv3d_1/bias:0'])]
    
    
    if trained_model=='trained'  :            
        hdf_training=h5py.File('./models/training_all.h5', 'r' )
        final_weights=[np.array(hdf_training['spatial_transformer3d_2/block1_conv1_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block1_conv1_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block2_conv1_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block2_conv1_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block3_conv1_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block3_conv1_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block4_conv1_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block4_conv1_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block4_conv3_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/block4_conv3_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/us1_conv2_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/us1_conv2_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/us2_conv2_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/us2_conv2_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/us3_conv2_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/us3_conv2_1/bias:0']),
                     np.array(hdf_training['spatial_transformer3d_2/dense_pred_2_1/kernel:0']),
                     np.array(hdf_training['spatial_transformer3d_2/dense_pred_2_1/bias:0'])]
                          
    
    return final_weights

