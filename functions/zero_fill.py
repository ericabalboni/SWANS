import numpy as np

def zero_fill(cropped_image_L, cropped_image_R):
    new_image_R = np.zeros((182,218,182)) 
    new_image_L = np.zeros((182,218,182))
    new_image_R[50:82, 76: 140, 40: 104] = cropped_image_R[:,:,:]
    new_image_L[100: 132, 76: 140, 40: 104] = cropped_image_L[:,:,:]
    return new_image_L, new_image_R

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    