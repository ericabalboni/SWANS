import numpy as np


def normalization(imgs):
    a = len(imgs)
    perc_99=np.zeros((a))
    perc_1=np.zeros((a))
    imgs_out=np.zeros((a,32,64,64))
    for i in range(a):
        perc_99[i]=np.percentile(imgs[i,:,:,:], 99)
        perc_1[i]=np.percentile(imgs[i,:,:,:], 1)
        imgs_out[i,:,:,:]=(imgs[i,:,:,:]-perc_1[i])*(1/(perc_99[i]-perc_1[i]))
    return imgs_out