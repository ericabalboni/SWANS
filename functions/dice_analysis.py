import numpy as np
import nibabel as nib


def dice(predict, true):
    im1 = np.asarray(predict).astype(np.bool)
    im2 = np.asarray(true).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    dice_coef = (2.*intersection.sum())/(im1.sum()+im2.sum())
    return dice_coef
def volume(niftimap):
    img=nib.load(niftimap)
    a,b,c=img.header.get_zooms()
    vol_store=img.get_fdata()
    vol = np.array(vol_store)
    vol_flat=vol.flatten()
    return np.count_nonzero(vol_flat)*a*b*c

