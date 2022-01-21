# Nicola Dinsdale 2021
# Erica Balboni
# Code for the segmentation of the hippocampus for T1 brain extracted images
##################################################################################
import argparse
parser = argparse.ArgumentParser(description='Segments the right and left hippocampi from T1 brain exctracted images')
parser.add_argument('-i','--image', metavar='image', type=str, required=True,
                    help='basename of the image to be segmented, e.g. t1_brain')
parser.add_argument('-m', '--model', metavar='model', type=str, default='trained', choices=['trained','locked'],
                    help='model which will be used for segmentation, it can be trained (default) or locked ')
args = parser.parse_args()
image=args.image
model=args.model

import nibabel as nib
import numpy as np
from os.path import join, split, splitext
from matplotlib import pyplot as plt

#custom libraries:
from dice_analysis import volume
from testing import predicting  
from zero_fill import zero_fill
from fsl_functions import flirt, fnirt, applywarp, invwarp, thr, binarize


Folder, file=split(image)
name, ext=splitext(file)

if ext=='.gz':
    name, ext1=splitext(name)
    ext=ext1+ext


def common(i):
    
    #Put maps and images in common MNI space
    print('Linear registration')
    flirt(join(i,name+ext), '/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz', 
             join(i,name+'_to_std_flirt.nii.gz'), join(i,name+'_to_std_flirt.m'))
    print('Non-linear registration')
    fnirt(join(i,name+'.nii.gz'), '/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz',
             join(i, name+'_to_std_flirt.m'), join(i,name+'_to_std_fnirt.nii.gz'), join(i,name+'_warpcoef'))
    
     
    
def subject(i):
    print('Calculating inverse warp')
    invwarp(join(i,name+'_warpcoef.nii.gz'), join(i,name+ext),
            join(i, name+'_inv_warpcoef.nii.gz'))
    print('Applying inverse warp')
    applywarp(join(i,name+'_LHipp_prediction.nii.gz'), join(i,name+'_LHipp_prediction_to_sub.nii.gz'),
              join(i,name+ext), join(i,name+'_inv_warpcoef.nii.gz'))
    applywarp(join(i,name+'_RHipp_prediction.nii.gz'), join(i,name+'_RHipp_prediction_to_sub.nii.gz'),
              join(i, name+ext), join(i,name+'_inv_warpcoef.nii.gz'))
    thr(join(i,name+'_RHipp_prediction_to_sub.nii.gz'), 0.5)
    binarize(join(i,name+'_RHipp_prediction_to_sub.nii.gz'))
    thr(join(i,name+'_LHipp_prediction_to_sub.nii.gz'), 0.5)
    binarize(join(i,name+'_LHipp_prediction_to_sub.nii.gz'))

    

###############################################################################################


common(Folder)


###################################################################################################
print('Testing data')
#Makes predictions and saves them
predict_L, predict_R, brain_Lcrop, brain_Rcrop = predicting(Folder, name, model) 

    
length = np.shape(predict_L)[3]


predict_L[predict_L < 0.5] = 0.
predict_R[predict_R < 0.5] = 0.
predict_L[predict_L >= 0.5] = 1.
predict_R[predict_R >= 0.5] = 1.

  
#########################################################################################################  
  
    
print('Start registration to subject space')    
predict_common_L = np.zeros((182,218,182))
predict_common_R = np.zeros((182,218,182))


predict_common_L[:,:,:], predict_common_R[:,:,:] = zero_fill(predict_L[:,:,:,0], predict_R[:,:,:,0])

func = nib.load(join(Folder, name+'_to_std_fnirt.nii.gz'))
predict_common_L_img = nib.Nifti1Image(predict_common_L, affine = func.affine)
predict_common_L_header = predict_common_L_img.header
predict_common_L_header['sform_code'] = 4
predict_common_L_header['qform_code'] = 4  
nib.save(predict_common_L_img, join(Folder,name+'_LHipp_prediction.nii.gz'))
    

func = nib.load(join(Folder, name+'_to_std_fnirt.nii.gz'))
predict_common_R_img = nib.Nifti1Image(predict_common_R, affine = func.affine)
predict_common_R_header = predict_common_R_img.header
predict_common_R_header['sform_code'] = 4
predict_common_R_header['qform_code'] = 4  
nib.save(predict_common_R_img, join(Folder,name+'_RHipp_prediction.nii.gz'))
  

subject(Folder)

###############################################################################################################
print('Volume analysis in subject space')
PR_vol=volume(join(Folder,name+'_RHipp_prediction_to_sub.nii.gz'))
PL_vol=volume(join(Folder,name+'_LHipp_prediction_to_sub.nii.gz'))
PR_vol_MNI=volume(join(Folder,name+'_RHipp_prediction.nii.gz'))
PL_vol_MNI=volume(join(Folder,name+'_LHipp_prediction.nii.gz'))
flags=np.array([['Right_Hippocampus', str(PR_vol)], ['Left_Hippocampus',str(PL_vol)], 
                ['Right_Hippocampus_MNI', str(PR_vol_MNI)], ['Left_Hippocampus_MNI', str(PL_vol_MNI)]])
np.savetxt(Folder+name+'_volumes.txt', flags,
              fmt='%s')
###############################################################################################################
print('Production of png file')

ax1=plt.subplot(2, 3, 1)
ax1.imshow(brain_Lcrop[0,8,:,:].transpose(1,0), cmap='gray', origin='lower')
ax1.imshow(np.ma.masked_where(predict_L[8,:,:,0].transpose(1,0)==0,predict_L[8,:,:,0].transpose(1,0)), 
           cmap='hsv', origin='lower', alpha=0.2)
ax1.axis('off')
ax2=plt.subplot(2, 3, 2)
ax2.imshow(brain_Lcrop[0,16,:,:].transpose(1,0), cmap='gray', origin='lower')
ax2.imshow(np.ma.masked_where(predict_L[16,:,:,0].transpose(1,0)==0,predict_L[16,:,:,0].transpose(1,0)), 
           cmap='hsv', origin='lower', alpha=0.2)
ax2.axis('off')
ax3=plt.subplot(2, 3, 3)
ax3.imshow(brain_Lcrop[0,24,:,:].transpose(1,0), cmap='gray', origin='lower')
ax3.imshow(np.ma.masked_where(predict_L[24,:,:,0].transpose(1,0)==0,predict_L[24,:,:,0].transpose(1,0)), 
           cmap='hsv', origin='lower', alpha=0.2)
ax3.axis('off')

ax4=plt.subplot(2, 3, 4)
ax4.imshow(brain_Rcrop[0,8,:,:].transpose(1,0), cmap='gray', origin='lower')
ax4.imshow(np.ma.masked_where(predict_R[8,:,:,0].transpose(1,0)==0,predict_R[8,:,:,0].transpose(1,0)), 
           cmap='PRGn', origin='lower', alpha=0.2)
ax4.axis('off')
ax5=plt.subplot(2, 3, 5)
ax5.imshow(brain_Rcrop[0,16,:,:].transpose(1,0), cmap='gray', origin='lower')
ax5.imshow(np.ma.masked_where(predict_R[16,:,:,0].transpose(1,0)==0,predict_R[16,:,:,0].transpose(1,0)), 
           cmap='PRGn', origin='lower', alpha=0.2)
ax5.axis('off')
ax6=plt.subplot(2, 3, 6)
ax6.imshow(brain_Rcrop[0,24,:,:].transpose(1,0), cmap='gray', origin='lower')
ax6.imshow(np.ma.masked_where(predict_R[24,:,:,0].transpose(1,0)==0,predict_R[24,:,:,0].transpose(1,0)), 
           cmap='PRGn', origin='lower', alpha=0.2)
ax6.axis('off')
plt.savefig(join(Folder,name+'_plots.png'))







