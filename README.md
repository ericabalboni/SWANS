# SWANS
## Spatial warping network for 3D segmentation of the hippocampus in MRI images

### MICCAI 2019: https://link.springer.com/chapter/10.1007/978-3-030-32248-9_32

Any issues please contact: erica.balboni@unimore.it

Prerequisites
-----------------
FMRIB Software Library (FSL) version 5.0 or higher (http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
	
python 3.6

tensorflow 1.15 (https://www.tensorflow.org/install/pip#ubuntu-macos)

keras 2.3.1

numpy 1.19.2

nibabel 3.1.1

nipype 1.5.1

matplotlib 3.3.2


Usage
-----------------
First, it is necessary to activate the tensorflow virtual environment and to add the folder "functions" to the pythonpath:

$ source /path_to_virt_env/bin/activate

$ export PYTHONPATH=/path_to_virt_env/lib/python3.6/site-packages:/path_to_swans_folder/functions:$PYTHONPATH

Second, the T1-weighted image has to be brain exctracted, for example with the tool BET of FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET)

Third, you have to excecute the following command:

$ python swans.py -i /path_to_the_image/t1_brain.nii.gz [-m trained]
	
Alternatively, you can use the baseline model:

$ python swans.py -i /path_to_the_image/t1_brain.nii.gz -m locked

Models are available under request and have to be inserted inside "models" folder
	
Output
-----------------
Inside the folder /path_to_the_image/ you will find:

segmentations of the left and right hippocampi in MNI space, which are called respectively "t1_brain_LHipp_prediction.nii.gz" and "t1_brain_RHipp_prediction.nii.gz"

segmentations of the left and right hippocampi in subject space, which are called respectively "t1_brain_LHipp_prediction_to_sub.nii.gz" and "t1_brain_RHipp_prediction_to_sub.nii.gz"

T1 image in the standard MNI space from linear transformation "t1_brain_to_std_flirt.nii.gz"

T1 image in the standard MNI space from non-linear transformation "t1_brain_to_std_fnirt.nii.gz"

text file containing hippocampal volumes "t1_brain_volumes.txt"

png figure of 3 slices of the segmentation overlapped to T1 image 

Example data
-----------------
Example data to be used as input can be found in https://neurovault.org/collections/12227/
