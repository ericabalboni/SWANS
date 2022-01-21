import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
import numpy.linalg as linalg
import nibabel as nib

import os

def create_inputs(segLR, brain):
    #*
    # Function to create the inputs of the network
    # Selects the slice with the hippocampus in, crops and creates a distance map
    # Returns the slice of brain and the corresponding truth distance map
    # If normalize is true then brain is normalized by the normPercentage
    # 100: 132, 76: 140, 40: 104
    segmentation = segLR[100: 132, 76: 140, 40: 104]
    #Read in the brain file and crop the7 same as the segmentation file
    # 
    brain = brain[100: 132, 76: 140, 40: 104]
    brain = brain/np.percentile(brain, 99)

    return brain, segmentation

def create_left(segLR, brain):
    # Does the same as the create inputs function but for the left hippocampus rather than the right one
    #50:82, 76: 140, 40: 104
    segmentation = segLR[50:82, 76: 140, 40: 104]
    # 
    brain = brain[50:82, 76: 140, 40: 104]
    brain = brain/np.percentile(brain, 99)
    return brain, segmentation

def read_in(path, filename):
    folderStore = nib.load(os.path.join(path, filename))
    fileStore = folderStore.get_fdata()
    filearray = np.array(fileStore)
    filearray = np.cast['int32'](filearray)
    return filearray

def save_file(array, filename, filestore):
    #Save a numpy array as a nii file to use in fsleyes
    nib.save(nib.Nifti1Image(array, filestore.affine), filename)
    return 0

def distance_transform(sample):
    #Creates a distance map for a given volume where things inside the volume have positive values
    #and things outside the brain have negative values
    #distance calcualated is a euclidean distance of how far the voxel is from background

    #Calculate the positive values
    sample [sample != 0 ] = 1
    postiveDistanceMap = distance_transform_edt(sample)

    #Inverse the binary image to get the negative distance map
    sample [sample == 0] = 2
    sample [sample == 1] = 0
    sample [sample == 2] = 1
    negativeDistanceMap = distance_transform_edt(sample)

    #Total distance map is the difference between the two maps
    totalDistanceMap = postiveDistanceMap - negativeDistanceMap

    return totalDistanceMap


def createElipsoid(im_size, scale):
    #Creates a solid elipsoid at the center and of the scale stated in a volume otherwise of size zero

    #Create a volume of zeros of the same size as the input image

    vol = np.zeros(im_size)

    #Create the elipsoid and center in matrix form
    A = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    center = [12, 24, 24]             #Incase want to hard code the center

    #Find the rotation matrix and radii of the Axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    u = np.linspace(0.0, 20.0*np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))*scale
    y = radii[1] * np.outer(np.sin(u), np.sin(v))*scale
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))*scale

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    xround = np.round(x).astype(int)
    yround = np.round(y).astype(int)
    zround = np.round(z).astype(int)

    #Update the zero volume
    vol[xround, yround, zround] = 1

    #fill in the holes to make a solid binary image
    vol = binary_fill_holes(vol)
    return vol

def createElipsoid2(im_size, scale = 15, center = [0, 0, 0]):
    #Creates a solid elipsoid at the center and of the scale stated in a volume otherwise of size zero

    #Create a volume of zeros of the same size as the input image

    vol = np.zeros(im_size)

    #Create the elipsoid and center in matrix form
    A = np.array([[2,0,0],[0,2,0],[0,0,2]])

    #Find the rotation matrix and radii of the Axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    u = np.linspace(0.0, 20.0*np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))*scale
    y = radii[1] * np.outer(np.sin(u), np.sin(v))*scale
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))*scale

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    xround = np.round(x).astype(int)
    yround = np.round(y).astype(int)
    zround = np.round(z).astype(int)

    #Update the zero volume
    vol[xround, yround, zround] = 1

    #fill in the holes to make a solid binary image
    vol = binary_fill_holes(vol)
    ellipse1=np.zeros((32,64,64))
    for i in range(32):
        for j in range(64):
            for k in range(64):
                if vol[i,j,k]==False:
                    ellipse1[i,j,k]=0
                if vol[i,j,k]==True:
                    ellipse1[i,j,k]=1
    return ellipse1

def create_sphere(im_size, radius=4, center=[0,0,0]):
    vol=np.zeros(im_size)
    coords = np.ogrid[:im_size[0], :im_size[1], :im_size[2]]
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
    vol = 1*(distance <= radius)
    vol = vol.astype(float)
    return vol





def create_inputs_brain(brain):
    brain = brain[100: 132, 76: 140, 40: 104]
    brain = brain/np.percentile(brain, 99)

    return brain

def create_left_brain(brain):
    
    brain = brain[50:82, 76: 140, 40: 104]
    brain = brain/np.percentile(brain, 99)
    return brain

def create_seg(segLR):
   segmentation = segLR[100: 132, 76: 140, 40: 104]
   return segmentation

def create_seg_left(segLR):
    segmentation = segLR[50:82, 76: 140, 40: 104]
    return segmentation



"""
im_size=(32,64,64)
sphere = np.zeros((32,64,64))
sphere = createElipsoid2(im_size, 11, [8,10,10])


sphere=read_in('/storages/FILESERVER/WORK/Hippocampus_Project/MCI_3T/M001', 't1_brain_LHipp_prediction_to_sub.nii.gz')


plt.imshow(sphere[8,:,:], cmap='Greys')


y,z,x=sphere.nonzero()
ax = plt.axes(projection='3d')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.scatter(x, y, z, zdir='z', c= 'black')
    

    


im_size = (32, 64, 64) 
#ellipse1=np.zeros((32,64,64))

sphere=createElipsoid2(im_size, scale=8, center=[13,24,18])




print(np.histogram(ellipse))
"""




