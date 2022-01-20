import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
#from skimage.draw import ellipse
import numpy.linalg as linalg
import nibabel as nib
import matplotlib
matplotlib.use("TkAgg")              # Deals with the matplotlib bug and makes sure that it runs properly
import matplotlib.pyplot as plt

def create_inputs(segfile, brainfile):
    #*
    # Function to create the inputs of the network
    # Selects the slice with the hippocampus in, crops and creates a distance map
    # Returns the slice of brain and the corresponding truth distance map
    # If normalize is true then brain is normalized by the normPercentage
    # *

    #Read in the segmentation file, binarise and crop
    segmentation, filestore = read_in(segfile)
    print(segmentation.shape)
    segmentation1 = segmentation[107: 139, 76: 140, 40: 104]
    #Read in the brain file and crop the7 same as the segmentation file
    brain, filestore = read_in(brainfile)
    print(brain.shape)
    # 113: 133, 80: 148, 40: 108
    brain = brain[107: 139, 76: 140, 40: 104]
    brain = brain/np.percentile(brain, 99)

    return brain, segmentation1

def create_left(segfile, brainfile):
    # Does the same as the create inputs function but for the left hippocampus rather than the right one
    # Read in the segmentation file and crop
    segmentation, _ = read_in(segfile)
    segmentation = segmentation[59:91, 76: 140, 40: 104]
    # 65:85, 80:148, 40:108
    brain, _ = read_in(brainfile)
    brain = brain[59:91, 76: 140, 40: 104]
    brain = brain/np.percentile(brain, 99)
    return brain, segmentation

def read_in(filename):
    folderStore = nib.load(filename)
    fileStore = folderStore.get_data()
    filearray = np.array(fileStore)
    return filearray, folderStore

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


def createElipsoid(im_size, scale = 15):
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
    return vol
