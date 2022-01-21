from nipype.interfaces import fsl


def flirt(inputfile, ref, out_file, out_mat):
    #FLIRT
    flt = fsl.FLIRT()
    flt.inputs.in_file = inputfile
    flt.inputs.reference = ref
    flt.inputs.output_type = 'NIFTI_GZ'
    flt.inputs.out_file = out_file
    flt.inputs.out_matrix_file = out_mat
    #flt.cmdline 
    #'flirt -in structural.nii -ref mni.nii -out structural_flirt.nii.gz -omat structural_flirt.mat -bins 640 -searchcost mutualinfo'
    flt.run()
    
    
def fnirt(inputfile, ref, affine, output, warp):    
    #FNIRT
    fnt = fsl.FNIRT(affine_file=affine)
    fnt.run(ref_file=ref, in_file=inputfile, warped_file=output, 
            spline_order=2, hessian_precision='float',
            field_file=warp) 
    
def applywarp(inputfile, output, ref, warp): 
    #APPLYWARP
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = inputfile
    aw.inputs.ref_file = ref
    aw.inputs.field_file = warp
    aw.inputs.out_file = output
    aw.run() 
    
def invwarp(inputfile, ref, output):   
    #INVWARP
    invwarp = fsl.InvWarp()
    invwarp.inputs.warp = inputfile
    invwarp.inputs.reference = ref
    invwarp.inputs.output_type = "NIFTI_GZ"
    invwarp.inputs.inverse_warp= output
    invwarp.run()

    
def fslmerge(input_string, output):  
    #MERGE
    merger = fsl.Merge()
    merger.inputs.in_files = input_string
    #['functional2.nii', 'functional3.nii']
    merger.inputs.dimension = 't'
    merger.inputs.output_type = 'NIFTI_GZ'
    merger.inputs.merged_file = output
    merger.run()
    #merger.cmdline
    #'fslmerge -t functional2_merged.nii.gz functional2.nii functional3.nii'
    
def thr(inputfile, value):
    thresh=fsl.Threshold()
    thresh.inputs.in_file=inputfile
    thresh.inputs.thresh=value
    thresh.inputs.out_file=inputfile
    thresh.run()
    
def binarize(inputfile):
    b=fsl.UnaryMaths()
    b.inputs.in_file=inputfile
    b.inputs.operation='bin'
    b.inputs.out_file=inputfile
    b.run()
    
def split(inputfile, basename):
    s=fsl.Split()
    s.inputs.in_file=inputfile
    s.inputs.dimension='t'
    s.inputs.out_base_name=basename
    s.run()
    
def volume(input_map):
    vol=fsl.ImageStats()
    vol.inputs.in_file=input_map
    vol.inputs.op_string='-V'
    vol.run()
    
    
    
def fslroi(input_image,output_image, X_min,X_size,Y_min,Y_size,Z_min,Z_size):
    roi=fsl.utils.ExtractROI()
    roi.run(in_file=input_image,
    roi_file=output_image,
    x_min=X_min,x_size=X_size,y_min=Y_min,y_size=Y_size,z_min=Z_min,z_size=Z_size)

def bet(input_image,output_image, c_x, c_y, c_z):
    my_bet=fsl.BET()
    my_bet.inputs.in_file = input_image
    my_bet.inputs.out_file= output_image
    my_bet.inputs.center=[c_x, c_y, c_z]
    my_bet.run()    

def bet1(input_image,output_image, c_x, c_y, c_z):
    my_bet=fsl.BET()
    my_bet.inputs.in_file = input_image
    my_bet.inputs.out_file= output_image
    my_bet.inputs.center=[c_x, c_y, c_z]
    my_bet.inputs.frac = 0.2
    my_bet.inputs.vertical_gradient=-0.35
    my_bet.run()
    
        
