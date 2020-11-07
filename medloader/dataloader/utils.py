import pdb
import copy
import time
import tqdm
import json
import urllib
import traceback
import numpy as np
from pathlib import Path
import SimpleITK as sitk # sitk.Version.ExtendedVersionString()

import medloader.dataloader.config as config

if config.IPYTHON_FLAG : tqdm_func = tqdm.tqdm_notebook
else                   : tqdm_func = tqdm.tqdm

############################################################
#                    DOWNLOAD RELATED                      #
############################################################

class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_zip(url_zip, filepath_zip, filepath_output, position_id=0):
    import urllib
    if config.IPYTHON_FLAG: position_id=0
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='[Download]' + str(url_zip.split('/')[-1]), position=position_id, leave=True) as pbar:
        urllib.request.urlretrieve(url_zip, filename=filepath_zip, reporthook=pbar.update_to)
    read_zip(filepath_zip, filepath_output)

def read_zip(filepath_zip, filepath_output, leave=True, position_id=0):
    import zipfile
    zip_fp = zipfile.ZipFile(filepath_zip, 'r')
    zip_fp_members = zip_fp.namelist()
    if config.IPYTHON_FLAG: position_id=0
    with tqdm_func(total=len(zip_fp_members), desc='[Unzip]' + str(filepath_zip.parts[-1]), leave=leave, position=position_id) as pbar_zip:
        for member in zip_fp_members:
            zip_fp.extract(member, filepath_output)
            pbar_zip.update(1)


############################################################
#            ITK (Registration) RELATED                    #
############################################################

def get_parameter_affine():
    import itk
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(parameter_map_rigid)
    # parameter_object.SetParameter(0, "AutomaticTransformInitialization", "true")
    return parameter_object

def get_registration_parameters_details(registration_parameters, verbose=False):
    if verbose:
        print ('')
        print (' ----------------------- REGISTRATION PARAMETERS ---------------------- ')
    registration_parameter_values = {}
    for i in range(registration_parameters.GetNumberOfParameterMaps()):
        transform = registration_parameters.GetParameterMap(i)
        transform_type = transform['Transform'][0]
        transform_vals = [float(each) for each in transform['TransformParameters']]
        if verbose:
            print (' - type: ', transform_type, ' || shape:',  np.array(transform_vals).shape)
        registration_parameter_values[transform_type] = np.array(transform_vals)
        if transform_type != config['transform']['bspline']:
            if verbose: print (' -- vals: ', transform_vals)
    if verbose:
        print (' --------------------------------------------------------------------- ')
        print ('')
    return registration_parameter_values

def write_parameter_affine(affine_transform_obj, output_directory, output_filename, write=False):
    """
    - Ref: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html
    """
    import SimpleITK as sitk
    
    transform_vals = [float(each) for each in affine_transform_obj['TransformParameters']]
    affine = sitk.AffineTransform(3)
    affine.SetTranslation(transform_vals[-3:])
    affine.SetMatrix(transform_vals[:-3])
    affine.SetCenter([float(each) for each in affine_transform_obj['CenterOfRotationPoint']])
    
    if write:
        filename = Path(output_directory).joinpath(output_filename)
        # print (' - affine params filename: ', filename.parts[-2:])
        sitk.WriteTransform(affine, str(filename))
        
    return affine

def write_parameter_bspline(bspline_transform_obj, output_directory, output_filename, write=False):
    import SimpleITK as sitk
    
    bspline = sitk.BSplineTransform(3,1)
    bspline.SetTransformDomainOrigin([float(each) for each in bspline_transform_obj['Origin']]) # from fixed image
    bspline.SetTransformDomainPhysicalDimensions([int(each) for each in bspline_transform_obj['Size']]) # from fixed image
    bspline.SetTransformDomainDirection([float(each) for each in bspline_transform_obj['Direction']]) # from fixed image
    
    fixedParams = [int(each) for each in bspline_transform_obj['GridSize']]
    fixedParams += [float(each) for each in bspline_transform_obj['GridOrigin']]
    fixedParams += [float(each) for each in bspline_transform_obj['GridSpacing']]
    fixedParams += [float(each) for each in bspline_transform_obj['GridDirection']]
    bspline.SetFixedParameters(fixedParams)
    bspline.SetParameters([float(each) for each in bspline_transform_obj['TransformParameters']])

    
    if write:
        filename = Path(output_directory).joinpath(output_filename)
        print (' - bspline params filename: ', filename.parts[-2:])
        sitk.WriteTransform(bspline, str(filename))
        
    return bspline

def write_registration_parameters(registration_parameters, output_directory, output_filenames, write_individual):
    
    parameter_objs = {}

    affine = None
    bspline = None
    
    for i in range(registration_parameters.GetNumberOfParameterMaps()):
        transform_obj = registration_parameters.GetParameterMap(i)
        transform_type = transform_obj['Transform'][0]
        
        # AFFINE
        if transform_type == config.REG_CONFIG['transform']['affine']:
            affine = write_parameter_affine(transform_obj, output_directory, output_filenames['affine'], write=write_individual)
            parameter_objs['affine'] = affine
            
        # BSPLINE
        elif transform_type == config.REG_CONFIG['transform']['bspline']:
            bspline = write_parameter_bspline(transform_obj, output_directory, output_filenames['bspline'], write=write_individual)    
            parameter_objs['bspline'] = bspline
    
    # Composite Transform
    if 'affine' in parameter_objs and 'bspline' in parameter_objs:
        # Ref: https://simpleitk.readthedocs.io/en/master/migrationGuide2.0.html
        composite = sitk.CompositeTransform([affine, bspline])
        filename = Path(output_directory).joinpath('composite.tfm')
        print (' - composite params filename: ', filename.parts[-2:])
        sitk.WriteTransform(composite, str(filename))

def copy_affine_transform(reg_param_new, transform_affine_old, keys_change):

    transform_affine_new = reg_param_new.GetDefaultParameterMap('affine')
    
    # Assign keys
    for keyname in transform_affine_old.keys():
        transform_affine_new[keyname] = transform_affine_old[keyname]
        if keyname in keys_change:
            transform_affine_new[keyname] = keys_change[keyname]

    # Remove keys
    new_keys = transform_affine_new.keys()
    old_keys = transform_affine_old.keys()
    for keyname in new_keys:
        if keyname not in old_keys:
            transform_affine_new.erase(keyname)
    
    reg_param_new.AddParameterMap(transform_affine_new)
    return reg_param_new
    
def copy_bspline_transform(reg_param_new, transform_bspline_old, keys_change):

    transform_bspline_new = reg_param_new.GetDefaultParameterMap('bspline')
    
    # Assign keys
    for keyname in transform_bspline_old.keys():
        transform_bspline_new[keyname] = transform_bspline_old[keyname]
        if keyname in keys_change:
            transform_bspline_new[keyname] = keys_change[keyname]

    # Remove keys
    new_keys = transform_bspline_new.keys()
    old_keys = transform_bspline_old.keys()
    for keyname in new_keys:
        if keyname not in old_keys:
            transform_bspline_new.erase(keyname)
    
    reg_param_new.AddParameterMap(transform_bspline_new)
    return reg_param_new

def copy_reg_params(reg_param_old, keys_change):
    import itk
    reg_param_new = itk.ParameterObject.New()
    
    for i in range(reg_param_old.GetNumberOfParameterMaps()):
        transform_obj_old = reg_param_old.GetParameterMap(i)
        transform_type_old = transform_obj_old['Transform'][0]
        
        # AFFINE
        if transform_type_old == config.REG_CONFIG['transform']['affine']:
            reg_param_new = copy_affine_transform(reg_param_new, transform_obj_old, keys_change)
        
        # BSPLINE
        if transform_type_old == config.REG_CONFIG['transform']['bspline']:
            reg_param_new = copy_affine_transform(reg_param_new, transform_obj_old, keys_change)
    
    return reg_param_new

############################################################
#                    ITK RELATED                           #
############################################################

def imwrite_sitk(data, filepath, dtype, compression=True):
    import itk

    def convert_itk_to_sitk(image_itk, dtype):
        
        img_array = itk.GetArrayFromImage(image_itk)
        if dtype in ['short', 'int16']:    
            img_array = np.array(img_array, dtype=np.int16)
        elif dtype in ['unsigned int', 'uint8']:
            img_array = np.array(img_array, dtype=np.uint8)
            
        image_sitk = sitk.GetImageFromArray(img_array, isVector=image_itk.GetNumberOfComponentsPerPixel()>1)
        image_sitk.SetOrigin(tuple(image_itk.GetOrigin()))
        image_sitk.SetSpacing(tuple(image_itk.GetSpacing()))
        image_sitk.SetDirection(itk.GetArrayFromMatrix(image_itk.GetDirection()).flatten())
        return image_sitk

    writer = sitk.ImageFileWriter()    
    writer.SetFileName(str(filepath))
    writer.SetUseCompression(compression)
    if 'SimpleITK' not in str(type(data)):
        writer.Execute(convert_itk_to_sitk(data, dtype))
    else:
        writer.Execute(data)

def read_itk(img_url):
    import itk
    if Path(img_url).exists():
        return itk.imread(img_url, itk.F)
    else:
        print (' - [read_itk()] Path does not exist: ', img_url)
        return None

def array_to_sitk(array_input, size=None, origin=None, spacing=None, direction=None, is_vector=False, im_ref=None):
    """
    This function takes an array and converts it into a SimpleITK image.

    Parameters
    ----------
    array_input: numpy
        The numpy array to convert to a SimpleITK image
    size: tuple, optional
        The size of the array
    origin: tuple, optional
        The origin of the data in physical space
    spacing: tuple, optional
        Spacing describes the physical sie of each pixel
    direction: tuple, optional
        A [nxn] matrix passed as a 1D in a row-major form for a nD matrix (n=[2,3]) to infer the orientation of the data
    is_vector: bool, optional
        If isVector is True, then the Image will have a Vector pixel type, and the last dimension of the array will be considered the component index.
    im_ref: sitk image
        An empty image with meta information
    
    Ref: https://github.com/hsokooti/RegNet/blob/46f345d25cd6a1e0ee6f230f64c32bd15b7650d3/functions/image/image_processing.py#L86
    """
    import SimpleITK as sitk
    verbose = False
    
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    if spacing is None:
        spacing = [1, 1, 1]  # the voxel spacing
    if direction is None:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if size is None:
        size = np.array(array_input).shape

    """
    ITK has a GetPixel which takes an ITK Index object as an argument, which is ordered as (x,y,z). 
    This is the convention that SimpleITK's Image class uses for the GetPixel method and slicing operator as well. 
    In numpy, an array is indexed in the opposite order (z,y,x)
    """
    sitk_output = sitk.GetImageFromArray(np.moveaxis(array_input, [0,1,2], [2,1,0]), isVector=is_vector) # np([H,W,D]) -> np([D,W,H]) -> sitk([H,W,D])
    
    if im_ref is None:
        sitk_output.SetOrigin(origin)
        sitk_output.SetSpacing(spacing)
        sitk_output.SetDirection(direction)
    else:
        sitk_output.SetOrigin(im_ref.GetOrigin())
        sitk_output.SetSpacing(im_ref.GetSpacing())
        sitk_output.SetDirection(im_ref.GetDirection())

    return sitk_output

def sitk_to_array(sitk_image):
    array = sitk.GetArrayFromImage(sitk_image)
    array = np.moveaxis(array, [0,1,2], [2,1,0]) # [D,W,H] --> [H,W,D]
    return array

def resampler_sitk(image_sitk, spacing=None, scale=None, im_ref=None, im_ref_size=None, default_pixel_value=0, interpolator=None, dimension=3):
    """
    :param image_sitk: input image
    :param spacing: desired spacing to set
    :param scale: if greater than 1 means downsampling, less than 1 means upsampling
    :param im_ref: if im_ref available, the spacing will be overwritten by the im_ref.GetSpacing()
    :param im_ref_size: in sikt order: x, y, z
    :param default_pixel_value:
    :param interpolator:
    :param dimension:
    :return:
    """

    import math
    import SimpleITK as sitk

    if spacing is None and scale is None:
        raise ValueError('spacing and scale cannot be both None')
    if interpolator is None:
        interpolator = sitk.sitkBSpline # sitk.Linear, sitk.Nearest

    if spacing is None:
        spacing = tuple(i * scale for i in image_sitk.GetSpacing())
        if im_ref_size is None:
            im_ref_size = tuple(round(i / scale) for i in image_sitk.GetSize())

    elif scale is None:
        ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
        if im_ref_size is None:
            im_ref_size = tuple(math.ceil(size_dim * ratio[i]) - 1 for i, size_dim in enumerate(image_sitk.GetSize()))
    else:
        raise ValueError('spacing and scale cannot both have values')

    if im_ref is None:
        im_ref = sitk.Image(im_ref_size, sitk.sitkInt8)
        im_ref.SetOrigin(image_sitk.GetOrigin())
        im_ref.SetDirection(image_sitk.GetDirection())
        im_ref.SetSpacing(spacing)
    

    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref,
                                            default_pixel_value=default_pixel_value,
                                            interpolator=interpolator)
    return resampled_sitk

def resampler_by_transform(im_sitk, dvf_t, im_ref=None, default_pixel_value=0, interpolator=None):
    import SimpleITK as sitk

    if im_ref is None:
        im_ref = sitk.Image(dvf_t.GetDisplacementField().GetSize(), sitk.sitkInt8)
        im_ref.SetOrigin(dvf_t.GetDisplacementField().GetOrigin())
        im_ref.SetSpacing(dvf_t.GetDisplacementField().GetSpacing())
        im_ref.SetDirection(dvf_t.GetDisplacementField().GetDirection())

    if interpolator is None:
        interpolator = sitk.sitkBSpline

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(dvf_t)

    # [DEBUG]
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    out_im = resampler.Execute(im_sitk)
    return out_im

def save_as_mha_mask(data_dir, patient_id, voxel_mask, voxel_img_headers):
    
    voxel_save_folder = Path(data_dir).joinpath(patient_id)
    Path(voxel_save_folder).mkdir(parents=True, exist_ok=True)

    orig_origin = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN]
    orig_pixel_spacing = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING]

    voxel_mask_sitk = array_to_sitk(voxel_mask.astype(config.DATATYPE_VOXEL_MASK)
                            , origin=orig_origin, spacing=orig_pixel_spacing)
    path_voxel_mask = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_TUMOR_3D)
    sitk.WriteImage(voxel_mask_sitk, str(path_voxel_mask), useCompression=True)

def save_as_mha(data_dir, patient_id, voxel_img, voxel_img_headers, voxel_mask, voxel_img_reg_dict={}, resample_save=True):
    try:
        """
        Thi function converts the raw numpy data into a SimpleITK image and saves as .mha

        Parameters
        ----------
        data_dir: Path
            The path where you would like to save the data
        patient_id: str
            A reference to the patient
        voxel_img: numpy
            A nD numpy array with [H,W,D] format containing radiodensity data in Hounsfield units
        voxel_img_headers: dict
            A python dictionary containing information on 'origin' and 'pixel_spacing'  
        voxel_mask: numpy
            A nD array with labels on each nD voxel
        resample_save: bool
            A boolean variable to indicate whether the function should resample
        """

        # Step 1 - Original Voxel resolution
        ## Step 1.1 - Create save dir
        voxel_save_folder = Path(data_dir).joinpath(patient_id)
        Path(voxel_save_folder).mkdir(parents=True, exist_ok=True)

        ## Step 1.2 - Save img voxel
        orig_origin = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN]
        orig_pixel_spacing = voxel_img_headers[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING]
        voxel_img_sitk = array_to_sitk(voxel_img.astype(config.DATATYPE_VOXEL_IMG)
                            , origin=orig_origin, spacing=orig_pixel_spacing)
        path_voxel_img = Path(voxel_save_folder).joinpath(config.FILENAME_IMG_3D)
        sitk.WriteImage(voxel_img_sitk, str(path_voxel_img), useCompression=True)

        ## Step 1.3 - Save mask voxel
        voxel_mask_sitk = array_to_sitk(voxel_mask.astype(config.DATATYPE_VOXEL_MASK)
                            , origin=orig_origin, spacing=orig_pixel_spacing)
        path_voxel_mask = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_3D)
        sitk.WriteImage(voxel_mask_sitk, str(path_voxel_mask), useCompression=True)

        ## Step 1.4 - Save registration params
        paths_voxel_reg = {}
        for study_id in voxel_img_reg_dict:
            if type(voxel_img_reg_dict[study_id]) == sitk.AffineTransform:
                path_voxel_reg = str(Path(voxel_save_folder).joinpath('{}.tfm'.format(study_id)))
                sitk.WriteTransform(voxel_img_reg_dict[study_id], path_voxel_reg)
                paths_voxel_reg[study_id] = str(path_voxel_reg)

        # Step 2 - Resampled Voxel
        if resample_save:
            new_spacing = config.VOXEL_RESO

            ## Step 2.1 - Save resampled img
            voxel_img_sitk_resampled = resampler_sitk(voxel_img_sitk, spacing=new_spacing, interpolator=sitk.sitkBSpline) 
            voxel_img_sitk_resampled = sitk.Cast(voxel_img_sitk_resampled, sitk.sitkInt16)
            path_voxel_img_resampled = Path(voxel_save_folder).joinpath(config.FILENAME_IMG_RESAMPLED_3D)
            sitk.WriteImage(voxel_img_sitk_resampled, str(path_voxel_img_resampled), useCompression=True)
            interpolator_img = 'sitk.sitkBSpline'
            
            ## Step 2.2 - Save resampled mask
            voxel_mask_sitk_resampled = []
            interpolator_mask = ''
            if 0:
                voxel_mask_sitk_resampled = resampler_sitk(voxel_mask_sitk, spacing=new_spacing, interpolator=sitk.sitkNearestNeighbor)
                interpolator_mask = 'sitk.sitkNearestNeighbor'

            elif 1:
                interpolator_mask = 'sitk.sitkLinear'
                new_size = voxel_img_sitk_resampled.GetSize()
                voxel_mask_resampled = np.zeros(new_size)
                for label_id in np.unique(voxel_mask):
                    if label_id != 0:
                        voxel_mask_singlelabel = copy.deepcopy(voxel_mask).astype(config.DATATYPE_VOXEL_MASK)
                        voxel_mask_singlelabel[voxel_mask_singlelabel != label_id] = 0
                        voxel_mask_singlelabel[voxel_mask_singlelabel == label_id] = 1
                        voxel_mask_singlelabel_sitk = array_to_sitk(voxel_mask_singlelabel
                            , origin=orig_origin, spacing=orig_pixel_spacing)
                        voxel_mask_singlelabel_sitk_resampled = resampler_sitk(voxel_mask_singlelabel_sitk, spacing=new_spacing
                                    , interpolator=sitk.sitkLinear) 
                        if 0:
                            voxel_mask_singlelabel_sitk_resampled = sitk.Cast(voxel_mask_singlelabel_sitk_resampled, sitk.sitkUInt8)
                            voxel_mask_singlelabel_array_resampled = sitk_to_array(voxel_mask_singlelabel_sitk_resampled)
                            idxs = np.argwhere(voxel_mask_singlelabel_array_resampled > 0)
                        else:
                            voxel_mask_singlelabel_array_resampled = sitk_to_array(voxel_mask_singlelabel_sitk_resampled)
                            idxs = np.argwhere(voxel_mask_singlelabel_array_resampled >= 0.5)
                        voxel_mask_resampled[idxs[:,0], idxs[:,1], idxs[:,2]] = label_id

                voxel_mask_sitk_resampled = array_to_sitk(voxel_mask_resampled 
                                , origin=orig_origin, spacing=new_spacing)

            voxel_mask_sitk_resampled = sitk.Cast(voxel_mask_sitk_resampled, sitk.sitkUInt8)    
            path_voxel_mask_resampled = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_RESAMPLED_3D)
            sitk.WriteImage(voxel_mask_sitk_resampled, str(path_voxel_mask_resampled), useCompression=True)

            # Step 2.3 - Save voxel info for resampled data
            voxel_mask_resampled_data = sitk_to_array(voxel_mask_sitk_resampled)
            brainstem_idxs = np.argwhere(voxel_mask_resampled_data == 1)
            brainstem_idxs_mean = np.mean(brainstem_idxs, axis=0)
            path_voxel_headers = Path(voxel_save_folder).joinpath(config.FILENAME_VOXEL_INFO)

            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED] = {config.KEYNAME_MEAN_BRAINSTEAM : brainstem_idxs_mean.tolist()}
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_PIXEL_SPACING] = voxel_img_sitk_resampled.GetSpacing()
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_ORIGIN] = voxel_img_sitk_resampled.GetOrigin()
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_SHAPE] = voxel_img_sitk_resampled.GetSize()
            voxel_img_headers[config.TYPE_VOXEL_RESAMPLED][config.TYPE_VOXEL_ORIGSHAPE] = {
                config.KEYNAME_INTERPOLATOR_IMG: interpolator_img
                , config.KEYNAME_INTERPOLATOR_MASK: interpolator_mask
            }


            write_json(voxel_img_headers, path_voxel_headers)   

        ## Step 3 - Save img headers
        path_voxel_headers = Path(voxel_save_folder).joinpath(config.FILENAME_VOXEL_INFO)
        write_json(voxel_img_headers, path_voxel_headers)

        return str(path_voxel_img), str(path_voxel_mask), paths_voxel_reg
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def read_mha(path_file):
    try:
        
        if Path(path_file).exists():
            img_mha = sitk.ReadImage(str(path_file))
            return img_mha
        else:
            print (' - [ERROR][read_mha()] Path issue: path_file: ', path_file)
            pdb.set_trace()
            
    except:
        traceback.print_exc()
        pdb.set_trace()

############################################################
#                    3D VOXEL RELATED                      #
############################################################

def get_self_label_id(label_name_orig, RAW_TO_SELF_LABEL_MAPPING, LABEL_MAP):
        
        try:
            if label_name_orig in RAW_TO_SELF_LABEL_MAPPING: 
                if RAW_TO_SELF_LABEL_MAPPING[label_name_orig] in LABEL_MAP:
    	            return LABEL_MAP[RAW_TO_SELF_LABEL_MAPPING[label_name_orig]]
                else:
                    return 0
            else:
                return 0
        except:
            traceback.print_exc()
            print (' - [ERROR][get_global_label_id()] label_name_orig: ', label_name_orig)
            pdb.set_trace()

def extract_contours_from_rtstruct(rtstruct_ds, RAW_TO_SELF_LABEL_MAPPING=None, LABEL_MAP=None):

    # Step 0 - Init
    contours = []
    labels_debug = {}

    # Step 1 - Loop and extract all different contours
    for i in range(len(rtstruct_ds.ROIContourSequence)):
        try:

            # Step 1.1 - Get contour 
            contour = {}
            contour['color'] = list(rtstruct_ds.ROIContourSequence[i].ROIDisplayColor)
            contour['contours'] = [s.ContourData for s in rtstruct_ds.ROIContourSequence[i].ContourSequence]
            contour['name'] = rtstruct_ds.StructureSetROISequence[i].ROIName
            if RAW_TO_SELF_LABEL_MAPPING is None and LABEL_MAP is None:
                contour['number'] = rtstruct_ds.ROIContourSequence[i].ReferencedROINumber
                assert contour['number'] == rtstruct_ds.StructureSetROISequence[i].ROINumber
            else:
                contour['number'] = get_self_label_id(contour['name'], RAW_TO_SELF_LABEL_MAPPING, LABEL_MAP)
            
            ## DEBUG [Remove me]
            if 'Cere' in contour['name'] or 'Cerebel' in contour['name']:
                print (' - ', contour['name'], ' || ')
                print ('')
            
            # Step 1.2 - Keep or not condition
            if contour['number'] > 0:
                contours.append(contour)

            # Step 1.3 - Some debugging
            labels_debug[contour['name']] = {'id': len(labels_debug) + 1}

        except:
            traceback.print_exc()
            pdb.set_trace()
    
    # Step 2 - Order your contours
    if len(contours):
        contours = list(sorted(contours, key = lambda obj: obj['number']))
        
    return contours, labels_debug

def process_contours(contours, params, voxel_mask_data):
    """
    Convert contours to voxel values
    """
    import skimage

    # Step 0 - Init
    class_ids = []
    study_class_id = None
    study_slice_id = None

    # Step 1 - Get some position and spacing params 
    z = params[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ZVALS]
    pos_r = params[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN][1]
    spacing_r = params[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING][1]
    pos_c = params[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN][0]
    spacing_c = params[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING][0]
    shape = params[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_SHAPE]

    # Step 2 - Loop over countours
    for _, contour in enumerate(contours):

        try:
            
            if study_class_id == contour['number']:
                print (' - [get_mask()] contour-number:', contour['number'], ' || contour-label:', contour['name'])

            class_id = int(contour['number'])
            class_ids.append(class_id)
            for c in contour['contours']:
                nodes = np.array(c).reshape((-1, 3))
                if len(nodes) > 1:
                    assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                    z_index = z.index(nodes[0, 2])
                    rows = (nodes[:, 1] - pos_r) / spacing_r  #pixel_idx = f(real_world_idx, ct_resolution)
                    cols = (nodes[:, 0] - pos_c) / spacing_c
                    rr, cc = skimage.draw.polygon(rows, cols)
                    if class_id == study_class_id and z_index == study_slice_id:
                        print (' - z_index: ', z_index)
                        print (' --- rr, cc: ', np.array(rr), np.array(cc))
                    
                    # voxel_mask_data[rr, cc, z_index] = class_id # This shows incorrectly in 3D Slicer for saggital/coronal views
                    voxel_mask_data[cc, rr, z_index] = class_id

        except:
            print (' - [ERROR][get_mask()] contour-number:', contour['number'], ' || contour-label:', contour['name'])
            traceback.print_exc()
    
    return voxel_mask_data

def split_into_overlapping_grids(len_total, len_grid, len_overlap, res_type='boundary'):
  res_range = []
  res_boundary = []

  A = np.arange(len_total)
  l_start = 0
  l_end = len_grid
  while(l_end < len(A)):
    res_range.append(np.arange(l_start, l_end))
    res_boundary.append([l_start, l_end])
    l_start = l_start + len_grid - len_overlap
    l_end = l_start + len_grid
  
  res_range.append(np.arange(len(A)-len_grid, len(A)))
  res_boundary.append([len(A)-len_grid, len(A)])
  if res_type == 'boundary':
    return res_boundary
  elif res_type == 'range':
    return res_range

def extract_numpy_from_dcm(patient_dir):
    """
    Given the path of the folder containing the .dcm files, this function extracts 
    a numpy array by combining them

    Parameters
    ----------
    patient_dir: Path
        The path of the folder containing the .dcm files
    """

    try:
        import pydicom
        from pathlib import Path

        if Path(patient_dir).exists():
            slices = [pydicom.filereader.dcmread(path_ct) for path_ct in Path(patient_dir).iterdir()]
            slices = list(filter(lambda x: 'ImagePositionPatient' in x, slices))
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # inferior to superior
            voxel_img_data = np.stack([s.pixel_array.T for s in slices], axis=-1) # [row, col, plane]
            return slices, voxel_img_data
        else:
            return [],[]
    except:
        traceback.print_exc()
        pdb.set_trace()

def perform_hu(voxel_img, intercept, slope):
    """
    Rescale Intercept != 0, Rescale Slope != 1 or Dose Grid Scaling != 1.
    The pixel data has not been transformed according to these values.
    Consider using the module ApplyDicomPixelModifiers after
    importing the volume to transform the image data appropriately.
    """
    try:
        import copy

        slope = np.float(slope)
        intercept = np.float(intercept)
        voxel_img_hu = copy.deepcopy(voxel_img).astype(np.float64)

        # Convert to Hounsfield units (HU)    
        if slope != 1:
            voxel_img_hu = slope * voxel_img_hu
            
        voxel_img_hu += intercept

        return voxel_img_hu.astype(config.DATATYPE_VOXEL_IMG)
    except:
        traceback.print_exc()
        pdb.set_trace()

def print_final_message():
    print ('')
    print (' - Note: You can view the 3D data in visualizers like MeVisLab or 3DSlicer')
    print (' - Note: Raw Voxel Data ({}/{}) is in Hounsfield units (HU) with int16 datatype'.format(config.FILENAME_IMG_3D, config.FILENAME_IMG_RESAMPLED_3D))
    print ('')

############################################################
#                SAVING/READING RELATED                    #
############################################################

def save_csv(filepath, data_array):
    Path(filepath).parent.absolute().mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, data_array, fmt='%s')

def read_csv(filepath):
    data = np.loadtxt(filepath, dtype='str')
    return data

def write_json(json_data, json_filepath):

    Path(json_filepath.parent.absolute()).mkdir(parents=True, exist_ok=True)

    with open(str(json_filepath), 'w') as fp:
        json.dump(json_data, fp, indent=4, cls=NpEncoder)

def read_json(json_filepath):

    if Path(json_filepath.parent.absolute()).exists():
        with open(str(json_filepath), 'r') as fp:
            data = json.load(fp)
            return data
    else:
        print (' - [ERROR][read_json()] json_filepath does not exist: ', json_filepath)
        return None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_nrrd(filepath, data, spacing):
    nrrd_headers = {'space':'left-posterior-superior', 'kinds': ['domain', 'domain', 'domain'], 'encoding':'gzip'}
    space_directions = np.zeros((3,3), dtype=np.float32)
    space_directions[[0,1,2],[0,1,2]] = np.array(spacing)
    nrrd_headers['space directions'] = space_directions

    import nrrd
    nrrd.write(str(filepath), data, nrrd_headers)

############################################################
#                    DOWNLOAD RELATED                      #
############################################################

class TCIAClient:
    """
     - Ref: https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+%28REST+API%29+Usage+Guide
     - Ref: https://github.com/TCIA-Community/TCIA-API-SDK/tree/master/tcia-rest-client-python/src
    """
    GET_IMAGE = "getImage"
    GET_MANUFACTURER_VALUES = "getManufacturerValues"
    GET_MODALITY_VALUES = "getModalityValues"
    GET_COLLECTION_VALUES = "getCollectionValues"
    GET_BODY_PART_VALUES = "getBodyPartValues"
    GET_PATIENT_STUDY = "getPatientStudy"
    GET_SERIES = "getSeries"
    GET_PATIENT = "getPatient"
    GET_SERIES_SIZE = "getSeriesSize"
    CONTENTS_BY_NAME = "ContentsByName"

    def __init__(self, baseUrl, resource):
        self.baseUrl = baseUrl + "/" + resource
        self.STATUS_OK = 200
        self.DECODER = 'utf-8'

    def execute(self, url, queryParameters={}, verbose=False):
        queryParameters = dict((k, v) for k, v in queryParameters.items() if v)
        queryString = "?%s" % urllib.parse.urlencode(queryParameters)
        requestUrl = url + queryString
        if verbose:
            print (' - [execute()] URL: ', requestUrl)
        request = urllib.request.Request(url=requestUrl, headers={})
        resp = urllib.request.urlopen(request)
        return resp

    def read_response(self, resp):
        if resp.status == self.STATUS_OK:
            return eval(resp.read().decode(self.DECODER))
        else:
            return None

    def get_patient(self,collection = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_PATIENT
        queryParameters = {"Collection" : collection , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_patient_study(self,collection = None , patientId = None , studyInstanceUid = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_PATIENT_STUDY
        queryParameters = {"Collection" : collection , "PatientID" : patientId , "StudyInstanceUID" : studyInstanceUid , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_series(self, collection=None, patientId=None, modality=None ,studyInstanceUid=None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_SERIES
        queryParameters = {"Collection" : collection, "patientId": patientId, "StudyInstanceUID" : studyInstanceUid , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_image(self , seriesInstanceUid , downloadPath, zipFileName):
        try:
            serviceUrl = self.baseUrl + "/query/" + self.GET_IMAGE
            queryParameters = { "SeriesInstanceUID" : seriesInstanceUid }   
            resp = self.execute(serviceUrl, queryParameters)
            filepath = Path(downloadPath).joinpath(zipFileName)
            data = resp.read()
            with open(filepath, 'wb') as fp:
                fp.write(data)
            
            tmp = list(Path(filepath).parts)
            tmp[-1] = tmp[-1].split('.zip')[0]
            filepath_output = Path(*tmp)
            read_zip(filepath, filepath_output, leave=False)

            Path(filepath).unlink()

        except:
            traceback.print_exc()

############################################################
#                           RANDOM                         #
############################################################
def get_name_patient_study_id(meta):
    try:
        meta = np.array(meta)
        meta = str(meta.astype(str))

        meta_split = meta.split('-')
        name = None
        patient_id = None
        study_id = None

        if len(meta_split) == 2:
            name = meta_split[0]
            patient_id = meta_split[1]
        elif len(meta_split) == 3:
            name = meta_split[0]
            patient_id = meta_split[1]
            study_id = meta_split[2]
        
        return name, patient_id, study_id
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def get_dataset_from_zip(meta2, dataset):
    datasets_this = []
    for batch_id in range(len(meta2)):
        dataset_name = meta2[batch_id].numpy().decode('utf-8').split('-')[0]
        datasets_this.append(dataset.get_subdataset(param_name=dataset_name))

    return datasets_this

############################################################
#                    DEBUG RELATED                         #
############################################################

def benchmark_model(model_time):
    time.sleep(model_time)

def benchmark(dataset_generator, model_time=0.1):

    print (' - [benchmark()]')
    t0 = time.time()
    for X,_,_,_ in dataset_generator:
        t1 = time.time()
        benchmark_model(model_time)
        t2 = time.time()
        print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s', '(',X.shape,')')
        # print (X.shape)
        t0 = time.time()

def benchmark_with_profiler(dataset_generator, model_time=0.1):
    """
     - Ref: https://www.tensorflow.org/guide/profiler#profiling_apis
    """
    import tensorflow as tf
    if tf.__version__ in ['2.3.0']:
        with tf.profiler.experimental.Profile('./logdir'): # makes the dir at the end of execution.
            print (' - [utils.benchmark_with_profiler()]')
            t0 = time.time()
            for X,Y,meta1,meta2 in dataset_generator:
                t1 = time.time()
                benchmark_model(model_time)
                t2 = time.time()
                print (' - Data Time: ', round(t1 - t0,5),'s || Model Time: ', round(t2-t1,2),'s', '(',X.shape,')')
                t0 = time.time()

def print_debug_header():
    print (' ============================================== ')
    print ('                   DEBUG ')
    print (' ============================================== ')

############################################################
#             DEBUG RELATED (INCOMPLETE)                   #
############################################################

def get_patient_centre(patient_id):
    print_debug_header()

    patient_path = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_MICCAI2015\\processed\\train\\data_2D\\{}\\img'.format(patient_id))    
    # patient_path = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\{}\\img'.format(patient_id))
    
    for path_img in patient_path.iterdir():
        if 'resampled' not in str(path_img.parts[-1]) and '.png' not in str(path_img.parts[-1]):
            print (path_img.parts[-1])
            get_image_centre(path_img, save=True)

def get_image_centre(path_img, show=False, save=False):
    try:
        # path_img = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\0522c0003\\img\\0522c0003_slice74_img.npy')
        # path_img = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\0522c0003\\img\\0522c0003_slice84_img.npy')
        # path_img = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\0522c0003\\img\\0522c0003_slice91_img.npy')

        import skimage
        import skimage.filters
        import skimage.feature #import blob_dog, blob_log, blob_doh
        if show or save:
            import matplotlib.pyplot as plt

        img = np.load(path_img)
        # img = img[50:-50, 50:-50]

        thresh = skimage.filters.threshold_otsu(img)
        img2 = copy.deepcopy(img)
        img2[img2 >= thresh] = 1
        img2[img2 < thresh] = 0

        if show or save:
            f,axarr = plt.subplots(1,2)
            axarr[0].imshow(img, cmap='gray')
            axarr[1].imshow(img2, cmap='gray')

        img2 = img2.astype(np.float64)
        
        contours = skimage.measure.find_contours(img2, level=0.99)
        for n, contour in enumerate(contours):
            axarr[1].plot(contour[:, 1], contour[:, 0], linewidth=2)

        if show or save:
            plt.suptitle(path_img.parts[-1])
        if show:
            plt.show()
        if save:
            path_savefig_parts = list(Path(path_img).parts)
            path_savefig_parts[-1] = 'tmp_' + str(path_savefig_parts[-1].split('.')[0]) + '.png'
            path_savefig = Path(*path_savefig_parts)
            plt.savefig(path_savefig)
            plt.close()
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def get_image_centre2(path_img, show=False, save=False):
    try:
        # path_img = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\0522c0003\\img\\0522c0003_slice74_img.npy')
        # path_img = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\0522c0003\\img\\0522c0003_slice84_img.npy')
        # path_img = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data2\\HaN_TCIACetuximab\\processed\\data_2D\\0522c0003\\img\\0522c0003_slice91_img.npy')

        import skimage
        import skimage.filters
        import skimage.feature #import blob_dog, blob_log, blob_doh
        if show or save:
            import matplotlib.pyplot as plt

        img = np.load(path_img)
        img = img[50:-50, 50:-50]
        thresh = skimage.filters.threshold_otsu(img)
        img2 = copy.deepcopy(img)
        img2[img2 >= thresh] = 1
        img2[img2 < thresh] = 0

        if show or save:
            f,axarr = plt.subplots(1,2)
            axarr[0].imshow(img, cmap='gray')
            axarr[1].imshow(img2, cmap='gray')

        blobs = skimage.feature.blob_dog(img2.astype(np.float64), min_sigma=100, max_sigma=300)
        if len(blobs):
            x,y,rad = blobs[0]
            if show or save:
                plt.scatter(x,y, c='red')
                print (img.shape, path_img.parts[-1], blobs)
                c1 = plt.Circle((x, y), rad, linewidth=2, fill=True, alpha=0.4)
                axarr[1].add_patch(c1)
                c2 = plt.Circle((x, y), 160, linewidth=2, fill=True, alpha=0.4)
                axarr[1].add_patch(c2)
        else:
            print (' - [get_image_centre()] No blob found')

        if show or save:
            plt.suptitle(path_img.parts[-1])
        if show:
            plt.show()
        if save:
            path_savefig_parts = list(Path(path_img).parts)
            path_savefig_parts[-1] = 'tmp_' + str(path_savefig_parts[-1].split('.')[0]) + '.png'
            path_savefig = Path(*path_savefig_parts)
            plt.savefig(path_savefig)
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def save_as_mha_wip(data_dir, patient_id, voxel_img, voxel_img_headers, voxel_mask, resample_save=True):
    try:
        """
        Thi function converts the raw numpy data into a SimpleITK image and saves as .mha

        Parameters
        ----------
        data_dir: Path
            The path where you would like to save the data
        patient_id: str
            A reference to the patient
        voxel_img: numpy
            A nD numpy array with [H,W,D] format containing radiodensity data in Hounsfield units
        voxel_img_headers: dict
            A python dictionary containing information on 'origin' and 'pixel_spacing'  
        voxel_mask: numpy
            A nD array with labels on each nD voxel
        resample_save: bool
            A boolean variable to indicate whether the function should resample
        """
        import pdb
        import traceback
        from pathlib import Path
        import SimpleITK as sitk

        # Step 1 - Original Voxel resolution
        ## Step 1.1 - Create save dir
        voxel_save_folder = Path(data_dir).joinpath(patient_id)
        Path(voxel_save_folder).mkdir(parents=True, exist_ok=True)
        
        ## Step 1.2 - Save img voxel
        voxel_img_sitk = array_to_sitk(voxel_img.astype(config.DATATYPE_VOXEL_IMG)
                            , origin=voxel_img_headers[config.KEYNAME_ORIGIN], spacing=voxel_img_headers[config.KEYNAME_PIXEL_SPACING])
        path_voxel_img = Path(voxel_save_folder).joinpath(config.FILENAME_IMG_3D)
        sitk.WriteImage(voxel_img_sitk, str(path_voxel_img))

        ## Step 1.3 - Save mask voxel
        voxel_mask_sitk = array_to_sitk(voxel_mask.astype(config.DATATYPE_VOXEL_MASK)
                            , origin=voxel_img_headers[config.KEYNAME_ORIGIN], spacing=voxel_img_headers[config.KEYNAME_PIXEL_SPACING])
        path_voxel_mask = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_3D) 
        sitk.WriteImage(voxel_mask_sitk, str(path_voxel_mask))

        # Step 2 - Resampled Voxel
        if resample_save:
            new_spacing = config.VOXEL_RESO

            ## Step 2.1 - Save resampled img
            t0 = time.time()
            voxel_img_sitk_resampled = resampler_sitk(voxel_img_sitk, spacing=new_spacing, interpolator=sitk.sitkBSpline) 
            t1 = time.time()
            voxel_img_sitk_resampled = sitk.Cast(voxel_img_sitk_resampled, sitk.sitkInt16)
            path_voxel_img_resampled = Path(voxel_save_folder).joinpath(config.FILENAME_IMG_RESAMPLED_3D)
            sitk.WriteImage(voxel_img_sitk_resampled, str(path_voxel_img_resampled))
            print (' - img resampling: ', t1 - t0)

            ## Step 2.2 - Save resampled mask
            if 0:
                import skimage
                import skimage.transform
                new_size = voxel_img_sitk_resampled.GetSize()
                voxel_mask_resampled = np.zeros(new_size)
                for label_id in np.unique(voxel_mask):
                    if label_id != 0:
                        voxel_mask_singlelabel = copy.deepcopy(voxel_mask).astype(config.DATATYPE_VOXEL_MASK)
                        voxel_mask_singlelabel[voxel_mask_singlelabel != label_id] = 0
                        voxel_mask_singlelabel[voxel_mask_singlelabel == label_id] = 1
                        voxel_mask_singlelabel_rescaled = skimage.transform.resize(voxel_mask_singlelabel
                                    , output_shape=new_size, preserve_range=True)
                        voxel_mask_singlelabel_rescaled[voxel_mask_singlelabel_rescaled < 0.5] = 0
                        idxs = np.argwhere(voxel_mask_singlelabel_rescaled > 0)
                        voxel_mask_resampled[idxs[:,0], idxs[:,1], idxs[:,2]] = label_id
                voxel_mask_resampled = voxel_mask_resampled.astype(config.DATATYPE_VOXEL_MASK)
                voxel_mask_sitk_resampled = array_to_sitk(voxel_mask_resampled 
                                , origin=voxel_img_headers['origin'], spacing=new_spacing)
                    
            elif 1:
                t0 = time.time()
                voxel_mask_sitk_resampled = resampler_sitk(voxel_mask_sitk, spacing=new_spacing, interpolator=sitk.sitkNearestNeighbor) 
                voxel_mask_sitk_resampled = sitk.Cast(voxel_mask_sitk_resampled, sitk.sitkUInt8)
                t1 = time.time()
                print (' - mask resampling: ', t1 - t0)
            
            elif 0:
                t0 = time.time()
                voxel_mask_sitk_resampled = resampler_sitk(voxel_mask_sitk, spacing=new_spacing, interpolator=sitk.sitkLinear) 
                voxel_mask_sitk_resampled = sitk.Cast(voxel_mask_sitk_resampled, sitk.sitkUInt8)
                t1 = time.time()
                print (' - mask resampling: ', t1 - t0)
            
            elif 0:
                new_size = voxel_img_sitk_resampled.GetSize()
                voxel_mask_resampled = np.zeros(new_size)
                for label_id in np.unique(voxel_mask):
                    if label_id != 0:
                        voxel_mask_singlelabel = copy.deepcopy(voxel_mask).astype(config.DATATYPE_VOXEL_MASK)
                        voxel_mask_singlelabel[voxel_mask_singlelabel != label_id] = 0
                        voxel_mask_singlelabel[voxel_mask_singlelabel == label_id] = 1
                        voxel_mask_singlelabel_sitk = array_to_sitk(voxel_mask_singlelabel
                            , origin=voxel_img_headers[config.KEYNAME_ORIGIN], spacing=voxel_img_headers[config.KEYNAME_PIXEL_SPACING])
                        voxel_mask_singlelabel_sitk_resampled = resampler_sitk(voxel_mask_singlelabel_sitk, spacing=new_spacing
                                    , interpolator=sitk.sitkLinear) 
                        voxel_mask_singlelabel_sitk_resampled = sitk.Cast(voxel_mask_singlelabel_sitk_resampled, sitk.sitkUInt8)
                        voxel_mask_singlelabel_array_resampled = sitk_to_array(voxel_mask_singlelabel_sitk_resampled)
                        idxs = np.argwhere(voxel_mask_singlelabel_array_resampled > 0)
                        voxel_mask_resampled[idxs[:,0], idxs[:,1], idxs[:,2]] = label_id

                voxel_mask_sitk_resampled = array_to_sitk(voxel_mask_resampled 
                                , origin=voxel_img_headers['origin'], spacing=new_spacing)
                voxel_mask_sitk_resampled = sitk.Cast(voxel_mask_sitk_resampled, sitk.sitkUInt8)

            path_voxel_img_resampled = Path(voxel_save_folder).joinpath(config.FILENAME_MASK_RESAMPLED_3D)
            sitk.WriteImage(voxel_mask_sitk_resampled, str(path_voxel_img_resampled))

        return path_voxel_img, path_voxel_mask
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def main():
    if 0:
        get_image_centre('', show=True)
    elif 1:
        path_data = Path('D:\\HCAI\\Project1-AutoSeg\\Code\\competitions\\medical_dataloader\\data\\HaN_HollandPTC\\processed\\data_3D\\Wilhelmina_Cherilyn_____Patient_ID.1_Study_ID.127\\Accession Nu.0')
        path_img = Path(path_data).joinpath('img.mha')
        path_mask = Path(path_data).joinpath('mask.mha')

        img_sitk = read_mha(path_img)
        img_array = sitk_to_array(img_sitk)
        img_array_headers = {config.KEYNAME_ORIGIN: img_sitk.GetOrigin(), config.KEYNAME_PIXEL_SPACING: img_sitk.GetSpacing()}
        mask_sitk = read_mha(path_mask)
        mask_array = sitk_to_array(mask_sitk)

        # save_path = Path(path_data, 'tmp')

        save_as_mha_wip(path_data, 'tmp', img_array, img_array_headers, mask_array, resample_save=True)    