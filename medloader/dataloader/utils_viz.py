import pdb
import copy
import traceback
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import medloader.dataloader.utils as utils
import medloader.dataloader.config as config

def cmap_for_dataset(dataset):
    LABEL_COLORS = getattr(config, dataset.name)['LABEL_COLORS']
    cmap_me = matplotlib.colors.ListedColormap(np.array([*LABEL_COLORS.values()])/255.0)
    norm = matplotlib.colors.BoundaryNorm(boundaries=range(0,cmap_me.N+1), ncolors=cmap_me.N)

    return cmap_me, norm

def get_info_from_label_id(label_id, dataset):
    """
    The label_id param has to be greater than 0
    """
    LABEL_MAP = getattr(config, dataset.name)['LABEL_MAP']
    LABEL_COLORS = getattr(config, dataset.name)['LABEL_COLORS']

    label_name = [label for label in LABEL_MAP if LABEL_MAP[label] == label_id]
    if len(label_name):
        label_name = label_name[0]
    else:
        label_name = None

    label_color = np.array(LABEL_COLORS[label_id])
    if np.any(label_color > 1):
        label_color = label_color/255.0

    return label_name, label_color

############################################################
#                             3D                           #
############################################################

def viz_3d_slices(voxel_img, voxel_mask, dataset, meta1, meta2, plots=4):
    """
    voxel_img : [B,H,W,D,C=1]
    voxel_mask: [B,H,W,D,C=1]
    """
    try:
        cmap_me, norm = cmap_for_dataset(dataset)

        for batch_id in range(voxel_img.shape[0]):
            voxel_img_batch = voxel_img[batch_id,:,:,:,0]
            voxel_mask_batch = voxel_mask[batch_id,:,:,:,0]
            height = voxel_img_batch.shape[-1]

            f,axarr = plt.subplots(2,plots)
            for plt_idx, z_idx in enumerate(np.random.choice(height, plots, replace=False)):
                axarr[0][plt_idx].imshow(voxel_img_batch[:,:,z_idx], cmap='gray')
                axarr[1][plt_idx].imshow(voxel_img_batch[:,:,z_idx], cmap='gray', alpha=0.4)
                axarr[1][plt_idx].imshow(voxel_mask_batch[:,:,z_idx], cmap=cmap_me, norm=norm, interpolation='none')
                axarr[0][plt_idx].set_title('Slice: {}/{}'.format(z_idx+1, height))
            
            name, patient_id, study_id = utils.get_name_patient_study_id(meta2[batch_id])
            if study_id is not None:
                filename = patient_id + '\n' + study_id
            else:
                filename = patient_id
            plt.suptitle(dataset.name + '\n' + filename)
            plt.show()

                
    except:
        traceback.print_exc()
        pdb.set_trace()

def viz_3d_mask(voxel_masks, dataset, meta1, meta2):
    """
    Expects a [B,H,W,D] shaped mask
    """
    try:
        import plotly.graph_objects as go
        import skimage
        import skimage.measure
        
        # Step 1 - Loop over all batch_ids
        for batch_id, voxel_mask in enumerate(voxel_masks):
            fig = go.Figure()
            label_ids = np.unique(voxel_mask)
            print (' - [utils_viz.viz_3d_mask()] label_ids: ', label_ids)
            if len(label_ids) == 1:
                if int(label_ids[0]) == 0:
                    continue
            
            # [REMOVE THIS]
            if len(label_ids) == 3:
                continue

            voxel_mask_ = voxel_mask
            
            # Step 2 - Loop over all label_ids
            for i_, label_id in enumerate(label_ids):
                
                if label_id == 0 : continue
                name, color = get_info_from_label_id(label_id, dataset[batch_id])
                print (' - label_id: ', label_id, '(',name,')')

                # Get surface information
                voxel_mask_tmp = np.array(copy.deepcopy(voxel_mask_)).astype(config.DATATYPE_VOXEL_MASK)
                voxel_mask_tmp[voxel_mask_tmp != label_id] = 0
                verts, faces, _, _ = skimage.measure.marching_cubes(voxel_mask_tmp, step_size=1)
                
                # https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Mesh3d.html
                visible=True
                fig.add_trace(
                    go.Mesh3d(
                        x=verts[:,0], y=verts[:,1], z=verts[:,2]
                        , i=faces[:,0],j=faces[:,1],k=faces[:,2]
                        , color='rgb({},{},{})'.format(*color)
                        , name=name, showlegend=True
                        , visible=visible
                        # , lighting=go.mesh3d.Lighting(ambient=0.5)
                    )
                )
            
            fig.update_layout(
                scene = dict(
                    xaxis = dict(nticks=10, range=[0,voxel_mask.shape[0]], title='X-axis'),
                    yaxis = dict(nticks=10, range=[0,voxel_mask.shape[1]]),
                    zaxis = dict(nticks=10, range=[0,voxel_mask.shape[2]]),
                )
                ,width=700,
                margin=dict(r=20, l=50, b=10, t=50)
            )
            fig.update_layout(legend_title_text='Labels', showlegend=True)
            fig.update_layout(scene_aspectmode='cube') # [data, cube]
            fig.update_layout(title_text='{} (BatchId={})'.format(meta2, batch_id))
            fig.show()

            pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()
