################################################### transformations

def scale_array_to_range(arr, curr_range_min, curr_range_max, new_range_min, new_range_max):
    '''
    scales array arr from range [curr_range_min, curr_range_max] to range [new_range_min, new_range_max]
    '''
    arr_0_1 = (arr - curr_range_min) / (curr_range_max - curr_range_min)
    arr_range = new_range_min + arr_0_1 * (new_range_max - new_range_min)
    return arr_range


def set_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


################################################### ploting
    
def plot_imgs_and_masks(img_list, true_mask_list, pred_mask_list = None):
    '''
    plots images, true and predicted masks. img_list shoud containt images from range [0, 255], true_/predicted_mask_list masks from range {0, 1, ..., 7}
    '''
    import matplotlib.pyplot as plt
    
    cols_count = 2 if pred_mask_list is None else 3
    fig, axes = plt.subplots(len(img_list), cols_count, figsize=(3 * cols_count, 3 * len(img_list)+1))

    for i in range(len(img_list)):
        if len(img_list) == 1:
            ax_img = axes[0]
            ax_true_mask = axes[1]
            if pred_mask_list is not None:
                ax_pred_mask = axes[2]
        else:
            ax_img = axes[i, 0]
            ax_true_mask = axes[i, 1]
            if pred_mask_list is not None:
                ax_pred_mask = axes[i, 2]

        ax_img.imshow(img_list[i], cmap='gray')
        ax_img.set_title('oryg. img')
        ax_true_mask.imshow(true_mask_list[i], cmap='jet')
        ax_true_mask.set_title('true mask')
        if pred_mask_list is not None:
            ax_pred_mask.imshow(pred_mask_list[i], cmap ='jet')
            ax_pred_mask.set_title('pred. mask')

    plt.show()



def plot_epoch_trainval_loss(base_path, model_name, axis):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    path = f'{base_path}/trained_models/model_losses/{model_name}_{axis}_lossepoch_train_val.csv'
    df = pd.read_csv(path)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(df.epoch, df.val_loss)
    ax.plot(df.epoch, df.train_loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss value')
    ax.legend(['val set','train set'])

    plt.show()


################################################### saving and reading

def save_arr_as_img(img_data, curr_img_range_min, curr_img_range_max, dir_path, name):
    '''
    saves img_data array as .png image, scaling it before to [0, 255] range
    '''
    import cv2
    import numpy as np
    import os
    img_data = np.uint8(scale_array_to_range(img_data, curr_img_range_min, curr_img_range_max, 0, 255))
    fout = os.path.join(dir_path, f'{name}.png')
    cv2.imwrite(fout, img_data)

def extract_ids(img_slice_fname, slice_type = 'img'):
    if slice_type not in ['img', 'pred', 'seg']:
        raise Exception("slice_type must be in range ['img', 'pred', 'seg']")
    import re
    type_suffix = '' if slice_type == 'img' else slice_type
    pattern = 'sub-feta([0-9]{3})'+f'{type_suffix}'+'-slice([0-9]{3})_([xyz]).png'
    groups = re.match(pattern, img_slice_fname).groups()
    return groups[0], groups[1], groups[2].upper()

def get_img_slice_fname(img_idx: int, slice_idx: int, axis: str, slice_type = 'img', slice_decimate_identifier = 3):
    if slice_type not in ['img', 'pred', 'seg']:
        raise Exception("slice_type must be in range ['img', 'pred', 'seg']")
    type_suffix = '' if slice_type == 'img' else slice_type
    return f'sub-feta{str(img_idx).zfill(slice_decimate_identifier)}{type_suffix}-slice{str(slice_idx).zfill(slice_decimate_identifier)}_{axis.lower()}.png'

def get_img_slice_dir_path(base_path: str, img_idx: int, axis: str, slice_type = 'img', slice_decimate_identifier = 3):
    import os
    if slice_type not in ['img', 'pred', 'seg']:
        raise Exception("slice_type must be in range ['img', 'pred', 'seg']")
    if axis not in ['X', 'Y', 'Z']:
        raise Exception("axis must me in range  ['X', 'Y', 'Z']")
    type_suffix = '' if slice_type == 'img' else slice_type

    volume_dir = f'sub-feta{str(img_idx).zfill(slice_decimate_identifier)}{type_suffix}'
    return os.path.join(base_path, volume_dir, axis)

def create_or_overwrite_dir(path):
    import os 
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def create_dir_if_not_exists(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)

def get_models_params(base_path):
    import pandas as pd
    import json
    import os
    model_params_dirpath = os.path.join(base_path, 'trained_models', 'model_params')
    model_params_files = os.listdir(model_params_dirpath)
    model_params_paths = [os.path.join(model_params_dirpath, model_params_file) for model_params_file in model_params_files ]
    params_rows = []
    for model_params_path in model_params_paths:
        with open(model_params_path) as json_file:
            params_rows.append(json.load(json_file))
    return pd.DataFrame(params_rows)\
        .replace('DiceCrossEntropyLoss(\n  (wass_loss): GeneralizedWassersteinDiceLoss()\n  (ce_loss): CrossEntropyLoss()\n)', 'DiceCrossEntropyLoss')\
        .replace('CrossEntropyLoss()', 'CrossEntropyLoss')

        
