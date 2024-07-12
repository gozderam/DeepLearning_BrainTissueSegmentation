import torch
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from utils_functions import save_arr_as_img, extract_ids, create_dir_if_not_exists, get_img_slice_fname, get_img_slice_dir_path, plot_imgs_and_masks, extract_ids, scale_array_to_range, create_or_overwrite_dir
import json
import numpy as np
import configs
from loader import load_BrainTissue_data
import nibabel as nib
import subprocess
import re
import pandas as pd


def save_predictions(preds_path, model_name, data, model, device):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            X, y, file_names = batch
            X, y = X.to(device), y.to(device)
            predictions = model(X)

            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1)
            pred_labels = pred_labels.float()

            # Resizing predicted images too original size
            pred_labels = transforms.Resize((configs.IMG_ORYG_HEIGHT, configs.IMG_ORYG_WIDTH))(pred_labels)

            # save pred masks
            __save_pred_labels(preds_path, model_name, pred_labels, file_names)

def get_trained_model_params(model_path):
    with open(model_path) as json_file:
        params_dict = json.load(json_file)
    return params_dict

def visualize_results(data_path, preds_path, model_name, img_idxs, slice_idxs, axes):
    '''
    data_path - base data path
    preds_path - predictions path
    model_name - name of the model (without axis sufix)
    img_idxs - indexes of images to show
    slice_idxs - indexes of slices of corresponding images to show
    axes - axes of corresponding slices to show
    '''
    img_paths = [os.path.join(get_img_slice_dir_path(data_path, img_idx, axis, 'img'), get_img_slice_fname(img_idx, slice_idx, axis, 'img')) \
        for img_idx, slice_idx, axis in zip(img_idxs, slice_idxs, axes)]

    true_mask_paths = [os.path.join(get_img_slice_dir_path(data_path, img_idx, axis, 'seg'), get_img_slice_fname(img_idx, slice_idx, axis, 'seg')) \
            for img_idx, slice_idx, axis in zip(img_idxs, slice_idxs, axes)]

    pred_mask_paths = [os.path.join(get_img_slice_dir_path(os.path.join(preds_path, model_name), img_idx, axis, 'pred'), get_img_slice_fname(img_idx, slice_idx, axis, 'pred')) \
            for img_idx, slice_idx, axis in zip(img_idxs, slice_idxs, axes)]

    imgs = [Image.open(img_path) for img_path in img_paths]
    true_masks = [Image.open(true_mask_path) for true_mask_path in true_mask_paths]
    pred_masks = [Image.open(pred_mask_path) for pred_mask_path in pred_mask_paths]

    plot_imgs_and_masks(imgs, true_masks, pred_masks)


def __save_pred_labels(preds_path, model_name, pred_labels, file_names):
    '''
    preds_path - base prediction paths
    model_name - name of the model (without axis suffix)
    pred_labels - predicted masks
    file_names - names of files for which predictions were made 
    '''
    pred_labels_list = list(pred_labels)
  
    for (img_file_name, y_mask) in zip(file_names, pred_labels_list):
        img_num, slice_num, axis = extract_ids(img_file_name)

        img_pred_dir = f"sub-feta{img_num}pred"
        create_dir_if_not_exists(os.path.join(preds_path, model_name, img_pred_dir))

        axis_dir_path = os.path.join(preds_path, model_name, img_pred_dir, axis)
        create_dir_if_not_exists(axis_dir_path)

        slice_name = img_pred_dir + f"-slice{slice_num}_{axis.lower()}"
        save_arr_as_img(y_mask.numpy(), 0, 7, axis_dir_path, slice_name)


def reconstruct_3D_mask_volumne(img_2D_path, x_coef=0.4, y_coef=0.4, z_coef=0.2, img_size=configs.IMG_ORYG_HEIGHT):
    '''
    reconstructs 3D mask based on predicted slices
    '''
    axes = ['X', 'Y', 'Z']

    axis_3D_volumes = [0]*3

    for axis_idx, axis in enumerate(axes):
        axis_path = os.path.join(img_2D_path, axis)
        slice_names = os.listdir(axis_path)
        axis_slice_paths = [os.path.join(axis_path, slice_name) for slice_name in slice_names]

        axis_slice_idxs = [int(extract_ids(slice_name, slice_type='pred')[1]) for slice_name in slice_names] # valid slices
        missing_slices_idxs = list(set(range(img_size)).difference(set(axis_slice_idxs))) # indexes of missing slices, meaning they were all black (0 color) in the input image

        axis_all_slices = [0] * img_size # both empty and valid slices

        for i in missing_slices_idxs:
            axis_all_slices[i] = np.zeros((img_size, img_size)).astype('uint8')

        for enum, i in enumerate(axis_slice_idxs):
            slice = Image.open(axis_slice_paths[enum])  # pil image in range [0, 255]
            slice = np.round(scale_array_to_range(np.array(slice), 0, 255, 0, 7)).astype('uint8') # scale round and to uint8
            axis_all_slices[i] = slice

        axis_3D_volumes[axis_idx] = np.stack(axis_all_slices, axis=axis_idx) 
    
    reconstrcted_mask = (x_coef*axis_3D_volumes[0] + y_coef*axis_3D_volumes[1] + z_coef*axis_3D_volumes[2]).astype('int')
    return reconstrcted_mask

def reconstruct_3D(preds_path, model_name, img_idx_start, img_idx_stop, slice_decimate_identifier = 3):
    pred_names = [f'sub-feta{str(i).zfill(slice_decimate_identifier)}pred' for i in range(img_idx_start, img_idx_stop+1)]
    pred_masks_paths = [os.path.join(preds_path, model_name, pred_name) for pred_name in pred_names]

    for idx, pred_mask_path in enumerate(pred_masks_paths):
        reconstructed_volumes_dir = os.path.join(pred_mask_path, '3D')
        create_or_overwrite_dir(reconstructed_volumes_dir)

        res = reconstruct_3D_mask_volumne(pred_mask_path)
        #np.save(os.path.join(reconstructed_volumes_dir, f'{pred_names[idx]}_3Drec.npy'), res)
        nii_img = nib.Nifti1Image(res, affine=np.eye(4))
        nib.loadsave.save(nii_img, os.path.join(reconstructed_volumes_dir, f'{pred_names[idx]}_3Drec.nii'))


def evaluate_model(model_name, model_constructor, get_model_path_f, get_params_path_f, test_idx_start, test_idx_stop, device):
    axes = ['X', 'Y', 'Z']

    create_or_overwrite_dir(os.path.join(configs.PREDS_PATH, model_name))

    for axis in axes:
        # networks params
        net = model_constructor(in_channels=1, classes=8).to(device)
        if device == "cpu":
            checkpoint = torch.load(get_model_path_f(model_name, axis), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(get_model_path_f(model_name, axis))
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        network_params = get_trained_model_params(f'{get_params_path_f(model_name, axis)}.json')
        img_height = network_params['img_height']
        img_width = network_params['img_width']
        batch_size = network_params['batch_size']

        # data
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width), interpolation=transforms.InterpolationMode.NEAREST)
        ])
        test_loader = load_BrainTissue_data(configs.DATA_2D_PATH, test_idx_start, test_idx_stop, axis, transform, batch_size, is_eval=True)

        # predict and save 
        save_predictions(configs.PREDS_PATH, model_name, test_loader, net, device)

    reconstruct_3D(configs.PREDS_PATH, model_name, test_idx_start, test_idx_stop)


def ps_run(cmd):
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    return str(completed.stdout)

def get_dice_volsim_hausd(true_label_mask_nii_path, pred_label_mask_nii_path, evaluate_tool_path):
    run_ps_command = f"{evaluate_tool_path} {true_label_mask_nii_path} {pred_label_mask_nii_path} -use DICE,VOLSMTY,HDRFDST"
    stdout = ps_run(run_ps_command)
    if stdout=='Moving image is empty!':
        raise Exception('Moving image is empty!')
    dice_pattern = r'.*DICE\\t= ([0-9.]*)\\t.*VOLSMTY\\t= ([0-9.]*)\\t.*HDRFDST\\t= ([0-9.]*)\\t.*'
    match = re.match(dice_pattern, stdout)
    dice, volsim, hausd = float(match.groups()[0]), float(match.groups()[1]), float(match.groups()[2])
    return dice, volsim, hausd


def save_metrics(model_names, img_idx_start, img_idx_stop, base_path, preds_path, slice_decimate_identifier=3):

    results_rows = []
    img_names = [f"sub-feta{str(i).zfill(slice_decimate_identifier)}" for i in range(img_idx_start, img_idx_stop+1)]

    for model_name in model_names:

        img_names_for_metrics = []
        label_indexes = []
        dices = []
        volsims = []
        hausds = []

        pred_label_mask_nii_path = f'./pred_label_mask_{model_name}.nii'
        true_label_mask_nii_path = f'./true_label_mask_{model_name}.nii'
        evaluate_tool_path = '../EvaluateSegmentation.exe'

        for img_name in img_names:
            pred_full_mask_3D_path = os.path.join(preds_path, model_name, img_name + 'pred', '3D', f'{img_name}pred_3Drec.nii')
            pred_full_mask = img = nib.load(pred_full_mask_3D_path).get_fdata().astype('int')

            true_full_mask_3D_path = os.path.join(base_path, 'data', 'FeTA_1.2.1_BIDS', 'feta1.2.1', 'derivatives', img_name, 'anat', f'{img_name}_T2w-SR_dseg.nii.gz')
            true_full_mask = nib.load(true_full_mask_3D_path).get_fdata().astype('int')

            for i in range(1, 8):
                # for each ith label 
                pred_label_mask = pred_full_mask.copy()
                pred_label_mask[pred_label_mask!=i] = 0
                pred_label_mask[pred_label_mask==i] = 1

                true_label_mask = true_full_mask.copy()
                true_label_mask[true_label_mask!=i] = 0
                true_label_mask[true_label_mask==i] = 1

                pred_label_mask_nii = nib.Nifti1Image(pred_label_mask, affine=np.eye(4))
                true_label_mask_nii = nib.Nifti1Image(true_label_mask, affine=np.eye(4))

                # save images to run evalutate tool on
                nib.loadsave.save(pred_label_mask_nii, pred_label_mask_nii_path)
                nib.loadsave.save(true_label_mask_nii, true_label_mask_nii_path)

                try:
                    dice, volsim, hausd = get_dice_volsim_hausd(true_label_mask_nii_path, pred_label_mask_nii_path, evaluate_tool_path)
                except Exception:
                    continue
                
                img_names_for_metrics.append(img_name)
                label_indexes.append(i)
                dices.append(dice)
                volsims.append(volsim)
                hausds.append(hausd)
                

        os.remove(pred_label_mask_nii_path)
        os.remove(true_label_mask_nii_path)

        results_rows.append([model_name, np.mean(dices), np.std(dices), np.mean(volsims), np.std(volsims), np.mean(hausds), np.std(hausds)]) 

    df = pd.DataFrame(results_rows, columns = ['model_name', 'dice_mean', 'dice_std', 'volsim_mean', 'volsim_std', 'hausd_mean', 'hausd_std'])
    df.to_csv(os.path.join(preds_path, 'results.csv'))
    return df