from dataset import BrainTissueDataset
from torch.utils.data import DataLoader

def load_BrainTissue_data(data_dir_path, img_idx_start, img_idx_stop, axis, transforms, batch_size, shuffle=True, is_eval = False):
    data = BrainTissueDataset(data_dir_path, img_idx_start, img_idx_stop, axis = axis, transform=transforms, is_eval=is_eval)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader