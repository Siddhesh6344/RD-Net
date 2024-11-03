import os  
import torch  
from torch.utils.data import Dataset  
import nibabel as nib  
import nrrd  



class MedicalImageDataset(Dataset):
    def __init__(self, volume_files, segmentation_files, volume_folder, segmentation_folder, transform=None):
        self.volume_files = volume_files
        self.segmentation_files = segmentation_files
        self.volume_folder = volume_folder
        self.segmentation_folder = segmentation_folder
        self.transform = transform

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        volume_path = os.path.join(self.volume_folder, self.volume_files[idx])
        segmentation_path = os.path.join(self.segmentation_folder, self.segmentation_files[idx])
        
        # Load volume file
        volume_img = nib.load(volume_path).get_fdata()
        volume_img = torch.tensor(volume_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Load segmentation file
        segmentation_data, _ = nrrd.read(segmentation_path)  # Read NRRD file
        segmentation_data = torch.tensor(segmentation_data, dtype=torch.long)  # Convert to PyTorch tensor
        
        # Find unique values (classes) in the segmentation data
        unique_classes = torch.unique(segmentation_data)
        num_classes = unique_classes.numel()
        
        # Create a one-hot encoded segmentation tensor
        D, H, W = segmentation_data.shape
        one_hot_segmentation = torch.zeros(num_classes, D, H, W, dtype=torch.float32)
        for i, class_val in enumerate(unique_classes):
            one_hot_segmentation[i] = (segmentation_data == class_val).float()
        
        if self.transform:
            volume_img = self.transform(volume_img)
        
        # Return volume image and one-hot encoded segmentation tensor
        return volume_img, one_hot_segmentation