import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import nibabel as nib
import nrrd
import SimpleITK as sitk
import numpy as np
import tempfile
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import time

volume_folder = 'scratch/xc17/ss5114/BTP/Resize_vol'
segmentation_folder = 'scratch/xc17/ss5114/BTP/Without_sacrum'
scaler = GradScaler()


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

    
def load_data_splits(file_path='data_splits.npz'):
    data = np.load(file_path)
    train_volumes = data['train_volumes']
    test_volumes = data['test_volumes']
    train_segmentations = data['train_segmentations']
    test_segmentations = data['test_segmentations']
    
    print(f"Data splits loaded from {file_path}")
    
    return train_volumes, test_volumes, train_segmentations, test_segmentations   

    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv_block(x)
        p = self.pool(x1)
        return x1, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bridge = ConvBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder4 = DecoderBlock(128, 64, 64)

        self.final_conv = nn.Conv3d(64, n_classes, kernel_size=1)

        self.activation = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b1 = self.bridge(p4)

        d1 = self.decoder1(b1, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        out = self.final_conv(d4)
        return self.activation(out)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,  # Added validation data
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data  # Validation data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.checkpoint_dir = 'scratch/xc17/ss5114/BTP/S/checkpoints_3dunet'
        self.scaler = GradScaler()
        self.losses = []  # Store training losses
        self.val_losses = []  # Store validation losses
        self.train_dice_array = []  # Store training dice array
        self.val_dice_array = []  # Store validation dice array

    def dice_coefficient(self, y_true, y_pred, smoothing=1):
        eps = smoothing
        device = y_pred.device  # Ensure both tensors are on the same device
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)

        # Number of classes (channels), assuming shape is [batch, channels, depth, height, width]
        n_outputs = y_pred.shape[1]  # In this case, it will be 6 (including the background)

        dice = torch.zeros(n_outputs, device=device)  # For storing dice for each class (excluding background)

        # Loop through each class (skip background class 0)
        for c in range(0, n_outputs):
            pred_class = y_pred[:, c, :, :, :].reshape(-1)  # Flatten the prediction for class c
            true_class = y_true[:, c, :, :, :].reshape(-1)  # Flatten the ground truth for class c

            intersection = torch.sum(pred_class * true_class)
            union = torch.sum(pred_class) + torch.sum(true_class)

            d = (2.0 * intersection + eps) / (union + eps)

            dice[c] = d  # Store the dice score for class c

        average_dice = torch.mean(dice)  # Average dice score (excluding background)
        return average_dice,dice

    def dice_coefficient_loss(self, y_true, y_pred):
        dice_scores,dice_array = self.dice_coefficient(y_true, y_pred)
        return 1 - dice_scores,dice_array  # Mean Dice loss over all classes

    def _run_batch(self, source, targets):
        source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
        self.optimizer.zero_grad()
        with autocast():
            output = self.model(source)
            loss,dice_array = self.dice_coefficient_loss(output, targets)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), dice_array  # Return both loss and dice scores

    def _validate_batch(self, source, targets):
        with torch.no_grad():
            with autocast():
                output = self.model(source)
                loss,dice_array = self.dice_coefficient_loss(output, targets)
               
        return loss.item(), dice_array

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        test_dice_array = torch.zeros(5, device=self.gpu_id) 
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,dice_array = self._run_batch(source, targets)
            epoch_loss += loss
            test_dice_array+=dice_array
            
            torch.cuda.empty_cache()

        avg_train_loss = epoch_loss / len(self.train_data)
        test_dice_array = test_dice_array / len(self.train_data) 
        self.losses.append(avg_train_loss)
        self.train_dice_array.append(test_dice_array.cpu().tolist())
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average Training Loss: {avg_train_loss:.4f}")
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average Test Dice Array: {test_dice_array.tolist()}")

        # Validation phase
        val_loss,val_dice_array = self._run_validation()
        self.val_losses.append(val_loss)
        self.val_dice_array.append(val_dice_array.cpu().tolist())
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average Validation Loss: {val_loss:.4f}")
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average Val Dice Array: {val_dice_array.tolist()}")
        
        return avg_train_loss, val_loss,test_dice_array,val_dice_array

    def _run_validation(self):
        epoch_val_loss = 0.0
        val_dice_array = torch.zeros(5, device=self.gpu_id)
        for source, targets in self.val_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,dice_array = self._validate_batch(source, targets)
            epoch_val_loss += loss
            val_dice_array+=dice_array

        avg_val_loss = epoch_val_loss / len(self.val_data)
        val_dice_array = val_dice_array / len(self.val_data)
        return avg_val_loss,val_dice_array

    def _save_checkpoint(self, epoch, train_loss, val_loss,test_dice_array,val_dice_array):
        ckp = {
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_dice_array': test_dice_array,
            'val_dice_array': val_dice_array
        }
        PATH = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            train_loss, val_loss,test_dice_array,val_dice_array = self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_loss, val_loss,test_dice_array,val_dice_array)



def load_train_objs(train_volumes, train_segmentations, test_volumes, test_segmentations):
    train_set = MedicalImageDataset(train_volumes, train_segmentations, volume_folder, segmentation_folder)  
    val_set = MedicalImageDataset(test_volumes, test_segmentations, volume_folder, segmentation_folder)  # Validation data
    model = UNet(in_channels=1, n_classes=5)  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=4
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    start = time.time()
    
    train_volumes, test_volumes, train_segmentations, test_segmentations = load_data_splits(file_path='scratch/xc17/ss5114/BTP/S/sample.npz')

    
    if rank == 0:
        print(f"Starting training with {world_size} GPUs")
    ddp_setup(rank, world_size)
    train_set, val_set, model, optimizer = load_train_objs(train_volumes, train_segmentations, test_volumes, test_segmentations)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size)  # Validation data
    trainer = Trainer(model, train_data, val_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

    # Plot loss vs epochs on rank 0
    if rank == 0:
        plt.plot(range(total_epochs), trainer.losses, marker='o', label='Train Loss')
        plt.plot(range(total_epochs), trainer.val_losses, marker='o', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training and Validation Loss vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    endd = time.time()
    print((endd-start)/60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size),
    
    

    
    
  