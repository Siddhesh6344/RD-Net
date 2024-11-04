import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import time
import torch.optim as optim
from model import RDNet
from Dataloader import CTPEL
from Trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import os

volume_folder = 'scratch/param/ss5114/BTP/vol_files'
segmentation_folder = 'scratch/param/ss5114/BTP/seg_mask'

batch_size = 1
learning_rate = 1e-4
num_epochs = 75 # We have trained till 120 epochs to see the model convergence, Model is getting converged after 75 epochs and results are obtained on val_set on 75 epochs
save_every = 5 
world_size = torch.cuda.device_count()  # 4 GPUs we have used NVIDIA V100 with VRAM 32GB

def load_data_splits(file_path='data_splits.npz'):  #data_splits as 80:20 for train and val set
    data = np.load(file_path)
    train_volumes = data['train_volumes']
    test_volumes = data['test_volumes']
    train_segmentations = data['train_segmentations']
    test_segmentations = data['test_segmentations']
    
    print(f"Data splits loaded from {file_path}")
    
    return train_volumes, test_volumes, train_segmentations, test_segmentations   

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

def load_train_objs(train_volumes, train_segmentations, test_volumes, test_segmentations):
    train_set = CTPEL(train_volumes, train_segmentations, volume_folder, segmentation_folder)  
    val_set = CTPEL(test_volumes, test_segmentations, volume_folder, segmentation_folder)  # Validation data
    model = RDNet(in_channels=1, out_channels=5)  
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #adaptive learning rate can be used by Gridsearch or weight moment adpative allocationg

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
    start_time = time.time()
    train_volumes, test_volumes, train_segmentations, test_segmentations = load_data_splits(file_path='scratch/param/ss5114/BTP/S/split.npz')

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

    end_time = time.time()
    print((end_time-start_time)/60) #Total of training and inference time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size),
