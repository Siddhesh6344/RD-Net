import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP




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
        self.checkpoint_dir = 'scratch/xc17/ss5114/BTP/G/pdr_revised2'
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

        average_dice = torch.mean(dice)
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


