import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from utils.data_loader import SingleFileDataset
from utils.models import FinalModel

import time
import datetime
import numpy as np

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_loader(path, batch_size, shuffle):
    dataset = SingleFileDataset(gt_path=path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
    return loader


# Training Loader
training_set_path = "data/training.npy"
batch_size = 16
train_loader = get_loader(training_set_path, batch_size, shuffle=True)

# Validation Loader
validation_set_path = "data/validation.npy"
val_loader = get_loader(validation_set_path, batch_size=1, shuffle=False)


## Training

# For plotting training graphs
training_description = {
    "train_loss": [],
    "val_loss": [],
    "time": []
}

model = FinalModel()

# Set slope annealing starting point
model.undersampler.threshold_random_mask.slope = nn.parameter.Parameter(torch.tensor(50, requires_grad=False, dtype=torch.float32))
model.to(device)

epochs = 500
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


best_loss = np.inf
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_checkpoint_location = f'Models/model-name-{date}'  # NOTE: Fill in appropriate model name
last_saved_epoch = 0
print(f"Model checkpoints being saved at: {model_checkpoint_location}")

run_validation = True
val_loss = 0
val_loop_count = 1

for epoch in range(epochs):
    # Reset loss counters
    total_loss = 0
    train_loop_count = 0
    
    
    model.train()
    start_time = time.time()
    loader_start = time.time()
    for data in train_loader:
        data = data.to(device)
        loader_end = time.time() - loader_start
        # print(f"loader: {loader_end:3.5f}")  # For profiling loaders
        iteration_start = time.time()
        
        optimizer.zero_grad() 
        output = model(data, writer)
        loss = loss_fn(output, data[:, 2:3]) # indexing range so it maintains dimensionality
        loss.backward()
        optimizer.step()
        
        # print(f"epoch: {time.time() - iteration_start:3.5f}")  # For profiling training forward pass time
        
        total_loss += loss.item()
        train_loop_count += 1
        
        # NOTE: Uncomment to increase frequncy of validation checks. 
        # if train_loop_count >= 10000:
        #     break
            
        loader_start = time.time()
    end_time = time.time()
        
    if run_validation:
        # Run validation loop and checkpoint model
        val_loss = 0
        val_loop_count = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = loss_fn(output, data[0:1, 2:3])
            
                val_loss += loss.item()
                val_loop_count += 1

        # Save model weights if they have improved
        if best_loss > val_loss:
            print(f"Checkpoint saved at epoch {epoch+1} with slope {model.undersampler.threshold_random_mask.slope:.2f}")
            torch.save({
                'epoch': epoch,
                'model_sd': model.state_dict(),
                'opt_sd': optimizer.state_dict(),
                'loss': val_loss
            }, model_checkpoint_location)
            best_loss = val_loss
            last_saved_epoch = epoch
        
        # Decrease learning rate if val error stops improving
        if last_saved_epoch < epoch - 10:
            lr *= 0.75
            last_saved_epoch += 5
            print(f"Learning rate changed: {lr:.6f}")
            

        print(f"{epoch+1}: training mse: {total_loss/train_loop_count:4.5f}; validation mse: {val_loss/val_loop_count:4.5f}; training time: {end_time - start_time:3.2f}")
        training_description["train_loss"].append(total_loss/train_loop_count)
        training_description["val_loss"].append(val_loss/val_loop_count)
        training_description["time"].append(end_time-start_time)
    
    # Slowly decrease data leakage in undersampler through slope annealing
    if model.undersampler.threshold_random_mask.slope >= 1000:
        model.undersampler.threshold_random_mask.slope = nn.parameter.Parameter(torch.tensor(1000, requires_grad=False, dtype=torch.float32))
    else:
        slope = int(model.undersampler.threshold_random_mask.slope)
        model.undersampler.threshold_random_mask.slope = nn.parameter.Parameter(torch.tensor(slope + 50, requires_grad=False, dtype=torch.float32))

