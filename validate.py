"""To use this validation file, you must setup a .npy dataset file and a
.json validation description file. You must then set the paths to those files below.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from utils.data_loader import SingleFileDataset

from utils.models import *
# (
#     FinalModel, 
#     FinalModelLineMask,
#     TransformerImgNet,
#     TransformerFreqNet,
#     UnetFull,
#     UnetImg,
#     UnetFreq,
# )

from enum import Enum
import numpy as np
import tqdm
import json
import sys

def compute_psnr(img1, img2):
    mse = torch.mean(torch.square(img1 - img2))
    return 20 * torch.log10(255 / torch.sqrt(mse))

def validate(model, loader, dataset_type='validation'):
    """Plot a formatted visualization of the model results."""
    
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    ssim_loss_fn = ssim
    
    mse = []
    mae = []
    ssim_score = []
    psnr_score = []
    
    with torch.no_grad():
        model.eval()
        for x in tqdm.tqdm(loader):
            original_image = x[0:1, 2:3] * 255
            model_image = model(x.to(device)).to("cpu") * 255
    
            mse.append(mse_loss_fn(model_image, original_image).item())
            mae.append(mae_loss_fn(model_image, original_image).item())
            ssim_score.append(ssim_loss_fn(model_image, original_image))
            psnr_score.append(compute_psnr(model_image, original_image))
            
    mse = np.array(mse)
    mae = np.array(mae)
    ssim_score = np.array(ssim_score)
    psnr_score = np.array(psnr_score)
    
    return mse, mae, ssim_score, psnr_score

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Read in model spec json
    # NOTE: uncomment the line below and point it to the appropriate file
    # results_path = "path/to/your/model_description_file.json"
    with open(results_path, "r") as file:
        model_json = json.load(file)

    # NOTE: set path to test dataset by uncommenting line below
    # data_path = "path/to/your/dataset.npy"
    batch_size = 1
    dataset = SingleFileDataset(gt_path=data_path)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Loop through unfinished validation models
    for model_specs in model_json:
        print(f"\n--- {model_specs['name']} ---")
        if model_specs["results"] is not None:
            print("Results already generated.")
            continue

        model_str = model_specs["model_str"]  
        sparsity = float(model_specs["sparsity"])
        model = getattr(sys.modules[__name__], model_specs["model_class"])(sparsity=sparsity)
        model.load_state_dict(torch.load(model_str)["model_sd"])
        model.to(device)

        mse, mae, ssim_score, psnr = validate(model, loader)
        results_dict = {
            "mse_avg": float(mse.mean()),
            "mae_avg": float(mae.mean()),
            "ssim_avg": float(ssim_score.mean()),
            "psnr_avg": float(psnr.mean()),
            "mse_std": float(mse.std()),
            "mae_std": float(mae.std()),
            "ssim_std": float(ssim_score.std()),
            "psnr_std": float(psnr.std()),
            "mse_95th": float(sorted(mse)[int(len(loader)*0.95)]),
            "mae_95th": float(sorted(mae)[int(len(loader)*0.95)]),
            "ssim_5th": float(sorted(ssim_score)[int(len(loader)*0.05)]),
            "psnr_5th": float(sorted(psnr)[int(len(loader)*0.05)])
        }
        model_specs["results"] = results_dict
        model.to("cpu")
        del model

    with open("utils/results_out.json", "w") as file:
        json.dump(model_json, file)


