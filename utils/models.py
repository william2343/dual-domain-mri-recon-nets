import torch
import torchvision
import torch.nn as nn

from utils.UndersamplerBatched import UndersamplerB
from utils.transformer import UFormer as UFormerFreq
from utils.Transformer_img import UFormer as UFormerImage
from utils.transformer_Combo import UFormer as UFormerCombo
from utils.img_unet import UNet as UNetImage
from utils.freq_unet import UNet as UNetFrequency
from utils.combo_unet import UNet as UNetCombination
from utils.transformer import model_out_to_img

class FinalModel(nn.Module):
    def __init__(self, sparsity=1/20, bsn=False):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity), bsn=bsn, pmask_shape=(128, 128, 5, 1)) # with appropriate parameters for undersampling network
        self.image_transformer = UFormerImage(in_channels=10) 
        self.frequency_transformer = UFormerFreq(in_channels=10)
        self.combiner = UFormerCombo(in_channels=2)
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.batch_norm2 = nn.BatchNorm2d(1)
        
    def forward(self, x):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        undersampled_freq, undersampled_img, _ = self.undersampler(x)

        reconstructed_img = self.image_transformer(undersampled_img)
        reconstructed_img = self.batch_norm1(reconstructed_img)
        reconstructed_freq = model_out_to_img(self.frequency_transformer(undersampled_freq)).unsqueeze(dim=1) # this
        reconstructed_freq = self.batch_norm2(reconstructed_freq)    

        final_image = self.combiner(torch.cat([reconstructed_img, reconstructed_freq], axis=1))    
        return final_image

class FinalModelLineMask(nn.Module):
    def __init__(self, sparsity=1/20, bsn=False):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity), bsn=bsn, pmask_shape=(128, 5, 1)) # with appropriate parameters for undersampling network
        self.image_transformer = UFormerImage(in_channels=10) 
        self.frequency_transformer = UFormerFreq(in_channels=10)
        self.combiner = UFormerCombo(in_channels=2)
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.batch_norm2 = nn.BatchNorm2d(1)
    
    def forward(self, x):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        undersampled_freq, undersampled_img, _ = self.undersampler(x)

        reconstructed_img = self.image_transformer(undersampled_img)
        reconstructed_img = self.batch_norm1(reconstructed_img)
        reconstructed_freq = model_out_to_img(self.frequency_transformer(undersampled_freq)).unsqueeze(dim=1) # this
        reconstructed_freq = self.batch_norm2(reconstructed_freq)    

        final_image = self.combiner(torch.cat([reconstructed_img, reconstructed_freq], axis=1))    
        return final_image

class TransformerImgNet(nn.Module):
    def __init__(self, sparsity=1/20):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity)) # with appropriate parameters for undersampling network
        self.image_transformer = UFormerImage(in_channels=10)         
    
    def forward(self, x):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        _, undersampled_img, _ = self.undersampler(x) # this

        reconstructed_img = self.image_transformer(undersampled_img) # this
        return reconstructed_img

class TransformerFreqNet(nn.Module):
    def __init__(self, sparsity=1/20):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity)) # with appropriate parameters for undersampling network
        self.frequency_transformer = UFormerFreq(in_channels=10)        
    
    def forward(self, x):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        undersampled_freq, _, _ = self.undersampler(x)
        reconstructed_freq = model_out_to_img(self.frequency_transformer(undersampled_freq)).unsqueeze(dim=1)
        return reconstructed_freq

class UnetFull(nn.Module):
    def __init__(self, sparsity=1/20):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity)) # with appropriate parameters for undersampling network
        self.image_transformer = UNetImage(in_channels=10, out_channels=1, init_features=4)
        self.frequency_transformer = UNetFrequency(in_channels=10, out_channels=2, init_features=4)
        self.combiner = UNetCombination(in_channels=2, out_channels=1, init_features=4)
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.batch_norm2 = nn.BatchNorm2d(1)
        
    
    def forward(self, x):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        undersampled_freq, undersampled_img, _ = self.undersampler(x)

        reconstructed_img = self.image_transformer(undersampled_img) 
        reconstructed_img = self.batch_norm1(reconstructed_img)
        reconstructed_freq = model_out_to_img(self.frequency_transformer(undersampled_freq)).unsqueeze(dim=1)
        reconstructed_freq = self.batch_norm2(reconstructed_freq)    
        final_image = self.combiner(torch.cat([reconstructed_img, reconstructed_freq], axis=1))
    
        return final_image

class UnetImg(nn.Module):
    def __init__(self, sparsity=1/20):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity)) # with appropriate parameters for undersampling network
        self.image_transformer = UNetImage(in_channels=10, out_channels=1, init_features=4)
    
    def forward(self, x, writer=None, epoch=0):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        _, undersampled_img, _ = self.undersampler(x)
        final_image = self.image_transformer(undersampled_img) 
        return final_image

class UnetFreq(nn.Module):
    def __init__(self, sparsity=1/20):
        super().__init__()
        self.undersampler = UndersamplerB(sparsity=(sparsity)) # with appropriate parameters for undersampling network
        self.frequency_transformer = UNetFrequency(in_channels=10, out_channels=2, init_features=4)
    
    def forward(self, x, writer=None, epoch=0):
        """This should take a sequence of full resolution MRI images,
        perform parameterized undersampling, then reconstruction in image + freq domain, then combine,
        and finally output reconstructed image.
        
        Input is 1x5x128x128 (old:128x128x5x1) into FinalModel, produces 1x1x128x128.
        """
        
        undersampled_freq, _, _ = self.undersampler(x)
        reconstructed_freq = model_out_to_img(self.frequency_transformer(undersampled_freq)).unsqueeze(dim=1)
    
        return reconstructed_freq