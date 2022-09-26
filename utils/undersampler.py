import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class ConcatenateZero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # given input of shape 10x10x1, returns output of shape e.g. 10x10x2
        return torch.cat([tensor, tensor*0], -1)

def visualize_complex_image(two_channel_image):
    # the input is 2 ..x..x2
    complex_one_channel_image = torch.complex(two_channel_image[:, :, 0], two_channel_image[:, :, 1])
    plt.imshow(np.absolute(complex_one_channel_image.detach().numpy()), cmap='viridis')
    ax = plt.gca()
    ax.grid(False)
    plt.show()

# this FFT is working correctly
class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # if one image, input should be of shape ..x..x2, and output will be ..x..x2 k-space data
        # for five images, input would be of shape ..x..x10, and output will be ..x..x10
        tempcpx = torch.complex(tensor[..., 0], tensor[..., 1]) # this'll shrink dimensionality to ..x..x1 if one image input
        if len(list(tempcpx.shape)) == 4:
            tempcpx = torch.permute(tempcpx, (0, 3, 2, 1))
        
        fft_im = torch.fft.fft2(tempcpx)

        if len(list(tempcpx.shape)) == 4:
            fft_im = torch.permute(fft_im, (0, 3, 2, 1))
        
        fft_im = torch.stack([torch.real(fft_im), torch.imag(fft_im)], dim=-1)
        return fft_im

# this works correctly
class IFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        tempcpx = torch.complex(tensor[..., 0], tensor[..., 1])
        if len(list(tempcpx.shape)) == 4:
            tempcpx = torch.permute(tempcpx, (0, 3, 2, 1))
        
        ifft_im = torch.fft.ifft2(tempcpx)
        if len(list(tempcpx.shape)) == 4:
            ifft_im = torch.permute(ifft_im, (0, 3, 2, 1))
        ifft_im = torch.stack([torch.real(ifft_im), torch.imag(ifft_im)], dim=-1)
        return ifft_im


# knowing input_shape is (128, 128, 5, 1) see how it could change any of the implementation here.
# debugging RescaleProbMap currently, need to tweak/understand Tom's to make it run

class ProbMask(nn.Module):
    def __init__(self, slope=10, eps=0.01, mask_shape=(128, 128, 5, 1)):
        super().__init__()
        # h = torch.randn(size=(2,5), requires_grad=True, dtype=torch.float32)
        self.slope = nn.parameter.Parameter(torch.tensor(slope, requires_grad=False, dtype=torch.float32)) 
        # create a 128x128x1 layer of weights, initialized using _logit_slope_random_uniform method
        # i.e. sampled from uniform distribution, has 128x128x1 shape, between [minval, 1-minval]
        
        minval = eps
        maxval = 1.0-eps
        self.layer_weights = nn.parameter.Parameter((maxval - minval) * torch.rand(mask_shape, dtype=torch.float32) + minval)
        self.sigmoid_activation = nn.Sigmoid()
    
    def forward(self, x):
        # x: 128x128x5x2
        mult = -1 * torch.log(1. / self.sigmoid_activation(self.layer_weights) - 1.) / self.slope # instead of sigmoid, what if add minimum of self.layer_weights to bring everything >0 instead?
        logit_weights = 0 * x[..., 0:1] + mult
        
        return self.sigmoid_activation(self.slope * logit_weights)




class RescaleProbMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sparsity, x):
        # x is 128x128x1
        xbar = torch.mean(x)
        r = sparsity / xbar
        beta = (1 - sparsity) / (1-xbar)
        le = torch.le(r, 1)
        # le = le.type(torch.FloatTensor).to("cuda:0")
        # le, r, beta are just single numbers
        out = le * x * r + (~le) * (1 - (1 - x) * beta)
        return out

# Realization of probability mask
class RandomMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, depth=5):
        # if this doesn't work, try threshs = torch.rand(shape=shape of input)
        lst = list(x.shape)
        lst[-2] = depth
        broadShape = tuple(lst)
        threshs = torch.rand(broadShape[1:], device=x.device)
        return (0*x) + threshs

class ThresholdRandomMask(nn.Module):
    def __init__(self, slope=12):
        super().__init__()
        self.slope = None
        if slope is not None:
#             self.slope = slope
            self.slope = nn.parameter.Parameter(torch.tensor(slope, requires_grad=False, dtype=torch.float32))
        self.sigmoid_activation = nn.Sigmoid()
    
    def forward(self, x):
        inputs = x[0]
        thresh = x[1]
        if self.slope is not None:
            return self.sigmoid_activation(self.slope * (inputs - thresh))
        else:
            output = inputs > thresh
            return output

class BSNMask(nn.Module):
    def __init__(self, slope=12):
        super().__init__()
        self.slope = None
        if slope is not None:
            #TODO: requires_grad?
            self.slope = nn.parameter.Parameter(torch.tensor(slope, requires_grad=False, dtype=torch.float32))
        self.threshold = Threshold.apply

    def forward(self, x):
        inputs = x[0]
        thresh = x[1]
        return self.threshold(inputs, thresh, self.slope)


class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold, slope):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(inputs, threshold, slope)
        return (inputs > threshold).type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, threshold, slope, = ctx.saved_tensors
#         input, threshold, slope = inputs
        sig_grad = slope * torch.sigmoid(slope*input) * (1 - torch.sigmoid(slope * input))
        return grad_output * sig_grad, grad_output * 0, grad_output * 0

class Undersample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        k_space_r = torch.mul(inputs[0][..., 0], inputs[1][..., 0])
        k_space_i = torch.mul(inputs[0][..., 1], inputs[1][..., 0])
        k_space = torch.stack([k_space_r, k_space_i], dim=-1)

        return k_space

# produces undersampled images as output, where undersampling pattern is parameterized.
class Undersampler(nn.Module):
    def __init__(self, 
                 input_shape=(128, 128, 5, 1),
                 sparsity=(1/20.0),
                 pmask_slope=5, 
                 pmask_init=None,
                 pmask_shape=(128, 128, 5, 1),
                 starting_slope=12,
                 bsn=False):
        super().__init__()
        self.sparsity = sparsity
        self.bsn = bsn

        # Layer initialization
        self.concatenate_zero = ConcatenateZero()
        self.fft = FFT()
        self.ifft = IFFT()
        self.prob_mask = ProbMask(slope=pmask_slope, mask_shape=pmask_shape)
        self.rescale_prob_mask = RescaleProbMap()
        self.random_mask = RandomMask()
        self.threshold_random_mask = ThresholdRandomMask(slope=starting_slope)
        self.bsn_mask = BSNMask(slope=starting_slope)
        self.undersample = Undersample()
    
    def forward(self, x, validate=False):
        # x: 128x128x5x1
        reshaped_input = x.permute(0, 2, 3, 1).unsqueeze(-1)

        complex_tensor = self.concatenate_zero(reshaped_input)

        # print(f"complex_tensor: {complex_tensor.shape}")
        # 128x128x5x2
        k_tensor = self.fft(complex_tensor)

        # print(f"k_tensor: {k_tensor.shape}")
        
        # k_tensor is 128x128x5x2
        prob_mask_tensor = self.prob_mask(k_tensor) # this is 128x128x5x1

        # print(f"prob_mask_tensor: {prob_mask_tensor.shape}")       
       
        #print("Nans at prob_mask_tensor: ", torch.isnan(prob_mask_tensor).any())
        # feed each mask through individually, 128x128x1, and concatenate back together afterwards
        prob_mask_0 = self.rescale_prob_mask(self.sparsity, prob_mask_tensor[:, :, :, 0])
        prob_mask_1 = self.rescale_prob_mask(self.sparsity, prob_mask_tensor[:, :, :, 1])
        prob_mask_2 = self.rescale_prob_mask(self.sparsity, prob_mask_tensor[:, :, :, 2])
        prob_mask_3 = self.rescale_prob_mask(self.sparsity, prob_mask_tensor[:, :, :, 3])
        prob_mask_4 = self.rescale_prob_mask(self.sparsity, prob_mask_tensor[:, :, :, 4])
        # print(f"prob_mask_4: {prob_mask_4.shape}")

        prob_mask_tensor = torch.stack((prob_mask_0, prob_mask_1, prob_mask_2, prob_mask_3, prob_mask_4), dim=-2)

        # print(f"prob_mask_tensor: {prob_mask_tensor.shape}")
        
        #print("Nans at prob_mask_tensor: ", torch.isnan(prob_mask_tensor).any())
        # prob_mask_tensor is 128x128x5x1
        # Try this one if the above one works poorlys
        # prob_mask_tensor = RescaleProbMap(self.sparsity, prob_mask_tensor)
        
        # thresh_tensor is 128x128x5x1
        thresh_tensor = self.random_mask(prob_mask_tensor, depth = 5) # depth refers to number of images fed into NN. 5 images.

        # print(f"thresh_tensor: {thresh_tensor.shape}")
        # last_tensor_mask is 128x128x5x1
        # last_tensor_mask = torch.round(self.threshold_random_mask([prob_mask_tensor, thresh_tensor]))  # NOTE: Torch.round is trial to make mask binary
        if self.bsn:
            last_tensor_mask = self.bsn_mask([prob_mask_tensor, thresh_tensor])
        else:
            last_tensor_mask = self.threshold_random_mask([prob_mask_tensor, thresh_tensor])
        # print(f"last_tensor_mask: {last_tensor_mask.shape}")

        # under_tensor, i.e. undersampled k space data, is 128x128x5x2
        under_tensor = self.undersample([k_tensor, last_tensor_mask])
        
        # last_tensor, i.e. complex-valued images, is 128x128x5x2
        last_tensor = self.ifft(under_tensor)

        # print(f"last_tensor: {last_tensor.shape}")

        # first_image = last_tensor[:, :, 0, :]
        # print("shape of the image: ", first_image.shape)
        # visualize_complex_image(last_tensor[:, :, 0, :])
        # visualize_complex_image(last_tensor[:, :, 1, :])
        # visualize_complex_image(last_tensor[:, :, 2, :])
        # visualize_complex_image(last_tensor[:, :, 3, :])
        # visualize_complex_image(last_tensor[:, :, 4, :])

        # reshaped_output_img = torch.cat([last_tensor[..., 0], last_tensor[..., 1]], -1).permute(dims=(2, 0, 1)).unsqueeze(dim=0)
        # reshaped_output_freq = torch.cat([under_tensor[..., 0], under_tensor[..., 1]], -1).permute(dims=(2, 0, 1)).unsqueeze(dim=0)

        reshaped_output_img = last_tensor.reshape(-1, 128, 128, 10).permute(dims=(0, 3, 1, 2))#.unsqueeze(dim=0)
        reshaped_output_freq = under_tensor.reshape(-1, 128, 128, 10).permute(dims=(0, 3, 1, 2))#.unsqueeze(dim=0)

        # if validate:
        #     return reshaped_output_freq, reshaped_output_img
        return reshaped_output_freq, reshaped_output_img, last_tensor_mask
