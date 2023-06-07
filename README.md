# Dual-domain Accelerated MRI Reconstruction using Transformers with Learning-based Undersampling
Python implementation in PyTorch of the models outlined in the above titled paper. We use a dual domain (image and k-space) approach using transformers, coupled with learning-based undersampling to perform accelerated reconstruction of dynamic MRI images. Paper to come.

## Training
To train, use the training.py file. You will need to provide a dataset since the one used in the paper has privacy restrictions preventing us from sharing. Then, set the paths to the training and validation datasets in the training.py file as well as any hyperparameters that you choose. Simply running training.py will then train the model and save it to a specified location.

### Dataset format
When creating a dataset, it must match the following specifications exactly to use the dataloader we provide.

- numpy array file (.npy) with dimensions ```(# of images) x image width (128) x image height (128)```
- single channel images
- must be sequences of length 50 images (ie. 0-49 are sequential and from one video, 50-99 are sequential and from a second video, etc.)

## Testing
To test, open the results_example.json file and add the path that you used for saving weights in the training step. Make sure all the other fields are accurate, set the paths to the json and dataset files in the validation.py file and then run that file to generate the results.

# Citations
If using this open-source code, please use the following citation:
Hong, G.Q., Morley, W.A.W., Wan, M., Su, Y., Cheng, H.L.M. (2022). Dual domain Accelerated MRI Reconstruction using Transformers with Learning based Undersampling. [Computer software]. GitHub. https://github.com/william2343/dual-domain-mri-recon-nets

# Acknowledgments
This code draws from the [UFormer](https://github.com/ZhendongWang6/Uformer) & [LOUPE](https://github.com/cagladbahadir/LOUPE) repositories.

# Abstract
Acceleration in MRI has garnered much attention from the deep-learning community in recent years, particularly for imaging large anatomical volumes such as the abdomen or moving targets such as the heart. A variety of deep learning approaches have been investigated, with most existing works using convolutional neural network (CNN)-based architectures as the reconstruction backbone, paired with fixed, rather than learned, k-space undersampling patterns. In both image domain and k-space, CNN-based architectures may not be optimal for reconstruction due to its limited ability to capture long-range dependencies. Furthermore, fixed undersampling patterns, despite ease of implementation, may not lead to optimal reconstruction. Lastly, few deep learning models to date have leveraged temporal correlation across dynamic MRI data to improve reconstruction. To address these gaps, we present a dual-domain (image and k-space), transformer-based reconstruction network, paired with learning-based undersampling that accepts temporally correlated sequences of MRI images for dynamic reconstruction. We call our model DuDReTLU-net. We train the network end-to-end against fully sampled ground truth dataset. Human cardiac CINE images undersampled at different factors (5−100) were tested. Reconstructed images were assessed both visually and quantitatively via the structural similarity index, mean squared error, and peak signal-to-noise. Experimental results show superior performance of DuDReTLU-net over state-of-the-art methods (LOUPE, k-t SLR, BM3D-MRI) in accelerated MRI reconstruction; ablation studies show that transformer-based reconstruction outperformed CNN-based reconstruction in both image domain and k-space; dual-domain reconstruction architectures outperformed single-domain reconstruction architectures regardless of reconstruction backbone (CNN or transformer); and dynamic sequence input leads to more accurate reconstructions than single frame input. We expect our results to encourage further research in the use of dual-domain architectures, transformer-based architectures, and learning-based undersampling, in the setting of accelerated MRI reconstruction. 
