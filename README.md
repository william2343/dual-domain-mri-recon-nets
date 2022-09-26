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
*Coming Soon*

# Acknowledgments
This code draws from the [UFormer](https://github.com/ZhendongWang6/Uformer) & [LOUPE](https://github.com/cagladbahadir/LOUPE) repositories.

# Abstract
