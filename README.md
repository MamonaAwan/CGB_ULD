# Unsupervised Landmark Discovery via Consistency-Guided Bottleneck

## Install Requirements
- Python 3.8 or later
- PyTorch 1.8 with torchvision
- OpenCV

## Data
- CelebA can be obtained from [here](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/celeba.zip). 
MAFL (training & test) is included.
Bounding box obtained to crop the images is computed from the landmarks provided in the CelebA dataset.
place the file 'list_landmarks_align_celeba.txt' in the same folder.

- AFLW can be found [here](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/aflw_release-2.zip).
- 300W-LP dataset used for training can be found [here](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing).
LS3D used for testing can be downloaded from [here](https://www.adrianbulat.com/face-alignment).
- Catshead Dataset can be found [here](https://www.kaggle.com/datasets/crawford/cat-dataset).
- Shoes Dataset can be downloaded from [here](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/). Numerical results for shoes are not possible since there are no ground truth annotations.

## Testing
To test our the pretrained models, download from the links below and place them in the folder ``pretrained_models_to_test"", run the testing script 'test_pretrained_model_script.sh'.

## Pretrained Models
Pretrained models are provided here.

## Training / Testing
To train/test our method use the corresponding command in the provided training/testing script.

