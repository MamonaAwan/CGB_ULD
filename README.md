# Unsupervised Landmark Discovery via Consistency-Guided Bottleneck

## Requirements
-Linux
-Python 3.8 or later
-PyTorch 1.8 with torchvision
-OpenCV

## Data
CelebA can be obtained from [here](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/celeba.zip). 
MAFL (training & test) is included.
Bounding box obtained to crop the images is computed from the landmarks provided in the CelebA dataset.
place the file ``list_landmarks_align_celeba.txt"" in the folder.

AFLW can be found [here](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/aflw_release-2.zip).
300W-LP dataset can be found [here](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing), LS3D can be downloaded from [here](https://www.adrianbulat.com/face-alignment).
Catshead Dataset can be found [here](https://www.kaggle.com/datasets/crawford/cat-dataset).
Shoes Dataset can be downloaded from [here](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/).

## Testing
To test our method, use the pretrained models in the folder ``Models_to_Test"", run the testing script.
Run the command
```
python test.py -f Models_to_Test -e alfw -d AFLW --data_path <path to dataset>
or
python test.py -f Models_to_Test -e mafl -d MAFL --data_path <path to dataset>
```
## Pretrained Models
Pretrained models are provided here.

## Training
To train our method use the corresponding command in training script.

