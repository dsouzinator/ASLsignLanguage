# ASLsignLanguage
This project was done in colaboration with [Chirag Gomber](https://github.com/ArtistBanda) 

A deep learning network trained to identify ASL alphabets from a live video stream. The letters X, Z, 'nothing' and 'cancel' were excluded from this dataset.

A future goal is to add these letters into a word along with an autocorrect feature.
The dataset can be obtained from [here](https://www.kaggle.com/grassknoted/asl-alphabet). 

Transfer learning was performed on a ResNet50 network. The model was pre trained on the imagenet database.  

The file [resnet_model.py](resnet_model.py) contains code used to train the network. 

[runModel.py](runModel.py) runs the network with a video stream from a webcam.

![The ASL alphabet language dataset](https://www.nidcd.nih.gov/sites/default/files/Content%20Images/NIDCD-ASL-hands-2014.jpg)
