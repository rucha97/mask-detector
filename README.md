# mask-detector
 CNN model for detecting type of mask from facial images

################
################
# Part - 1
################
################

# Instructions

1) Extract data.zip to have data folder in current working directory. Within data folder there is a subdirectory called classified which has all the images for the 4 classes.

2) It is best to run the included .ipynb file but I have also provided .py file in case you like to run it in terminal.
For the .py file,
    To train a new model, run command:
    $ python demo.py train
    
    To test a previously saved model, run command:
    $ python demo.py test <path to saved .pkl file>
    
# List of file
1) main.ipynb - Jupyuter notebook used for project
2) demo.py - python file to train and test models, to be used when jupyter file can't be run.
3) Data.zip - Dataset
4) ConvNet5.pkl - saved file with trained model, optimizer, loss func & epochs.
5) README.txt - this file.
6) Report.pdf - Project report.
7) urls.txt - URLs for imagedownloaded from internet
8) Expectations-of-Originality-Feb14-2012.pdf - Originality form signed by all team members

#Image sources
Most of the images are obtained from Kaggle for which we have made reference in the report(please check there). Still, the classes were imbalanced so we have added more images for which urls are in urls.txt file.

################
################
# Part - 2
################
################

1) Data.zip now has 2 more folders, bias analyses and supplemental data
2) Assignment part 2.ipynb is used for doing the bias analysis and K fold cross validation.
3) Report.pdf is updated with new content for k-fold  cross validation and bias analysis.
4) bew_urls.txt is added for referencing the the supplemental images.
5) models directory contains both previous and new model after training.