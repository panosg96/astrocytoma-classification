# astrocytoma-classification

This program is developed in matlab and the goal is to classify a histopathological image of the brain, in order to detect the grade of a possible astrocytoma.

The user initially loads a histopathological image. 
The image goes through a pre-processing state in order to be prepared for the next steps. 
After applying certain functions in order to isolate the cell nuclei, the segmented image is ready and then the next step is to extract features. 
After extracting several morphological and texture features, the data is prepared and then the selection of features follows. 
The classifiers that are used are kNN (k=3), Probabilistic Neural Network and Bayesian Classifier. 
After getting a result from every classifier, the final result is created through a majority voting which states the final grade (High or Low)  


The project consists of 3 files:
main.m --> This is the main program where the user can load an image in order to get the result of the grade detection

data_extraction.m --> This file is used to extract the features from the images that are used as a dataset, and then these are used to create the 2 classes for high and low grade labels

Exhaustive_LOO_training.m --> This file is used to get the results of the accuracy for every classifier that is used, implmenting an exhaustive search and a Leave-one-out method between the features. This was done in order to find the best combination of features that can be used for a maximum accuracy in the classification.
