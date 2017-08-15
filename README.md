# tf_metadynamics: Using Software for things they are not meant to be used for

This repo is a fun weekend project designed to show how computational graphs can be turned into collective variables for enhanced sampling using existing libraries. This was done in two parts. 

PyTorchMetadynamics.ipynb 
1). Implement [Metadynamics](https://en.wikipedia.org/wiki/Metadynamics) into [PyTorch](http://pytorch.org/) using a custom loss function depenedent on the history. The forces then become the negative of the derivatives which are automatically obtained via back propagation. The details are in the jupyter notebook PyTorchMetadynamics.This was performed on the Muller potential. 

PlumedImagesNeuralNetwork.ipynb
2). Implement a 3-layer Image Net Classifier into [Plumed](plumed.github.io), encode the images as molecular trajectories, and use Plumed to predict the raw scores.
