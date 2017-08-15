# tf_metadynamics: Using PyTorch to enhance molecular simulations and using Plumed to classify images

This repo is a fun weekend project designed to show how complex PyTorch computational graphs can be turned into collective variables inside Plumed. This was done in two parts.  

PyTorchMetadynamics.ipynb 

This Jupyter notebook encodes  [Metadynamics](https://en.wikipedia.org/wiki/Metadynamics) into [PyTorch](http://pytorch.org/) using a custom loss function depenedent on the history. The forces then become the negative of the derivatives which are automatically obtained via back propagation. This was performed on the Muller potential.  

PlumedImagesNeuralNetwork.ipynb

This Jupyter Notebook transfers a 3-layer Image Net PyTorch Classifier into [Plumed](plumed.github.io), encodes the images as molecular trajectories, and use Plumed to predict the un-normalized image scores. The plumed input file ```image_plumed.dat``` has the actual plumed neural network script.
