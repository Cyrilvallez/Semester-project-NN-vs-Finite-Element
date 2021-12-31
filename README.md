# Semester project : Characterisation of neural networks and comparison with finite element methods

Author : Cyril Vallez (<cyril.vallez@epfl.ch>)

_In this project, we study how neural networks compute approximation of real function in one and several dimensions. We put an emphasis on how the approximation is constructed, meaning that we try to visualize and understand what happens in the neurons in the hidden layers of the networks. We compare this to finite element methods, for which analytic bounds and convergence order of the approximations are well known. Moreover, we try to understand what parameters are most important when computing such approximations with neural networks, and the impact they have on the global and local quality of the approximation._

The final report _report.pdf_ is included.

## Folders
- Data : *contains all the data used to train and test the models*
- Saved-models : *contains all the models that have been trained* 
- Figures : *contains all saved figures*
- Tools : *Internal library containing helpers functions and model definitions*

## Libraries and Packages
External Libraries
- NumPy
- Matplotlib
- Pandas
- Tensorflow
- Keras
- TQDM
- Widgets

Internal Libraries
- Tools : *contains the definition of the different tensorflow models used in _Custom_models.py_, some helpers functions in _Helpers.py_ and some pre-set to create nice figures using matplotlib in _Plot.py_*
