# MACHINE FAULT CLASSIFICATION USING HAMILTONIAN NEURAL NETWORKS
Jeremy Shen, Sourav Banerjee, Gabriel Terejanu

## Abstract
A  new  approach  is  introduced  to  classify  faults  in  rotating machinery based on the total energy signature estimated from sensor measurements. The overall goal is to go beyond using black-box  models  and  incorporate  additional  physical  con-straints that govern the behavior of mechanical systems. Observational data is used to train Hamiltonian neural networks that describe the conserved energy of the system for normaland various abnormal regimes. The estimated total energyfunction, in the form of the weights of the Hamiltonian neural network, serves as the new feature vector to discriminate between the faults using off-the-shelf classification models. The experimental results are  obtained using the MaFaulDadatabase, where the proposed model yields a promising area under the curve (AUC) of 0.78 for the binary classification(normal vs abnormal) and 0.84 for the multi-class problem(normal, and 5 different abnormal regimes). 

## Dependencies
-Pytorch

-Pycaret

-Numpy

## Acknowledgements
Extension to Sam Greydanus, Misko Dzamba, and Jason Yosinski's original work on [Hamiltonian Neural Networks](https://github.com/greydanus/hamiltonian-nn)
This material is based upon work supported by the National Institute of Food and Agriculture (NIFA)/USDA under Grand No. 2017-67017-26167.
