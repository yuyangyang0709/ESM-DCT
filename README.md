# ESM-DCT
The catalog consts of:
1.The inputdata after calculating the global area-weighted mean.
2.The outputdata of the reconstruction errors in the BGRU_AE and Post processing scripts
3.Source code of training and testing the BGRU_AE.

The program runs in the following steps:
1. Put the Inputdata/data (the training datasets and validation datasets should be splited) into train+validate_BiGRU_autoencoding.py to train the BGRU-AE model. In train+validate_BiGRU_autoencoding.py, the sequence_length is the number of variables. The learning rate is used to control the magnitude of updating weights in each iteration of the model. The epoch is the time when the training datasets have been used by the model for training. The layer number is the number of hidden layers. The hidden-size is the number of units in each hidden layer. The batch-size is used to define the number of training samples used by the model in each iteration. 
2. Put the Inputdata/data (training datasets) into threshold_BiGRU_autoencoding.py to get the threshold of reconstruction errors of training datasets.
3. Put the Inputdata/data_test (testing datasets) in to test_BiGRU_autoencoding.py. If the reconstruction error of one member of the testing datasets can be higher for those of maximum value of the training datasets, the ESM-DCT return the member as “failure”. 

Computing environments:
Python 3.6
