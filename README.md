# MIND-Crypt
A Machine Learning-Based Framework for Assessing Cryptographic Indistinguishability of Lightweight Block Ciphers


## Data Generation
- For IND experiments with SPECK cipher, the dataset will be generated directly by the scripts training, validating and testing the DL model.
- To generate dataset for the SIMON cipher, the code is provided in the `simon-speck/Python/simon/simon-data-gen.py`. Once the dataset is generated used the `SIMON-IND-experiment.py` file for conducting IND experiments on `SIMON` cipher. 
