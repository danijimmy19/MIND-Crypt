# MIND-Crypt: A Machine Learning-Based Framework for Assessing Cryptographic Indistinguishability of Lightweight Block Ciphers

This repository provides machine learning-based framework, MIND-Crypt. It contains experiments and utilities for assessing the indistinguishability of lightweight block ciphers SPECK32/64 and SIMON32/64. Code is organized to support reproducible research and rapid experimentation for cryptanalysis tasks.

> **Note:**  
> This paper is published in  
> **22nd Annual International Conference on Privacy, Security, and Trust (PST 2025)**

## Environment Setup
Install dependencies via `conda`:
```Shell
conda env create -f resources/environment.yml
conda activate mind-crypt
```

## Data Generation
- For IND experiments with the **SPECK cipher**, the dataset is generated automatically by the experiment script during the training, validation, and testing of the deep learning model.
- To generate a dataset for the **SIMON cipher**, use the script provided at `src/simon-speck/Python/simon/simon-data-gen.py`. After generating the dataset, proceed with your IND analysis by running `simon-IND-experiment.py` on the newly created data.

# Usage
All main experiments are located in `src/` and can be executed through the command line.

## Run SPECK Indistinguishability Experiment
```shell
python src/speck-IND-experiment.py \
  --num_rounds 15 \
  --depth 10 \
  --num_epochs 200 \
  --bs 5000 \
  --models_dir_path /path/to/save/trained/model \
  --dataset_dir_path /path/to/speck-dataset \
  --statistics_dir_path /path/to/save/results
```

## Run SIMON Indistinguishability Experiment
```shell
python src/simon-IND-experiment.py \
  --num_rounds 15 \
  --depth 10 \
  --num_epochs 200 \
  --bs 5000 \
  --models_dir_path /path/to/save/trained/model \
  --dataset_dir_path /path/to/gohr/simon-dataset \
  --statistics_dir_path /path/to/save/results
```


## Source Notebooks
- `Baseline Experiment - Gohr Implementation - random vs real.ipynb`: This notebook is used for replicating the results of distinguishing random text from the real ciphertext (`random-vs-real`) using Gohr's experimental settings.

# Citation
If this work is useful for your research, please cite:
```text
@inproceedings{yourkey2025,
  author = {Jimmy Dani, Kalyan Nakka and Nitesh Saxena},
  title = {A Machine Learning-Based Framework for Assessing Cryptographic Indistinguishability of Lightweight Block Ciphers},
  howpublished = {Cryptology {ePrint} Archive, Paper 2024/852},
  year = {2024},
  url = {https://eprint.iacr.org/2024/852}
}
```


## Contact
For questions, please contact Jimmy Dani, Texas A&M University ([danijy@tamu.edu](mailto:danijy@tamu.edu)).