# EEG Classification Experiments

This repository contains code for training and evaluating a CNNTransformerLSTM for EEG signal classification. The model is designed to classify different types of motor execution (ME) and motor imagery (MI) tasks from EEG data.

## Prerequisites

### Installation

This project uses Python 3.10

The repository includes a `requirements.txt` file with all necessary dependencies. To install:

```bash
pip install -r requirements.txt
```

## Datasets

The project contains two packages each of which contains script for training and testing using a respective dataset. In each package there is a .ini configuration file.
For the Kodera-29 dataset, the original implementation has been preserved and is described in README_original.md file.

## data availability
- Shuqfa-103: https://data.mendeley.com/datasets/dpmtgrn8d8/4
- Brunner-9: https://www.bbci.de/competition/iv/
- Kodera-29: https://zenodo.org/records/7893847


## Running the Experiments

To run the experiment with the default configuration:


```bash
python shuqfa103_train.py [-f|--config_file <path_to_config>]
```

```bash
python kodera29_train.py [-f|--config_file <path_to_config>]
```

