<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project

Detections of modifications in our DNA is very important because blah blah blah.
Supervised methods exists, but is unable to detect the modifications unknown during the labelling process.
Therefore, we use take unsupervised approach, Variational Autoencoders, to this problem to help us overcome the limitations of data labelling.

This project contains the code for both DNA and RNA data.

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Datasets required to run this code
* Deepsignal dataset
* RNA dataset (Explain how to get this)

### Installation

1. Clone the repo
```sh
git clone https://github.com/ChewKinWhye/nanoVAE.git
```
2. Create venv
```sh
Look up commands for venv
```
3. Install requirements
```sh
pip install requirements.txt
```

<!-- USAGE EXAMPLES -->
## Usage

For DNA modifications
```sh
python VAE_DNA.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 500000 --output_filename VAE_DNA
```

For RNA modifications
```sh
python VAE_RNA.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 900000 --output_filename VAE_RNA
```

<!-- CONTACT -->
## Contact

Chew Kin Whye - kinwhyechew@gmail.com

