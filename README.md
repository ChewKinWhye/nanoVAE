<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



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
git clone https://github.com/your_username_/Project-Name.git
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
<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Chew Kin Whye - kinwhyechew@gmail.com

