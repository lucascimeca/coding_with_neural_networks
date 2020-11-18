# Coding With Neural Networks, by Luca Scimeca

![Alt Text](https://github.com/lucascimeca/coding_with_neural_networks/blob/master/assets/modularTLfCD-NN_no_title_short.gif)
![Alt Text](/assets/modularTLfCD-NN_no_title_short.gif | width=100)

The code in this repository has been developed as a project for the course "Center for Brains, Minds and Machines summer school 2019". 
The code is meant to capture a novel compositional neural network achitecture developed during the course, and shown to be able to both learn to play "Game of Life" on arbitrarily long boards as well as perform "parity bit checking" on arbitrarily long bite strings.

* To now more about the project look at the presentation in "Luca Scimeca - Coding with Neural Networks.pdf" in the "media folder".
* To see some demo runs of the code look at the video files in the "media folder"

## Installation

The code was tested on pycharm in an environment with the following dependencies:

* python 3.6.7
* imutils 0.5.2
* tensorflow-gpu 1.14.0
* opencv 3.4.2
* numpy 1.15.4
* matplotlib 2.2.3


## Usage

The code developed for the course is contained in the python files within the higher level folder and the `data_providers.py` within the 'mlp' folder.
All train models are contained within the 'out' folder, including training stats, checkpoints and frozen graphs. 

### Game of Life

To run the example run of Game of Life as learned by the networks just run the code in `play_game_of_life.py`.

To retrain the models within the game refer to `train_models.py`, then uncomment and run the sub architectures as necessary. After the training ends you can re-run the game with the re-learned models through `play_game_of_life.py`.


### Parity bit Checking

To run the example run of Parity bit Checking as learned by the networks just run the code in `parity_bit_checking.py`.

To retrain the models for the test refer to `train_models.py`, then uncomment and run the sub architectures as necessary. After the training ends you can re-run the test with the re-learned models through `parity_bit_checking.py`.



## License
[GPL](https://www.gnu.org/licenses/#GPL)
