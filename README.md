![Python version](https://img.shields.io/badge/python-3.8-brightgreen.svg) [![DOI](https://zenodo.org/badge/583738628.svg)](https://zenodo.org/badge/latestdoi/583738628)


# TrackSegNet

## Purposes

Recent advances in the field of microscopy allow the capture, at nanometer resolution, of the motion of fluorescently-labeled particles in live cells such as proteins or chromatin loci. Therefore, the development of methods to characterize the dynamics of a group of particles has become more than necessary.

`TrackSegNet` is a tool designed for the classification and segmentation of experimental trajectories, specifically those obtained from single-particle tracking microscopy data, into different diffusive states.

- To enable the training of the LSTM neural network, synthetic trajectories are initially generated, and the parameters of the generator can be fine-tuned.

- Upon completion of the training process, the experimental trajectories are classified at each point using the trained model. Subsequently, the trajectories are segmented and grouped based on their respective diffusive states. In this context, "diffusive states" refer to the distinct modes or patterns observed in the movement of particles.

- For each segmented track, the diffusion constant (![equation](https://latex.codecogs.com/svg.image?\inline&space;D)) and anomalous exponent (![equation](https://latex.codecogs.com/svg.image?\inline&space;\alpha)) are further estimated. This is accomplished by computing the mean squared displacement (MSD), providing valuable insights into the dynamic behavior of the particles within each identified diffusive state.


![pipeline](paper/pipeline.png)

## Installation and requirements

### Clone the repository
```
git clone https://github.com/hkabbech/TrackSegNet.git
cd TrackSegNet
```

### Either create and run a docker container

```bash
# Build a docker image (Rebuild the image after changing the parameters):
docker compose build
# Run the container:
docker compose run tracksegnet-env
```

### Or create a virtual environment and install the packages

Requirement: python3.8 or equivalent and the virtualenv library

```bash
# Create the environment:
python -m venv tracksegnet-env
# virtualenv -p /usr/bin/python3 tracksegnet-env
# Activate the environment:
source ./tracksegnet-env/bin/activate # for Windows PowerShell: .\tracksegnet-env\Scripts\Activate.ps1 (run as administrator)
# Install the required python libraries:
python -m pip install -r requirements.txt
```

Note for later, to deactivate the virtual environment, type `deactivate`.


## Prepare your data

### Data organization

Organize your data in a folder `SPT_experiment`, each sub-folder should contain a file storing the trajectory coordinates in a `MDF` or `CSV` file format.

If `CSV` format is used, the headers should be: `x, y, frame, track_id`

```bash
.
├── data/
│   └── SPT_experiment/
│       ├── Cell_1
│       │    ├── *.tif
│       │    └── *.mdf
│       ├── Cell_2
│       │    ├── *.tif
│       │    └── *.mdf
│       ├── Cell_3
│       │    ├── *.tif
│       │    └── *.mdf
│       └── ...
│
├── src/
├── tracksegnet-env/
├── parms.csv
├── tracksegnet.py
└── ...
```

### Change the main parameters

Tune the main parameters of the training in the `params.csv` file according to your experiment:

* `num_states` the number of diffusive states for the classification(from 2 to 6 states). This number can vary from 2 to 6 states, but it is recommended to choose 2 to 4 states.
* `state_i_diff` and `state_i_alpha` the approximate motion parameters for each of the ![equation](https://latex.codecogs.com/svg.image?\inline&space;N) diffusive state. The diffusion constant ![equation](https://latex.codecogs.com/svg.image?\inline&space;D) is dimensionless, and the anomalous exponent value ![equation](https://latex.codecogs.com/svg.image?\inline&space;\alpha) is ranging from 0 to 2 (![equation](https://latex.codecogs.com/svg.image?\inline&space;]0-1[): subdiffusion, ![equation](https://latex.codecogs.com/svg.image?\inline&space;1): Brownian motion, ![equation](https://latex.codecogs.com/svg.image?\inline&space;]1-2[): superdiffusion).
* `pt_i_j` the probability of transitionning from state i to state j. The total number of probabilities should be ![equation](https://latex.codecogs.com/svg.image?\inline&space;N^2).

The remaining parameters are related to the experimental dataset:

* `data_path`, the path of the dataset of trajectories to segment.
* `track_format`, the format of the files containing the trajectory coordinates, either `MDF` (see `MTrackJ` data file format) or `CSV`
* `time_frame`, the time interval between two trajectory points in seconds.
* `pixel_size`, the dimension of a pixel in ![equation](https://latex.codecogs.com/svg.image?\inline&space;$\mu m).


Note that the program will run on the toy example if the parameters are unchanged.

For updating the parameters of the track simulation and neural network training, please make the changes in the main file `tracksegnet.py`.


## Run the program


```bash
python tracksegnet.py parms.csv
```

## Reference

Yavuz, S., Kabbech, H., van Staalduinen, J., Linder, S., van Cappellen, W.A., Nigg, A.L., Abraham, T.E., Slotman, J.A., Quevedo, M. Poot, R.A., Zwart, W., van Royen, M.E., Grosveld, F.G., Smal, I., Houtsmuller, A.B. (2023). Compartmentalization of androgen receptors at endogenous genes in living cells, *Nucleic Acids Research* 51(20), [https://doi.org/10.1093/nar/gkad803](https://doi.org/10.1093/nar/gkad803). 

