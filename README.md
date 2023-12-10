![Python version](https://img.shields.io/badge/python-3.8-brightgreen.svg) [![DOI](https://zenodo.org/badge/583738628.svg)](https://zenodo.org/badge/latestdoi/583738628)


# TrackSegNet

## Purposes


`TrackSegNet` is a tool designed for the classification and segmentation of experimental trajectories, specifically those obtained from single-particle tracking microscopy data, into different diffusive states.

To enable the training of the LSTM neural network, synthetic trajectories are initially generated, and the parameters of the generator can be fine-tuned.

Upon completion of the training process, the experimental trajectories are classified at each point using the trained model. Subsequently, the trajectories are segmented and grouped based on their respective diffusive states. In this context, "diffusive states" refer to the distinct modes or patterns observed in the movement of particles.

For each segmented track, the diffusion constant ($D$) and anomalous exponent ($\alpha$) are further estimated. This is accomplished by computing the mean squared displacement (MSD), providing valuable insights into the dynamic behavior of the particles within each identified diffusive state.


![pipeline](paper/pipeline.png)

## Installation and requirements

### Clone the repository
```
git clone https://github.com/hkabbech/TrackSegNet.git
cd TrackSegNet
```

### Either create and run a docker container

Build a docker image (Rebuild the image after changing the parameters):
```bash
docker compose build
```
Run the container:

```bash
docker compose run tracksegnet-env
```

### Or create the environment and install the packages

```bash
sudo apt install python3-virtualenv
virtualenv -p /usr/bin/python3 tracksegnet-env
source tracksegnet-env/bin/activate # for Windows: tracksegnet-env\bin\activate
python -m pip install -r requirements.txt
```

Run the program:

```bash
./tracksegnet.py parms.csv
```

Note, to deactivate the virtual environment, type `deactivate`.


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

Update the main parameters in the `parms.csv` file according to your experiment:

- `data_path`: the path containing your data folder `SPT_experiment` to analyze
- `track_format`: The format of the files containing the trajectory coordinates, should be `MDF` or `CSV`
- `time_frame`: the time interval between two trajectory points (in second)
- `pixel_size`: the dimension of a pixel (in µm)
- `num_states`: the number of diffusive states for the classification(from 2 to 6 states)
- `state_X_diff`: The expected diffusion value for state X (in µm^2/s).
- `state_X_alpha`: The expected anomalous exponent α value for state X (from 0 to 2 -- ]0-1[: subdiffusion, 1: Brownian motion, ]1-2[: superdiffusion).
- `pt_i_j`: the probability of transitionning from the state i to the state j. The total number of probabilities should be $N^2$.

Note that the program will run on the toy example if the parameters are unchanged.

For updating the parameters of the track simulation and neural network training, please make the changes in the main file `tracksegnet.py`.


## Reference

Yavuz, S., Kabbech, H., van Staalduinen, J., Linder, S., van Cappellen, W.A., Nigg, A.L., Abraham, T.E., Slotman, J.A., Quevedo, M. Poot, R.A., Zwart, W., van Royen, M.E., Grosveld, F.G., Smal, I., Houtsmuller, A.B. (2023). Compartmentalization of androgen receptors at endogenous genes in living cells, *Nucleic Acids Research* 51(20), [https://doi.org/10.1093/nar/gkad803](https://doi.org/10.1093/nar/gkad803). 

