![Python version](https://img.shields.io/badge/python-3.8-brightgreen.svg)

# TrackSegNet

Trajectory segmentation into diffusive states using LSTM neural network.

## Installation and requirements

### Clone the repository
```
git clone https://github.com/hkabbech/TrackSegNet.git
cd TrackSegNet
```

### Activate the python environment

On Unix or MacOS, run:
```
source tracksegnet-env/bin/activate
```
On Windows, run:
```
tracksegnet-env\Scripts\activate
```
Note, to deactivate the virtual environment, type `deactivate`

### (Create the environment and install the packages)

```
pip install virtualenv
python -m venv tracksegnet-env
source tracksegnet-env/bin/activate
python -m pip install -r requirements.txt
```

## Prepare your data

### Data organization

Organize your data in a folder `SPT_Experiment`, each sub-folder should contain a file containing the trajectories in `mdf` format.

```
.
├── data/
│   └── SPT_Experiment/
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

- `data_path`: the path containing your data folder `SPT_Experiment` to analyze
- `time_frame`: the time between two frames (in second)
- `pixel_size`: the size of a pixel (in micrometer)
- `num_states`: the number of diffusive states for the classification(from 2 to 6 states)
- `state_X_diff`: The expected diffusion value for state X (in μm^2/s).
- `state_X_alpha`: The expected anomalous exponent alpha value for state X (from 0 to 2 -- 0-1: subdiffusion, 1: Brownian motion, 1-2: superdiffusion).

Note that the program will run on the toy example if the parameters are unchanged.

For updating the parameters of the track simulation and neural network training, please make the changes in the main file `tracksegnet.py`.


## Run the program

```
python tracksegnet.py parms.csv
```

## Reference

Arts, M., Smal, I., Paul, M. W., Wyman, C., & Meijering, E. (2019).
**Particle Mobility Analysis Using Deep Learning and the Moment Scaling Spectrum.** _Scientific Reports_, 9(1), 17160. [https://doi.org/10.1038/s41598-019-53663-8](https://doi.org/10.1038/s41598-019-53663-8).
