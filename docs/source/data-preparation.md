# Prepare your data


## Data organization


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

## Change the main parameters


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
