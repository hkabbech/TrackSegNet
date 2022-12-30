![Python version](https://img.shields.io/badge/python-3.7-brightgreen.svg)

# DL-MSS: A Particle Mobility Analysis tool

**DL**: Classification of trajectories into X tracklet states using a LSTM neural network  
**MSS**: Analysis of tracklets with the Moment Scaling Spectrum

## Installation and requirements

### Clone the repository
```
git clone https://github.com/hkabbech/DL-MSS.git
cd DL-MSS
```

### Dependencies

- Conda/[Miniconda3](https://docs.conda.io/en/latest/miniconda.html) to create an environment with the python librairies
- [LaTeX](https://www.latex-project.org/get/) (TeX Live distribution recommended) to write mathematical formula on the plots

### Installation of the required python packages

#### By creating a conda environment
```
conda env create --file environment.yml
conda activate dl-mss-env
```
#### Or manually with `pip install`
```
pip install ipython numpy pandas scipy tqdm matplotlib seaborn colorcet pillow more-itertools pydotplus tensorflow==2.2 keras==2.3.1
```

## Prepare your data

### Organization in a folder 

Organize your data in a folder `Experiment_X`, each sub-folder should contains a `*.mdf` file containing the trajectories to analyse and a `*.tif` file the maximum projection of the time-lapse.

```
.
├── data/
│   └── Experiment_X/
│       ├── Cell_1
│       │   ├── *.tif
│       │   └── *.mdf
│       ├── Cell_2
│       │   ├── *.tif
│       │   └── *.mdf
│       ├── Cell_3
│       │   ├── *.tif
│       │   └── *.mdf
│       └── Cell_4
│           ├── *.tif
│           └── *.mdf
├── results/
├── src/
├── dl_mss.py
├── parameters.py
├── environment.yml
└── README.md
```

### Change the main parameters

Change the parameters according to your experiment in the python script `dl_mss.py`.

Main parameters that might be changed are:
- `data_path`: the path containing your data folder `Experiment_X` to analyze
- `time_frame`: the time between two frames (in second)
- `pixel_size`: conversion from pixel to micrometer (μm)
- `all_states`: Add/remove states and/or change the values of the diffusion and Hurst/scaling

## Run the program

```
python dl_mss.py
```

Note that if the `data_path` parameter in `dl_mss.py` is unchanged, the program will run on the toy example.

## Reference

Arts, M., Smal, I., Paul, M. W., Wyman, C., & Meijering, E. (2019).
**Particle Mobility Analysis Using Deep Learning and the Moment Scaling Spectrum.** _Scientific Reports_, 9(1), 17160. [https://doi.org/10.1038/s41598-019-53663-8](https://doi.org/10.1038/s41598-019-53663-8).
