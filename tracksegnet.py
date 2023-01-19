#!/usr/bin/env python3
# coding: utf-8

"""
    Segmentation of trajectories into N tracklet states using LSTM neural network classifier

    Usage:
        # Change the parameters in the main before running the following command:
        ./dl_mss.py
"""

# Third-party modules
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Local modules
from src.simulate_tracks import run_track_simulation
from src.experimental_tracks import extract_all_mdf, predict_states
from src.generate_lstm_model import generate_lstm_model
from src.analysis import compute_ptm, make_tracklet_lists, compute_all_msd,\
    plot_scatter_alpha_diffusion, plot_proportion

## For GPU usage, uncomment the following lines:
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# CONFIG = ConfigProto()
# CONFIG.gpu_options.allow_growth = True
# SESSION = InteractiveSession(config=CONFIG)


def get_color_list(num_states):
    """Create list of colors"""
    if num_states == 1:
        colors = ['k']
    if num_states == 2:
        colors = ['darkblue', 'red']
    elif num_states == 3:
        colors = ['darkblue', 'darkorange', 'red']
    elif num_states == 4:
        colors = ['darkblue', 'darkorange', 'red', 'green']
    elif num_states == 5:
        colors = ['darkblue', 'darkorange', 'red', 'green', 'darkviolet']
    else:
        print('Please, select 5 states or less.')
    return colors

if __name__ == "__main__":

    START_TIME = datetime.now()

    PARMS_FILENAME = sys.argv[0]
    PARMS_FILENAME = 'parms_THZ1_inhibitor.tsv'
    # PARMS_FILENAME = 'parms_UV.tsv'
    PARMS_DF = pd.read_csv(PARMS_FILENAME, sep='\t').set_index('parms').squeeze().to_dict()

    NSTATES = int(PARMS_DF['num_states'])
    PARMS = {
        ## Path to the trajectories to analyze:
        'data_path': Path(PARMS_DF['data_path']),

        ## Microscope settings:
        'time_frame': float(PARMS_DF['time_frame']), # Time between two frames in second
        'pixel_size': float(PARMS_DF['pixel_size']), # in micron

        ## Diffusive states:
        'num_states': NSTATES,
        # List of diffusive state:
        # motion parameters describing the behaviour of tracjectories within a state:
        # (1) the anomalous exponent alpha (expressing the confinement)
        # (2) the sigma value (related to the diffusion: sigma = np.sqrt((diffusion/unit)*2))
        'all_states': [{
            'alpha': float(PARMS_DF[f'state_{state}_alpha']),
            'sigma': float(PARMS_DF[f'state_{state}_sigma'])
            } for state in range(1, NSTATES+1)],

        ## Restrictions on the track length:
        'length_threshold': 6,

        ## Trajectory simulation:
        'n_dim': 2, # X and Y coordinates => 2 dimensions
        'track_length_fixed': True,
        'track_length': 27,
        'num_simulate_tracks': 10000,
        # transition probabilities
        'ptm': np.zeros(NSTATES) + 0.1 + (1 - NSTATES / 10) * np.identity(NSTATES),
        'min_frames':4, # Minimal amount of frames
        'min_frames_in_same_state':4, # Minimal amount of frames for particles remaining in a state
        'beta':100, # Mean of exponential distribution for length tracks
        'num_interval':10, # Parameter for retricted motion
        'motion':20, # Parameter for retricted motion

        ## Parameters to generate the LSTM model:
        'num_features': 6,
        'hidden_units': 200,
        'epochs': 1000, # Maximum number of epochs
        'patience': 15, # Parameter for the EarlyStopping function
        'batch_size': 2**5,
        'percent': {'val': 1/3, 'test':1/4},

        # Cell folders containing tracks.simple.mdf file to analyze
        'folder_names': [Path(f"{PARMS_DF['data_path']}/{folder}")\
                         for folder in os.listdir(PARMS_DF['data_path'])],

        # Colors and labels for plotting:
        'colors': get_color_list(NSTATES),

        # Figure saving:
        'fig_format': 'png',
        'fig_dpi': 200,
        'fig_transparent': False,
    }


    STATES = f"{PARMS['num_states']}states=[" + '_'.join([f"({state['alpha']},{state['sigma']})"\
        for _, state in enumerate(PARMS['all_states'])]) + "]"

    # Paths:
    PARMS.update({
        'simulated_track_path': Path(f"results/lstm_models/{PARMS['num_simulate_tracks']}_simulated_tracks"),
        'result_path': Path(f"results/{PARMS['data_path'].name}_{int(PARMS['time_frame']*1000)}ms_{STATES}"),
    })
    PARMS.update({
        'model_path': PARMS['simulated_track_path']/f'model_{STATES}',
    })

    os.makedirs(PARMS['model_path'], exist_ok=True)
    os.makedirs('data', exist_ok=True)

    ## BUILD LSTM MODEL
    ###################
    SIM_CSV = PARMS['simulated_track_path']/'simulated_tracks_df.csv'
    if not os.path.isfile(SIM_CSV):
        SIM_DF = run_track_simulation(PARMS)
    else:
        print('\nLoad simulated trajectories...')
        SIM_DF = pd.read_csv(SIM_CSV)

    if not os.path.isfile(PARMS['model_path']/'best_model.h5'):
        generate_lstm_model(SIM_DF, PARMS)

    ## PREDICT TRACKLET STATE
    #########################
    print('\nLoad trained model...')
    MODEL = load_model(PARMS['model_path']/'best_model.h5', compile=False)

    TRACK_CSV = PARMS['result_path']/f"{PARMS['data_path'].name}_tracks_df.csv"
    if not os.path.isfile(TRACK_CSV):
        TRACK_DF = extract_all_mdf(PARMS)
    else:
        print('\nLoad experimental trajectories...')
        TRACK_DF = pd.read_csv(TRACK_CSV)

    if 'state' not in TRACK_DF:
        predict_states(TRACK_DF, MODEL, PARMS)

    ## ANALYSIS
    ###########
    print(compute_ptm(TRACK_DF, PARMS))
    TRACKLET_LISTS = make_tracklet_lists(TRACK_DF, PARMS)
    MOTION_PARMS = compute_all_msd(TRACKLET_LISTS, PARMS)
    plot_scatter_alpha_diffusion(MOTION_PARMS, PARMS)
    # TODO in scatterplot: markersize and distribution based on the number of data points!
    plot_proportion(TRACKLET_LISTS, PARMS)

