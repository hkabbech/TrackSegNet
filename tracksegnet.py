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
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# Local modules
from src.track import simulate_tracks, extract_all_mdf
from src.generate_lstm_model import generate_lstm_model
from src.plot import get_color_list, get_label_list


## For GPU usage, uncomment the following lines:
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# CONFIG = ConfigProto()
# CONFIG.gpu_options.allow_growth = True
# SESSION = InteractiveSession(config=CONFIG)


if __name__ == "__main__":

    START_TIME = datetime.now()


    PAR = {
        ## Path to the trajectories to analyze:
        'data_path': Path('data/toy_example'),

        ## Parameters related to images / microscope settings:
        'time_frame': 0.007, # Time between two frames in second
        'pixel_size': 0.1, # in um (micron)
        'img_size': {'x': 450, 'y': 75}, # Image size in pixel

        # List of diffusive state:
        # Each dictionary contains parameters desribing the behaviour of tracjectories within a state
        # These motion parameters are the anomalous exponent alpha (representing the confinement) and
        # the sigma value (influence on the diffusion: sigma = np.sqrt((diffusion/unit)*2))
        'all_states': [
            ## Hurst coefficient and sigma factor:
            {'alpha': 0.2, 'sigma': 0.3, 'state': 'immobile'},
            {'alpha': 0.5, 'sigma': 0.6, 'state': 'slow'},
            {'alpha': 1, 'sigma': 1.1, 'state': 'fast'} 
        ]
    }

    # Number of states
    PAR.update({'num_states': len(PAR['all_states'])})

    PAR.update({

        ## Restrictions on the number of frames:
        'threshold': {
            'track': 3,
            'tracklet': [1 for _ in range(PAR['num_states'])]
        },

        # Number of dimensions:
        'n_dim': 2, # X and Y coordinates => 2 dimensions
        'track_length_fixed': True,
        'track_length': 27,
        ## Parameters to simulate trajectories:
        'num_simulate_tracks': 1500, # Number of tracks to simulate
        'ptm': np.zeros(PAR['num_states']) + 0.1 + (1 - PAR['num_states'] / 10) * np.identity(PAR['num_states']),
        # Probability Transition Matrix: the probability to switch to any other state is 0.1
        'min_frames':4, # Minimal amount of frames
        'min_frames_in_same_state':4, # Minimal amount of frames when particle remains in the same state
        'beta':100, # Mean of exponential distribution for length tracks
        'num_interval':10, # Parameter for retricted motion
        'motion':20, # Parameter for retricted motion

        ## Parameters to generate the LSTM model:
        'num_features': 6,
        'hidden_units': 200,
        'epochs': 1000, # Maximum number of epochs
        'patience': 15, # Parameter for the EarlyStopping function
        'batch_size': 2**5,
        'percentage': {'val': 1/3, 'test':1/4},

        # Cell folders containing tracks.simple.mdf file to analyze
        'folder_names': [PAR['data_path']/folder for folder in os.listdir(PAR['data_path'])],

        # Colors and labels for plotting:
        'colors': get_color_list(PAR['num_states']),
        'labels': get_label_list(PAR['num_states'])
    })


    STATES = f'{PAR["num_states"]}states=[' + '_'.join([f'({state["alpha"]},{state["sigma"]})' for nstate, state in enumerate(PAR['all_states'])]) + ']'

    # Paths:
    PAR.update({
        'simulated_track_path': Path(f'results/lstm_models/{PAR["num_simulate_tracks"]}_simulated_tracks_{int(PAR["time_frame"]*1000)}ms'),
        'result_path': Path(f'results/{PAR["data_path"].name}_{STATES}_{int(PAR["time_frame"]*1000)}ms'),
    }),
    PAR.update({
        'model_path': PAR["simulated_track_path"]/f'model_{STATES}',
        'plot_path': PAR["result_path"]/'analysis_plots',
    })

    os.makedirs(PAR['model_path'], exist_ok=True)
    os.makedirs(PAR['plot_path'], exist_ok=True)
    os.makedirs('data', exist_ok=True)


    ## BUILT LSTM MODEL
    ###################
    sim_csvfilename = PAR['simulated_track_path']/'simulated_tracks_df.csv'
    if not os.path.isfile(sim_csvfilename):
        sim_df = simulate_tracks(PAR)
    else:
        sim_df = pd.read_csv(sim_csvfilename)

    # if not os.path.isfile(PAR['model_path']/'best_model.h5'):
    generate_lstm_model(sim_df, PAR)
    

    ## PREDICT TRACKLET STATE
    #########################
    MODEL = load_model(PAR['model_path']/'best_model.h5', compile=False)

    track_csvfilename = PAR['result_path']/f"{PAR['data_path'].name}_tracks_df.csv"
    if not os.path.isfile(track_csvfilename):
        track_df = extract_all_mdf(PAR)
    else:
        track_df = pd.read_csv(track_csvfilename)

    predict_states(track_df, MODEL)



    ## ANALYSIS
    ###########

    # DATA.create_tracklet_lists(PAR)
    # DATA.compute_mss_analysis(PAR)
    # DATA.count_tracklets_per_mdf(PAR['result_path'], PAR)
    # DATA.compute_ptm(PAR['result_path'], PAR)
    # with open(PAR['result_path']/'TrackletAnalysis.p', 'wb') as file:
    #     pickle.dump(DATA, file)

