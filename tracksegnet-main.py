#!/usr/bin/env python
# coding: utf-8

"""
    Segmentation of trajectories into N diffusive states using LSTM neural network classifier

    Usage:
        # Adjust parameters in parms.csv before running the following command on the terminal:
        python tracksegnet-main.py parms.csv
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
from tracksegnet.utils import check_parms, get_color_list
from tracksegnet.simulate_tracks import run_track_simulation
from tracksegnet.experimental_tracks import extract_all_tracks, predict_states
from tracksegnet.generate_lstm_model import generate_lstm_model
from tracksegnet.analysis import compute_ptm, make_tracklet_lists, compute_all_msd,\
    plot_scatter_alpha_diffusion, plot_proportion, plot_displ, plot_angles, plot_vac

if __name__ == "__main__":

    START_TIME = datetime.now()

    PARMS_FILENAME = sys.argv[-1]
    # PARMS_FILENAME = 'parms.csv'
    PARMS_DF = pd.read_csv(PARMS_FILENAME, sep='\t').set_index('parms').squeeze().to_dict()
    PARMS_DF = check_parms(PARMS_DF)
    NSTATES = PARMS_DF['num_states']
    PARMS = {
        ## Path to the trajectories to analyze:
        'data_path': Path(PARMS_DF['data_path']),
        'track_format': 'MDF',

        ## Microscope settings:
        'time_frame': PARMS_DF['time_frame'], # Time between two frames in second
        'pixel_size': PARMS_DF['pixel_size'], # in micron

        ## Diffusive states:
        'num_states': NSTATES,
        # List of diffusive state:
        # motion parameters describing the behaviour of tracjectories within a state:
        # (1) the anomalous exponent alpha (expressing the confinement)
        # (2) the sigma value (related to the diffusion: sigma = np.sqrt((diffusion/unit)*2))
        'all_states': [{
            'diff': PARMS_DF[f'state_{state}_diff'],
            'alpha': PARMS_DF[f'state_{state}_alpha']
        } for state in range(1, NSTATES+1)],

        ## Restrictions on the track length:
        'length_threshold': 14,

        ## Trajectory simulation:
        'track_length_fixed': True,
        'track_length': 27,
        'num_simulate_tracks': 10000,
        # transition probabilities
        'ptm': np.array([[PARMS_DF[f'pt_{state1}_{state2}'] for state1 in range(1, NSTATES+1)]\
                        for state2 in range(1, NSTATES+1)]),
        'beta':100, # Mean of exponential distribution for length tracks

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

    PARMS['unit_diff'] = (PARMS['pixel_size']**2)/PARMS['time_frame'] # in um**2/2
    STATES = f"{PARMS['num_states']}states=[" + '_'.join([f"({state['alpha']},{state['diff']})"\
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
    os.makedirs(PARMS['result_path'], exist_ok=True)

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
        TRACK_DF = extract_all_tracks(PARMS)
    else:
        print('\nLoad experimental trajectories...')
        TRACK_DF = pd.read_csv(TRACK_CSV)

    if 'state' not in TRACK_DF:
        predict_states(TRACK_DF, MODEL, PARMS)

    ## ANALYSIS
    ###########
    PTM_CSV = PARMS['result_path']/'transition_probabilities.csv'
    if not os.path.isfile(PTM_CSV):
        PTM = compute_ptm(TRACK_DF, PARMS)
    else:
        PTM = pd.read_csv(PTM_CSV)
    print(PTM)

    if not os.path.isfile(PARMS['result_path']/"motion_parms_state_1.csv"):
        TRACKLET_LISTS = make_tracklet_lists(TRACK_DF, PARMS)
        MOTION_PARMS = compute_all_msd(TRACKLET_LISTS, PARMS)
    else:
        TRACKLET_LISTS = make_tracklet_lists(TRACK_DF, PARMS)
        MOTION_PARMS = [pd.read_csv(PARMS['result_path']/f"motion_parms_state_{state+1}.csv") for state in range(PARMS['num_states'])]

    plot_proportion(TRACKLET_LISTS, PARMS)
    plot_scatter_alpha_diffusion(MOTION_PARMS, PARMS)
    plot_displ(TRACKLET_LISTS, PARMS)
    plot_angles(TRACKLET_LISTS, PARMS)
    plot_vac(TRACKLET_LISTS, PARMS)
