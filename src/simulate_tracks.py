"""Simulate fBm trajectories.

This module contains a few functions used to simulate fractional Brownian motion (fBm) trajectories.
The fBm kernel (Lundahl et al. 1986) is used to generate each dimension using a given alpha and
sigma values (the motion parameters). The tracks are made of a mixture of N states.
"""

# Third-party modules
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local functions
from src.compute_features import compute_all_features
from src.analysis import compute_ptm


def get_fbm(total_frame, alpha, sigma):
    """Generates a 1D fBm using the formula from Lundahl et al. 1986."""
    time_interval = np.arange(0, total_frame, 1)
    x_array, y_array = np.meshgrid(time_interval, time_interval)
    kval = np.abs(x_array - y_array)
    diff = sigma**2 / 2.0
    fbm_kernel = diff * (np.abs(kval + 1)**alpha - 2*np.abs(kval)**alpha + np.abs(kval - 1)**alpha)
    fbm = np.linalg.cholesky(fbm_kernel)
    # generate franctional noise from independent Gaussian samples
    fbm = fbm.dot(np.random.normal(0, 1, total_frame))
    return fbm

def generate_fbm_tracks(parms):
    """Creates fBm trajectories and stores tham in a pandas DataFrame."""
    # List of initial states (between 0 and parms['num_states']) generated randomly
    init_state_list = np.random.randint(parms['num_states'], size=parms['num_simulate_tracks'])

    # List of track length:
    if parms['track_length_fixed'] is True:
        if parms['track_length'] < parms['min_frames']:
            parms['track_length'] = parms['min_frames']
        track_len_list = np.full((parms['num_simulate_tracks']), parms['track_length'])

    else:
        # Length generated randomly from an exponential distribution
        track_len_list = np.random.exponential(parms['beta'], parms['num_simulate_tracks'])
        track_len_list = track_len_list.astype(int) + parms['min_frames']

    track_df = None
    track_id = 0
    with tqdm(total=parms['num_simulate_tracks']) as pbar:
        for track_len, state_i in zip(track_len_list, init_state_list):
            # Initialization of the new track:
            track = pd.DataFrame({
                'track_id': track_id,
                'frame': range(track_len),
                'x': 0,
                'y': 0,
                'state': 0,
            })
            total = 0
            # The x, y and state values are generated for the whole track length
            while total < track_len:
                # The amount of steps is generated
                if parms['num_states'] == 1:
                    new_steps = track_len - total
                else:
                    new_steps = int(np.random.geometric(1-parms['ptm'][state_i][state_i]))
                        #+ parms['min_frames']
                    # Condition to make sure the amount of steps does not exceed the track length:
                    if total + new_steps > track_len:
                        new_steps = track_len - total
                        # if new_steps < parms['min_frames']:
                            # state_i = track['state'][total-1]

                # The same state is attributed for each step
                track.loc[total:total+new_steps-1, 'state'] = state_i
                # The x and y coordinates are generated based on the diffusion parameters:
                alpha = parms['all_states'][state_i]['alpha']
                sigma = parms['all_states'][state_i]['sigma']
                track.loc[total:total+new_steps-1, 'x'] = get_fbm(new_steps, alpha, sigma)
                track.loc[total:total+new_steps-1, 'y'] = get_fbm(new_steps, alpha, sigma)
                # Update of the current state and the amount of total steps
                if parms['num_states'] > 1:
                    state_i = np.delete(np.arange(parms['num_states']), state_i)\
                        [np.random.randint(parms['num_states']-1)]
                total += new_steps
            track['x'] = np.cumsum(track['x'])
            track['y'] = np.cumsum(track['y'])
            if track_df is None:
                track_df = track
            else:
                track_df = pd.concat((track_df, track))
            pbar.update(1)
            track_id += 1
            # break
    track_df = track_df.reset_index(drop=True)
    return track_df

def run_track_simulation(parms):
    """Simulation of fBm trajectories."""
    print(f"\nSimulation of {parms['num_simulate_tracks']:,d} trajectories...")
    sim_df = generate_fbm_tracks(parms)
    print(compute_ptm(sim_df, parms))
    compute_all_features(sim_df)
    sim_df.to_csv(parms['simulated_track_path']/'simulated_tracks_df.csv')
    return sim_df
