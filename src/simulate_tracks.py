"""Simulate fBm trajectories.

This module contains functions used to simulate fractional Brownian motion (fBm) trajectories.
Each dimension is generated with the fBm kernel (Lundahl et al. 1986) using a diffusion and alpha value.
The tracks are made of a mixture of N states following the probabilities to transition.
"""

# Third-party modules
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local functions
from src.compute_features import compute_all_features
from src.analysis import compute_ptm


def get_fbm(total_frame, alpha, diff):
    """Generates a 1D fBm using the formula from Lundahl et al. 1986.

    :param total_frame: The total number of frames to generate
    :type total_frame: int
    :param alpha: The alpha value for the simulation.
    :type alpha: float
    :param diff: The diffusion value for the simulation.
    :type diff: float
    :return: The 1-dimensional fBm generated track.
    :rtype: np.array
    """
    time_interval = np.arange(0, total_frame, 1)
    x_array, y_array = np.meshgrid(time_interval, time_interval)
    kval = np.abs(x_array - y_array)
    fbm_kernel = diff * (np.abs(kval + 1)**alpha - 2*np.abs(kval)**alpha + np.abs(kval - 1)**alpha)
    fbm = np.linalg.cholesky(fbm_kernel)
    # generate franctional noise from independent Gaussian samples
    fbm = fbm.dot(np.random.normal(0, 1, total_frame))
    return fbm

def generate_fbm_tracks(parms):
    """Creates multiple fBm trajectories with a mixture of states which will be used for training the neural network.

    :param params: Stored parameters containing instructions to generate fBm trajectories.
    :type parms: dict
    :return: The dataframe storing all generated fBm trajectories with keys: `track_id`, `frame`, `x`, `y` and `state`
    :rtype: pd.DataFrame
    """
    # List of initial states (between 0 and parms['num_states']) generated randomly
    init_state_list = np.random.randint(parms['num_states'], size=parms['num_simulate_tracks'])

    # List of track length:
    if parms['track_length_fixed'] is True:
        if parms['track_length'] < parms['length_threshold']:
            parms['track_length'] = parms['length_threshold']
        track_len_list = np.full((parms['num_simulate_tracks']), parms['track_length'])

    else:
        # Length generated randomly from an exponential distribution
        track_len_list = np.random.exponential(parms['beta'], parms['num_simulate_tracks'])
        track_len_list = track_len_list.astype(int) + parms['length_threshold']

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
                        #+ parms['length_threshold']
                    # Condition to make sure the amount of steps does not exceed the track length:
                    if total + new_steps > track_len:
                        new_steps = track_len - total
                        # if new_steps < parms['length_threshold']:
                            # state_i = track['state'][total-1]

                # The same state is attributed for each step
                track.loc[total:total+new_steps-1, 'state'] = state_i
                # The x and y coordinates are generated based on the diffusion parameters:
                alpha = parms['all_states'][state_i]['alpha']
                diff_unitless = parms['all_states'][state_i]['diff'] / parms['unit_diff']
                track.loc[total:total+new_steps-1, 'x'] = get_fbm(new_steps, alpha, diff_unitless)
                track.loc[total:total+new_steps-1, 'y'] = get_fbm(new_steps, alpha, diff_unitless)
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
    """Run the simulation by first generating fBm trajectories and then computing all features.

    :param params: Stored parameters containing instructions to generate fBm trajectories.
    :type parms: dict
    :return: The dataframe storing all generated fBm trajectories and computed features. Dataframe also saved as a csv file.
    :rtype: pd.DataFrame
    """
    print(f"\nSimulation of {parms['num_simulate_tracks']:,d} trajectories...")
    sim_df = generate_fbm_tracks(parms)
    print(compute_ptm(sim_df, parms))
    compute_all_features(sim_df)
    sim_df.to_csv(parms['simulated_track_path']/'simulated_tracks_df.csv')
    return sim_df
