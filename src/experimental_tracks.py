"""Extract experimental trajectories.

This module contains functions used for the extraction of real experimental trajectories (stored as
MDF or CSV file format).
"""

# Third-party modules
import os
from re import findall, split
from random import uniform
# from multiprocessing import Pool, cpu_count
# from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
from more_itertools import consecutive_groups

# Local functions
from src.compute_features import compute_all_features


def fill_gaps_l1(track):
    """Fills the 1-length gaps with an intermediate point having an additional bias (to add randomness).

    :param track: Given track which its 1-length gaps will be filled with an appropriate generated point.
    :type track: pd.DataFrame
    :return: (The updated track with 1-length gaps filled, the number of gaps filled for this given track)
    :rtype: (pd.DataFrame, int)
    """
    # get gaps:
    start = int(track['frame'].iloc[0])
    end = int(track['frame'].iloc[-1])
    missing_frames = sorted(set(range(start, end + 1)).difference(track['frame']))
    gaps = [list(group) for group in consecutive_groups(missing_frames)]
    num_gaps_l1 = 0
    for _, gap in enumerate(gaps):
        if len(gap) != 1:
            continue
        num_gaps_l1 += 1
        missing = gap[0]
        prev_point = track[track['frame'] == missing-1].iloc[0].to_dict()
        next_point = track[track['frame'] == missing+1].iloc[0].to_dict()
        average_points = {'x': np.mean([prev_point['x'], next_point['x']]),
                          'y': np.mean([prev_point['y'], next_point['y']])}
        displ = {'x': abs(prev_point['x'] - next_point['x']),
                 'y': abs(prev_point['y'] - next_point['y'])}
        bias = {'x': uniform(-displ['x']/4, displ['x']/4),
                'y': uniform(-displ['y']/4, displ['y']/4)}
        if displ['x'] == 0 and displ['y'] == 0:
            new_point = pd.DataFrame({
                'index': 'X',
                'x': average_points['x'],
                'y': average_points['y'],
                'frame': np.array([missing]),
                'data folder': track.iloc[0]['data folder'],
                'track_id': track.iloc[0]['track_id']
            })
        else:
            if displ['x'] != 0:
                slop = (average_points['y'] - prev_point['y'])\
                      / (average_points['x'] - next_point['x'])
            else:
                slop = (average_points['y'] - average_points['y'])\
                      / (average_points['x'] - average_points['x']+1)
            new_point = pd.DataFrame({
                'index': 'X',
                'x': average_points['x'] - bias['x'],
                'y': average_points['y'] - bias['x']*slop - bias['y'],
                'frame': np.array([missing]),
                'data folder': track.iloc[0]['data folder'],
                'track_id': track.iloc[0]['track_id']
            })
        track = pd.concat(
            (track[track['frame'] < missing], new_point, track[track['frame'] > missing])
        )
    return track, num_gaps_l1

def fill_gaps(parms, track_df):
    """Fill the gap in trajectory frames by taking an intermediate point and splits the trajectory into two separate tracks for gaps larger than 1 frame.

    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param track_df: Dataframe containing all extracted trajectories with keys: `x`, `y`, `frame`, `data_folder`, `track_id`.
    :type track_df: pd.DataFrame
    :return: The updated dataframe with filled gaps or splitted tracks.
    :rtype: pd.DataFrame
    """
    num_tracks = len(track_df['track_id'].unique())
    print(f'\nCheck and fill gaps in {num_tracks:,d} trajectories...')
    track_df_2 = None
    total_gaps_l1, total_gaps_ln = 0, 0
    for track_id in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == track_id]
        # fill gaps with one point coord missing:
        new_track, num_gaps_l1 = fill_gaps_l1(track)
        new_track = new_track.reset_index(drop=True)
        total_gaps_l1 += num_gaps_l1
        # The Track is splitted for gaps of length-2 or more:
        sta = new_track.iloc[0].name
        end = new_track.iloc[-1].name
        indexes = new_track[new_track['frame'].diff(periods=-1) < -1].index.to_list()
        total_gaps_ln += len(indexes)
        indexes.append(end)
        for i, ind in enumerate(indexes):
            # new 'track_id':
            new_track.loc[sta:ind, 'track_id'] = split('_0$', new_track.iloc[0]['track_id'])[0]+f'_{str(i)}'
            sta = ind+1
        if track_df_2 is None:
            track_df_2 = new_track
        else:
            track_df_2 = pd.concat((track_df_2, new_track))
    # Keep only tracks longer than parms['length_threshold']:
    print(f'\nKeep trajectories longer than {parms["length_threshold"]}..')
    track_df_3 = None
    for track_id in tqdm(track_df_2['track_id'].unique()):
        track = track_df_2[track_df_2['track_id'] == track_id]
        if len(track) >= parms['length_threshold']:
            if track_df_3 is None:
                track_df_3 = track
            else:
                track_df_3 = pd.concat((track_df_3, track))
        else:
            total_gaps_ln -= 1
    print(f"{total_gaps_l1:,d} length-1 gaps were filled in", end=' ')
    print(f"and {total_gaps_ln:,d} tracks created due to length-2+ gaps.")
    return track_df_3.reset_index(drop=True)

def extract_1_mdf(folder_name, parms, filename='tracks.simple.mdf'):
    """Extracts experimental trajectories from a single MDF file and returns a pandas DataFrame which stores
    the trajectories.

    :param folder_name: Name of the folder containing the trajectory file.
    :type folder_name: str
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param filename: File which stores trajectory coordinates extracted from the tracking.
                     `[Defaults: 'tracks.simple.mdf']`
    :type filename: str
    :return: All extracted trajectories as a dataframe with keys: `x`, `y`, `frame`, `data_folder`, `track_id`.
    :rtype: pd.DataFrame
    """    
    if not os.path.isfile(folder_name/filename):
        for filename in os.listdir(folder_name):
            if filename.endswith(".mdf"):
                break
    # Initialization
    index = 0
    track_df = None
    # coordinates of the current track:
    current_coord = {'frame': np.array([]), 'x': np.array([]), 'y': np.array([])}
    with open(folder_name/filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            # Find a new Track
            if line[:5] == 'Track':
                # If old Track longer than the threshold then it is added to the list
                if len(current_coord['x']) >= parms['length_threshold']:
                    track = pd.DataFrame(current_coord)
                    track['data folder'] = str(folder_name) # PosixPath to str
                    track['track_id'] = folder_name.name.split('_')[-1] + '_' + index
                    if track_df is None:
                        track_df = track
                    else:
                        track_df = pd.concat((track_df, track))
                # Initialize new Track
                index = findall(r'\d+', line)[0]
                current_coord = {'x': np.array([]), 'y': np.array([]), 'frame': np.array([])}
            # Add Point to the current Track
            elif line[:5] == 'Point':
                contents = line.split(' ')
                current_coord['x'] = np.append(current_coord['x'], float(contents[2]))
                current_coord['y'] = np.append(current_coord['y'], float(contents[3]))
                current_coord['frame'] = np.append(current_coord['frame'], int(contents[5]))
            # End of the file, the last Track is added
            elif line[:3] == 'End':
                # If old Track longer than the threshold then it is added to the list
                if len(current_coord['x']) >= parms['length_threshold']:
                    track = pd.DataFrame(current_coord)
                    track['data folder'] = str(folder_name) # PosixPath to str
                    track['track_id'] = folder_name.name.split('_')[-1] + '_' + index
                    if track_df is None:
                        track_df = track
                    else:
                        track_df = pd.concat((track_df, track))
    return track_df

def extract_1_csv(folder_name, parms, filename='tracks.csv'):
    """Extracts experimental trajectories from a single CSV file and returns a pandas DataFrame which stores
    the trajectories.

    :param folder_name: Name of the folder containing the trajectory file.
    :type folder_name: str
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param filename: File which stores trajectory coordinates extracted from the tracking.
                     `[Defaults: 'tracks.csv']`
    :type filename: str
    :return: All extracted trajectories as a dataframe with keys: `x`, `y`, `frame`, `data_folder`, `track_id`.
    :rtype: pd.DataFrame
    """ 
    if not os.path.isfile(folder_name/filename):
        for filename in os.listdir(folder_name):
            if filename.endswith(".csv"):
                break
    tracks = pd.read_csv(filename)
    tracks['data folder'] = str(folder_name)

    track_df = None
    for track_id in tracks['track_id'].unique():
        track = tracks[tracks['track_id'] == track_id]
        if len(track['x']) >= parms['length_threshold']:
            track['track_id'] = folder_name.name.split('_')[-1] + '_' + track_id
            if track_df is None:
                track_df = track
            else:
                track_df = pd.concat((track_df, track))
    return track_df

def extract_all_tracks(parms):
    """Extracts the trajectories from several MDF (or CSV) files, fills the gaps and return a unique pandas
    DataFrame containing tracks.

    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: All extracted trajectories as a dataframe with keys: `x`, `y`, `frame`, `data_folder`, `track_id`.
    :rtype: pd.DataFrame
    """
    num_files = len(parms['folder_names'])
    print(f"\nExtraction of tracjectories from {num_files} mdf files...")
    all_track_df = []
    if parms['track_format'] == 'MDF':
        for folder_name in tqdm(parms['folder_names']):
            all_track_df.append(extract_1_mdf(folder_name, parms))
    elif parms['track_format'] == 'CSV':
        for folder_name in tqdm(parms['folder_names']):
            all_track_df.append(extract_1_csv(folder_name, parms))
    track_df = pd.concat(all_track_df).reset_index()
    print(f'\n{track_df.shape[0]} coordinate points')
    print(track_df.head())
    track_df_2 = fill_gaps(parms, track_df)
    print(f'\n{track_df_2.shape[0]} coordinate points')
    print(track_df_2.head())
    compute_all_features(track_df_2)
    print(track_df_2.head())
    track_df_2.to_csv(parms['result_path']/f"{parms['data_path'].name}_tracks_df.csv", index=False)
    return track_df_2

def predict_states(track_df, model, parms):
    """Predicts the states for each trajectory using a trained model and saves the resulting state prediction as a csv file.

    :param track_df: Name of the folder containing the trajectory file.
    :type track_df: pd.DataFrame
    :param model: Trained neural network.
    :type model: keras model
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    """
    num_tracks = len(track_df['track_id'].unique())
    print(f'\nState prediction of {num_tracks:,d} trajectories...')
    for track_id in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == track_id]
        features = track[['displ_x', 'displ_y', 'dist', 'mean_dist_1', 'mean_dist_2', 'angle']]
        features = features.to_numpy()[1:-1]
        features = features.reshape(1, features.shape[0], features.shape[-1])
        predicted_states = model.predict(features, verbose=0)
        predicted_states = [np.nan] + predicted_states.argmax(axis=2)[0].tolist() + [np.nan]
        track_df.loc[track.index, 'state'] = predicted_states
    print('')
    print(track_df.head())
    track_df.to_csv(parms['result_path']/f"{parms['data_path'].name}_tracks_df.csv", index=False)
