"""Compute track features.

This module contains functions computing different features which will be used as input of the deep-learning network (for both training and testing).
"""

# Third-party modules
import numpy as np
from tqdm import tqdm


def compute_displacements(track, delta):
    """Computes x and y displacements between each pair of points i and i+delta.

    :param track: Given track where displacements will be computed along its entire length.
    :type track: pd.DataFrame
    :param delta: Lag interval between frames.
    :type delta: int
    :return: Computed (x displacements, y displacements) along the track.
    :rtype: (np.array, np.array)
    """
    displ_x = track[delta:].reset_index(drop=True)['x']-track.reset_index(drop=True)['x']
    displ_y = track[delta:].reset_index(drop=True)['y']-track.reset_index(drop=True)['y']
    return displ_x.to_numpy(), displ_y.to_numpy()

def compute_dist(displ_x, displ_y):
    """Computes distances based on x and y displacements.

    :param displ_x: x displacements of a given track.
    :type displ_x: np.array
    :param displ_y: y displacements of a given track.
    :type displ_y: np.array
    :return: Computed distances along the track.
    :rtype: np.array
    """
    return np.sqrt(displ_x**2 + displ_y**2)

def compute_mean_distances(track, delta, num=1):
    """Computes a mean of distances from point i-num to point i+num.

    :param track: Given track where mean of distances will be computed along its entire length.
    :type track: pd.DataFrame
    :param delta: Lag interval between frames while measuring the displacements.
    :type delta: int
    :param num: Number of neighboring point for the average. [Defaults: +/- 1 close neighbors]
    :type num: int
    :return: The mean of distances.
    :rtype: np.array
    """
    displ_x, displ_y = compute_displacements(track, delta)
    dist = compute_dist(displ_x, displ_y)
    num_frames = len(displ_x)
    mean_dist = []
    for i in range(num_frames):
        start = i - num
        start = max(start, 0)
        end = i + num
        if end > num_frames - num - delta:
            end = num_frames - num - delta
        # print(dist[start:end+1], end='\n\n')
        if len(dist[start:end+1]) < 1:
            mean_dist.append(np.nan)
        else:
            mean_dist.append(np.mean(dist[start:end+1]))
    return mean_dist

def compute_angles(displ_x, displ_y):
    """Computes angles using two consecutive displacements using x and y 1D-displacements.

    :param displ_x: x displacements of a given track.
    :type displ_x: np.array
    :param displ_y: y displacements of a given track.
    :type displ_y: np.array
    :return: Angles computed along the track.
    :rtype: np.array
    """
    alpha_1 = np.arctan2(displ_y[:-1], displ_x[:-1])
    alpha_2 = np.arctan2(displ_y[1:], displ_x[1:]) + 2*np.pi
    angles = (alpha_2 - alpha_1) % (2*np.pi)
    angles[angles > np.pi] = angles[angles > np.pi] - (2*np.pi)
    angles = np.insert(angles, 0, np.nan)
    return angles

def compute_all_features(track_df):
    """Computes all features necessary for the inputs of the deep-learning network.
    The features are directly added to the dataframe of trajectories.

    :param track_df: All trajectories as a dataframe with keys: `x`, `y`, `frame`, `data_folder`, `track_id`.
    :type track_df: pd.DataFrame
    """
    print('\nCompute features...')
    track_df[['displ_x', 'displ_y', 'dist', 'mean_dist_1', 'mean_dist_2', 'angle']] = np.nan
    for track_id in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == track_id]
        displ_x, displ_y = compute_displacements(track, delta=1)
        track_df.loc[track.index, 'displ_x'] = displ_x
        track_df.loc[track.index, 'displ_y'] = displ_y
        track_df.loc[track.index, 'dist'] = compute_dist(displ_x, displ_y)
        track_df.loc[track.index, 'mean_dist_1'] = compute_mean_distances(track, delta=1)
        track_df.loc[track.index, 'mean_dist_2'] = compute_mean_distances(track, delta=2)
        track_df.loc[track.index, 'angle'] = compute_angles(displ_x, displ_y)
