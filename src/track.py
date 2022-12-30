import numpy as np
import pandas as pd
from tqdm import tqdm

## FEATURES
###########

def compute_displacements(track, order):
    """Compute x and y displacements between point i and point i+order"""
    displ_x = track[order:].reset_index()['x']-track.reset_index()['x']
    displ_y = track[order:].reset_index()['y']-track.reset_index()['y']
    return displ_x.to_numpy(), displ_y.to_numpy()

def compute_dist(displ_x, displ_y):
    return np.sqrt(displ_x**2 + displ_y**2)

def compute_mean_distances(track, order, point=1):
    """Compute a mean of distances (having a define order) between point i-point, point i and
    point i+point"""
    displ_x, displ_y = compute_displacements(track, order)
    dist = compute_dist(displ_x, displ_y)
    num_frames = len(displ_x)
    mean_dist = []
    for i in range(num_frames):
        start = i - point
        if start < 0:
            start = 0
        end = i + point
        if end > num_frames - point - order:
            end = num_frames - point - order
        # print(dist[start:end+1], end='\n\n')
        if len(dist[start:end+1]) < 1:
            mean_dist.append(np.nan)
        else:
            mean_dist.append(np.mean(dist[start:end+1]))
    return mean_dist

def compute_angles(displ_x, displ_y):
    alpha_1 = np.arctan2(displ_y[:-1], displ_x[:-1])
    alpha_2 = np.arctan2(displ_y[1:], displ_x[1:]) + 2*np.pi
    angles = (alpha_2 - alpha_1) % (2*np.pi)  
    angles[angles>np.pi] = angles[angles>np.pi] - (2*np.pi)
    angles = np.insert(angles, 0, np.nan)
    return angles

def compute_all_features(df):
    print('\nCompute features...')
    df[['displ_x', 'displ_y', 'dist', 'mean_dist_1', 'mean_dist_2', 'angle']] = np.nan
    for N in tqdm(df['track_id'].unique()):
        track = df[df['track_id'] == N]
        displ_x, displ_y = compute_displacements(track, order=1)
        df.loc[track.index, 'displ_x'] = displ_x
        df.loc[track.index, 'displ_y'] = displ_y
        df.loc[track.index, 'dist'] = compute_dist(displ_x, displ_y)
        df.loc[track.index, 'mean_dist_1'] = compute_mean_distances(track, order=1)
        df.loc[track.index, 'mean_dist_2'] = compute_mean_distances(track, order=2)
        df.loc[track.index, 'angle'] = compute_angles(displ_x, displ_y)


## SIMULATE FBM TRACKS
######################

def getCovarMatrix(T, alpha, sigma):
    ti = np.arange(0, T, 1)
    xx, yy = np.meshgrid(ti, ti)
    k = np.abs(xx - yy)
    return (sigma**2 / 2.0) * (np.abs(k + 1)**alpha + np.abs(k - 1)**alpha - 2 * np.abs(k)**alpha)
    # return (sigma) * (np.abs(k + 1)** (2 * H) + np.abs(k - 1)** (2 * H) - 2 * np.abs(k)** (2 * H))
    # return (np.abs(k + 1)** (2 * H) + np.abs(k - 1)** (2 * H) - 2 * np.abs(k)** (2 * H))

def getfBn(T, alpha, sigma):
    R = getCovarMatrix(T, alpha, sigma)
    L = np.linalg.cholesky(R)
    # generate franctional noise from independent Gaussian samples 
    return L.dot(np.random.normal(0, 1, T))

def generate_tracks(par):
    """
        Create a list of trajectories
    """
    # List of initial states (between 0 and par['num_states']) generated randomly
    init_state_list = np.random.randint(par['num_states'], size=par['num_simulate_tracks'])

    # List of track length:
    if par['track_length_fixed'] is True:
        if par['track_length'] < par['min_frames']:
            par['track_length'] = par['min_frames']
        track_len_list = np.full((par['num_simulate_tracks']), par['track_length'])

    else:
        # Length generated randomly from exponential distribution
        track_len_list = np.random.exponential(par['beta'], par['num_simulate_tracks'])
        track_len_list = track_len_list.astype(int) + par['min_frames']


    track_df = None
    track_id = 0
    with tqdm(total=par['num_simulate_tracks']) as pbar:
        for track_len, state_i in zip(track_len_list, init_state_list):
            # Initialization of the new track:
            track = pd.DataFrame({
                'track_id': track_id,
                'frame': range(track_len),
                'x': 0,
                'y': 0,
                'state': 0,
            })

            total_steps = 0
            # The x, y and state values are generated for the whole length of the track
            while total_steps < track_len:
                # The amount of steps is generated
                if par['num_states'] == 1:
                    new_steps = track_len - total_steps
                else:
                    new_steps = int(np.random.geometric(1-par['ptm'][state_i][state_i])) #+ par['min_frames']
                    # Condition to make sure the amount of steps does not exceed the track length:
                    if total_steps + new_steps > track_len:
                        new_steps = track_len - total_steps
                        # if new_steps < par['min_frames']:
                            # state_i = track['state'][total_steps-1]

                # The same state is attributed for each step
                track.loc[total_steps:total_steps+new_steps-1, 'state'] = state_i
                # The x and y coordinates are generated depending on the type of diffusion
                alpha = par['all_states'][state_i]['alpha']
                sigma = par['all_states'][state_i]['sigma']
                track.loc[total_steps:total_steps+new_steps-1, 'x'] = getfBn(T=new_steps, alpha=alpha, sigma=sigma)
                track.loc[total_steps:total_steps+new_steps-1, 'y'] = getfBn(T=new_steps, alpha=alpha, sigma=sigma)
                # print(track)
                # print(state_i, new_steps)
                # Update of the current state and the amount of steps total_steps
                if par['num_states'] > 1:
                    state_i = np.delete(np.arange(par['num_states']), state_i)[np.random.randint(par['num_states']-1)]
                total_steps += new_steps
            track['x'] = np.cumsum(track['x'])
            track['y'] = np.cumsum(track['y'])
            if track_df is None:
                track_df = track
            else:
                track_df = pd.concat((track_df, track))
            pbar.update(1)
            track_id += 1
            # break
    track_df = track_df.reset_index()
    return track_df

def simulate_tracks(par):
    """Simulation of N trajectories (Track objects)"""

    # plot_path = par['simulated_track_path']/'plots'
    # os.makedirs(plot_path, exist_ok=True)

    print('\nSimulation of {:,d} trajectories...'.format(par['num_simulate_tracks']))
    sim_df = generate_tracks(par)
    print(compute_ptm(sim_df, par))
    compute_all_features(sim_df)
    sim_df.to_csv(par['simulated_track_path']/'simulated_tracks_df.csv')
    return sim_df


## EXTRACT REAL TRACKS
######################

def extract_all_mdf(par):
    """Extract trajectories from one or more MDF files and create a unique list of Track objects

    Parameters
    ----------
    par: dict
        Main parameters stored in a dictionary.
        The parameter `folder_names` is used get the list of folder names used to extract the
        MDF files
    """
    num_files = len(par['folder_names'])
    print("\nExtraction of tracjectories from {} mdf files...".format(num_files))
    func = partial(extract_1_mdf, par)
    with Pool(cpu_count()) as pool:
        all_track_df = [track for track in tqdm(pool.imap_unordered(func, par['folder_names']),
                                                                    total=num_files)]
    track_df = pd.concat(all_track_df).reset_index()
    fill_gaps(track_df, par)
    compute_all_features(track_df)
    track_df.to_csv(par['result_path']/f"{par['data_path'].name}_tracks_df.csv")

    return track_df

def extract_1_mdf(par, folder_name):
    """Extract trajectories from one mdf file and return a list of Track objects."""
    # Find mdf file to analyze
    filename = 'tracks.simple.mdf'
    if not os.path.isfile(folder_name/filename):
        for filename in os.listdir(folder_name):
            if filename.endswith(".mdf"):
                break
    # Initialization
    index = 0
    track_df = None
    # coordinates of the current track:
    current_coord = {'frame': np.array([]), 'x': np.array([]), 'y': np.array([])}
    with open(folder_name/filename, 'r') as file:
        for line in file.readlines():
            # Find a new Track
            if line[:5] == 'Track':
                # If old Track longer than the threshold then it is added to the list
                if len(current_coord['x']) >= par['threshold']['track']:
                    track = pd.DataFrame(current_coord)
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
                if len(current_coord['x']) >= par['threshold']['track']:
                    track = pd.DataFrame(current_coord)
                    track['track_id'] = folder_name.name.split('_')[-1] + '_' + index
                    if track_df is None:
                        track_df = track
                    else:
                        track_df = pd.concat((track_df, track))
    return track_df

def fill_gaps(par):
    """While one frame is missing in `Track.table['frame']`, the gap is filled in with a random
    value. While more than one frame is missing continuously in a Track, then the Track is
    truncated in two Tracks"""
    num_tracks = len(track_df['track_id'].unique())
    print(f'\nCheck and fill gaps in {num_tracks:,d} trajectories...')
    with tqdm(total=len(self.track_list)) as pbar:
    tracks_to_remove, tracks_to_add = [], []
    num_gaps_l1 = 0
    for N in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == N]
        num_gaps_l1 += track.fill_gaps_l1()
        # The Track is splitted for gaps of length-2 or more:
        tables = np.split(track.table, np.where(np.diff(track.table['frame']) != 1)[0]+1)
        if len(tables) != 1:
            for table in tables:
                if len(table) >= par['threshold']['track']:
                    tracks_to_add.append(Track(track.folder_name, track.id, table))
            tracks_to_remove.append(track)
    for track in tracks_to_remove:
        self.track_list.remove(track)
    for track in tracks_to_add:
        self.track_list.append(track)
    print(f"{num_gaps_l1:,d} length-1 gaps were filled in and {len(tracks_to_add):,d} tracks created due to length-2+ gaps.")

def predict_states(track_df, model):
    num_tracks = len(track_df['track_id'].unique())
    print(f'\nState prediction of {num_tracks:,d} trajectories...')
    for N in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == N]

        features = np.array(track[['displ_x', 'displ_y', 'dist', 'mean_dist_1', 'mean_dist_2', 'angle']].to_numpy()[1:-1])
        features = features.reshape(1, features.shape[0], features.shape[-1])
        predicted_states = model.predict(features)
        predicted_states = [np.nan] + predicted_states.argmax(axis=2)[0].tolist() + [np.nan]
        track_df.loc[track.index, 'state'] = predicted_states
