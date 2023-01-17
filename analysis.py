"""analysis functions

This modules contains functions for trajectory analysis.
"""

# Third-party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns


def convert_sigma(diffusion, parms):
    """Convert diffusion to sigma value."""
    unit = (parms['pixel_size']**2)/parms['time_frame']
    sigma = np.sqrt((diffusion/unit)*2)
    return sigma

def convert_diffusion(sigma, parms):
    """Convert sigma to diffusion value."""
    unit = (parms['pixel_size']**2)/parms['time_frame']
    diffusion = ((sigma^2)/2)*unit
    return diffusion

def compute_ptm(track_df, parms):
    """Compute the probability transition matrix."""
    counts = np.zeros((parms['num_states'], parms['num_states']), dtype=int)
    for track_id in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == track_id]
        for i, j in zip(track['state'], track['state'][1:]):
            if pd.notna(i) and pd.notna(j):
                # print(i, j)
                counts[int(i)][int(j)] += 1
    # estimate Probs
    sum_of_rows = counts.sum(axis=1)
    # if some row-sums are zero, replace with any number becasue the sum in in denomimator and
    # the numerator will be 0 in any case, so we will get the correct 0 for p_ij in the result
    sum_of_rows[sum_of_rows == 0] = 1
    pis = counts / sum_of_rows[:, None]
    ptm = pd.DataFrame(pis)
    ptm['num'] = sum_of_rows
    return ptm

def make_tracklet_lists(track_df, parms):
    """Segment the trajectories into tracklet (based on the state group) and regroup them within the
    same list."""
    tracklet_lists = [[] for _ in range(parms['num_states'])]
    for track_id in tqdm(track_df['track_id'].unique()):
        track = track_df[track_df['track_id'] == track_id]
        track = track.dropna()
        ind = np.where(np.diff(track['state']) != 0)[0]
        ind = np.insert(ind, len(ind), len(track)-1)
        tmp = 0
        for i in ind:
            tracklet_tmp = track[tmp:i+1]
            state = tracklet_tmp.iloc[0]['state']
            if not pd.notna(state):
                print(tracklet_tmp)
                continue
            tracklet_lists[int(state)].append(tracklet_tmp)
            tmp = i+1
    # Check states
    for state in range(parms['num_states']):
        for track in tracklet_lists[state]:
            mean_states = np.mean(track['state'])
            if mean_states not in (0.0, 1.0, 2.0):
                print(mean_states)
    return tracklet_lists

def compute_msd(track, size, dim=2):
    """Compute the mean square displacement (MSD) for a given track in order to estimate D and alpha
    using the formula:  log(MSD(dt)) ~ alpha.log(dt) + log(C), with C = 2nD.
    """
    def f_slope_intercept(x_val, a_val, b_val):
        """Linear regression y = ax + b."""
        return a_val*x_val + b_val
    if size <= 2:
        size = 3
    coords = {'x': track['x'].to_numpy(), 'y': track['y'].to_numpy()}
    delta_array = np.arange(1, size+1)
    msd = np.zeros(delta_array.size)
    sigma = np.zeros(delta_array.size)
    for i, delta in enumerate(delta_array):
        if dim == 2:
            x_displ = coords['x'][delta:]-coords['x'][:-delta]
            y_displ = coords['y'][delta:]-coords['y'][:-delta]
            res = abs(x_displ)**2 + abs(y_displ)**2
            msd[i] = np.mean(res)
            sigma[i] = np.std(res)
        else:
            print('Dimension should be 2.')
    # popt, _ = curve_fit(f_slope_intercept, np.log(delta_array), np.log(msd))
    popt, _ = curve_fit(f_slope_intercept, np.log(delta_array), np.log(msd), sigma=sigma,
                        absolute_sigma=True)
    alpha = popt[0] # slope
    log_c = popt[1] # intercept
    diffusion = np.exp(log_c)/(2*dim)
    return (msd, delta_array, alpha, log_c, diffusion)

def compute_all_msd(tracklet_lists, parms):
    """Compute the MSD for each given track."""
    motion_parms = [{'alpha': [], 'diffusion': []} for _ in range(parms['num_states'])]
    for state, tracklet_list in enumerate(tracklet_lists):
        print(state)
        for tracklet in tracklet_list:
            if len(tracklet) >= parms['length_threshold']:
                _, _, alpha, _, diffusion = compute_msd(tracklet, size=4)
                diffusion = diffusion * ((parms['pixel_size']**2)/(parms['time_frame']))
                print(f"spot {tracklet['track_id'].iloc[0]},", end=' ')
                print(f"L = {len(tracklet)}, alpha = {alpha:.3}, diffusion = {diffusion:.3}")
                motion_parms[state]['alpha'].append(alpha)
                motion_parms[state]['diffusion'].append(diffusion)
    return motion_parms

def plot_scatter_alpha_diffusion(motion_parms, parms):
    """Make a scatter plot of both the alpha and diffusion distributions."""
    func = None
    for state in range(parms['num_states']):#[2, 1, 0]:
        col = parms['colors'][state]
        dataframe = pd.DataFrame({
            'x': motion_parms[state]['diffusion'],
            'y': motion_parms[state]['alpha']
        })
        if func is None:
            func = sns.JointGrid(x='x', y='y', data=dataframe, height=6.5)
            axs = func.ax_joint
        tmp_alpha = np.array(motion_parms[state]['alpha'])
        tmp_diff = np.array(motion_parms[state]['diffusion'])
        axs.scatter(tmp_diff, tmp_alpha, alpha=0.7, color=col, edgecolor='none', s=2)
        # s=0.3
        centroid = {'diffusion': np.median(tmp_diff),
                    'alpha': np.median(tmp_alpha)}
        axs.plot(centroid['diffusion'], centroid['alpha'], 'o', color='0.1', alpha=1, ms=3)
        axs.plot(centroid['diffusion'], centroid['alpha'], 'o', color='0.1', alpha=1, ms=1, mew=8)
        # bins=150, bins=200)
        sns.histplot(x=tmp_diff, color=col, ax=func.ax_marg_x, alpha=0.5, bins=50)
        sns.histplot(y=tmp_alpha, color=col, ax=func.ax_marg_y, alpha=0.5, bins=50)
    func.ax_joint.set_xscale('log')
    axs.set_xlabel(r'$\mathrm{D}$ $[\mu m^2/s]$')
    axs.set_ylabel(r'$\mathrm{\alpha}$')
    # func.ax_joint.set_xscale('log')
    # func.ax_marg_x.set_xscale('log')
    # func.ax_marg_x.set_yscale('log')
    func.ax_marg_x.grid(axis='x', alpha=0.5)
    func.ax_marg_y.grid(axis='y', alpha=0.5)
    axs.set_xlim([10**-4, 10**2])
    axs.set_ylim([0, 2])
    axs.grid(alpha=0.5)
    fig = plt.gcf()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    # plt.savefig('figures/fig4_scatter_alpha_D_msd_dpi=600.png', dpi=600, transparent=True)
    plt.show()
