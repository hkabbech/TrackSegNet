"""Analysis and plotting functions

This modules contains functions for trajectory analysis and plotting.
"""

# Third-party modules
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
import seaborn as sns

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'figure.labelsize': 13,
    'mathtext.fontset': 'dejavusans'
})

def convert_sigma(diffusion, parms):
    """Converts diffusion to sigma value.

    :param diffusion: Diffusion value [um**2/s] to convert.
    :type diffusion: float
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: sigma value.
    :rtype: float
    """
    sigma = np.sqrt((diffusion/parms['unit_diff'])*2)
    return sigma

def convert_diffusion(sigma, parms):
    """Converts sigma to diffusion value.

    :param sigma: Sigma value to convert.
    :type sigma: float
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: Diffusion value [um**2/s].
    :rtype: float
    """
    diffusion = ((sigma**2)/2)*parms['unit_diff']
    return diffusion

def compute_ptm(track_df, parms):
    """Computes the probability transition matrix (PTM).

    :param track_df: Dataframe containing all extracted trajectories.
    :type track_df: pd.DataFrame
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: probabilities to transition between states
    :rtype: pd.DataFrame
    """
    print('\nCompute probability transition matrix (PTM)...')
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
    ptm.columns = ptm.columns+1
    ptm['num'] = sum_of_rows
    ptm.to_csv(parms['result_path']/'transition_probabilities.csv', index=False)
    return ptm

def make_tracklet_lists(track_df, parms):
    """Segments the trajectories into tracklet (based on the state group) and regroup them within the
    same list.

    :param track_df: Dataframe containing all extracted trajectories.
    :type track_df: pd.DataFrame
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :rtype: list
    """
    print('\nSegment and group tracklets per diffusive states...')
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
            if mean_states not in [float(i) for i in range(parms['num_states'])]:
                print(mean_states)
    return tracklet_lists

def compute_msd(track, size, dim=2):
    """Computes the mean square displacement (MSD) for a given track in order to estimate D and alpha
    using the formula:  log(MSD(dt)) ~ alpha.log(dt) + log(C), with C = 2nD.
    
    :param track: Dataframe containing a trajectory's coordinates.
    :type track: pd.DataFrame
    :param size: Number of delta time points to use for the MSD curve fit.
    :type size: int
    :param dim: Dimentionality of the track [Defaults: 2].
    :type dim: int
    :return: (msd values, time axis, measured alpha (slope), measured log_C (intercept), resulting diffusion)
    :rtype: (np.array, np.array, float, float, float)
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
    """Computes the MSD for each given track.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :return: Dictionary containing lists of measured motion parameters for each track with keys: alpha, diffusion and track_id.
    :rtype: dict
    """
    print('\nCompute MSD...')
    motion_parms = [{'alpha': [], 'diffusion': [], 'track_id': []}\
                    for _ in range(parms['num_states'])]
    for state in tqdm(range(parms['num_states'])):
        for tracklet in tracklet_lists[state]:
            if len(tracklet) >= parms['length_threshold']:
                _, _, alpha, _, diffusion = compute_msd(tracklet, size=4)
                diffusion = diffusion * parms['unit_diff']
                # print(f"spot {tracklet['track_id'].iloc[0]},", end=' ')
                # print(f"L = {len(tracklet)}, alpha = {alpha:.3}, diffusion = {diffusion:.3}")
                motion_parms[state]['alpha'].append(alpha)
                motion_parms[state]['diffusion'].append(diffusion)
                motion_parms[state]['track_id'].append(tracklet['track_id'].iloc[0])
        pd.DataFrame(motion_parms[state]).to_csv(
            parms['result_path']/f"motion_parms_state_{state+1}.csv", index=False)
    return motion_parms

def plot_scatter_alpha_diffusion(motion_parms, parms):
    """Makes a scatterplot of the alpha and diffusion distributions.

    :param motion_parms: Dictionary containing lists of measured motion parameters for each track with keys: alpha, diffusion and track_id.
    :type motion_parms: dict
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    """
    print('\nPlot D/alpha scatterplot...')
    print('centroids...')
    func = None
    for state in range(parms['num_states']):
        col = parms['colors'][state]
        dataframe = pd.DataFrame({
            'x': motion_parms[state]['diffusion'],
            'y': motion_parms[state]['alpha']
        })
        if func is None:
            func = sns.JointGrid(x='x', y='y', data=dataframe, height=6.5)
            axs = func.ax_joint
        tmp_diff = np.array(motion_parms[state]['diffusion'])
        tmp_alpha = np.array(motion_parms[state]['alpha'])
        tmp_diff = tmp_diff[tmp_alpha >= 0]
        tmp_alpha = tmp_alpha[tmp_alpha >= 0]
        axs.scatter(tmp_diff, tmp_alpha, alpha=0.6, color=col, edgecolor='none', s=2)
        centroid = {'diffusion': np.median(tmp_diff), 'alpha': np.median(tmp_alpha)}
        print(f"state {state+1}: alpha = {centroid['alpha']:>4.2f},", end=" ")
        print(f"D = {centroid['diffusion']:>4.2f} µm²/s")
        axs.plot(centroid['diffusion'], centroid['alpha'], 'o', color='0.1', alpha=1, ms=3)
        axs.plot(centroid['diffusion'], centroid['alpha'], 'o', color='0.1', alpha=1, ms=1, mew=8)
        # bins=150, bins=200)
        bins = np.logspace(-2.5, 1.5, 200)
        sns.histplot(x=tmp_diff, color=col, ax=func.ax_marg_x, alpha=0.5, bins=bins, stat='count')
        bins = np.linspace(0, 2, 100)
        sns.histplot(y=tmp_alpha, color=col, ax=func.ax_marg_y, alpha=0.5, bins=bins, stat='count')
    axs.set_xscale('log')
    axs.set_xlabel(r'$\mathrm{D}$ $[\mu m^2/s]$')
    axs.set_ylabel(r'$\mathrm{\alpha}$')
    func.ax_marg_x.grid(axis='x', alpha=0.5)
    func.ax_marg_y.grid(axis='y', alpha=0.5)
    axs.set_xlim([10**-2, 10**0.5])
    axs.set_ylim([0, 2])
    axs.grid(alpha=0.5)
    fig = plt.gcf()
    # fig.suptitle(f"{parms['data_path'].name} dataset")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{parms['result_path']}/fig_{parms['data_path'].name}_scatterplot_msd"+\
                f".{parms['fig_format']}",
                dpi=parms['fig_dpi'], transparent=parms['fig_transparent'])
    plt.close()

def plot_proportion(tracklet_lists, parms):
    """Plots the proportion of tracklets in each diffusive state as a pie chart.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    """
    print('\nPlot proportion...')
    tracklet_num = [len(tracklet_list) for tracklet_list in tracklet_lists]
    print(tracklet_num)
    labels = [f'state {state+1}\nn = {len(tracklet_list)}'\
              for state, tracklet_list in enumerate(tracklet_lists)]
    fig, axs = plt.subplots(1, figsize=(4, 4))
    axs.pie(tracklet_num, labels=labels, autopct='%1.1f%%', colors=parms['colors'], wedgeprops={"alpha": 0.6})
    axs.axis('equal')
    fig.suptitle(f"{parms['data_path'].name} dataset")
    fig.tight_layout()
    plt.savefig(f"{parms['result_path']}/fig_{parms['data_path'].name}_tracklet_proportion."+\
                f"{parms['fig_format']}",
                dpi=parms['fig_dpi'], transparent=parms['fig_transparent'])
    plt.close()

def compute_displ(tracklet_lists, parms, dtime=1):
    """Computes the displacements for each tracklet state.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param dtime: Time interval between two data points [Default: 1]
    :type dtime: int
    :return: List of calculated displacements for each tracklet group.
    :rtype: list
    """
    all_displ_tracklet = [{dtime: None} for _ in range(len(tracklet_lists))]
    for state in tqdm(range(parms['num_states'])):
        for tracklet in tracklet_lists[state]:
            displ_x = tracklet[dtime:].reset_index(drop=True)['x']-\
                tracklet.reset_index(drop=True)['x']
            displ_x = displ_x.to_numpy().astype(np.float64)
            displ_y = tracklet[dtime:].reset_index(drop=True)['y']-\
                tracklet.reset_index(drop=True)['y']
            displ_y = displ_y.to_numpy().astype(np.float64)
            if all_displ_tracklet[state][dtime] is None:
                all_displ_tracklet[state][dtime] = {'x': displ_x, 'y': displ_y}
            else:
                all_displ_tracklet[state][dtime]['x'] = np.append(
                    all_displ_tracklet[state][dtime]['x'], displ_x)
                all_displ_tracklet[state][dtime]['y'] = np.append(
                    all_displ_tracklet[state][dtime]['y'], displ_y)

        all_displ_tracklet[state][dtime]['x'] = all_displ_tracklet[state][dtime]['x']\
            [~np.isnan(all_displ_tracklet[state][dtime]['x'])]*parms['pixel_size']
        all_displ_tracklet[state][dtime]['y'] = all_displ_tracklet[state]\
            [dtime]['y'][~np.isnan(all_displ_tracklet[state][dtime]['y'])]*parms['pixel_size']
    return all_displ_tracklet

def plot_displ(tracklet_lists, parms, dtime=1):
    """Plots the distribution of displacements per tracklet state.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param dtime: Time interval between two data points [Default: 1]
    :type dtime: int
    """
    print(f'\nPlot displacements with dt={dtime}...')
    all_displ_tracklet = compute_displ(tracklet_lists, parms, dtime=dtime)

    plt.close()
    hist = {}
    for state, all_displ_tracklet_state in enumerate(all_displ_tracklet):
        displ_tmp = np.concatenate((all_displ_tracklet_state[dtime]['x'],
                                    all_displ_tracklet_state[dtime]['y']))
        binwidth = 0.02
        bins = np.arange(-0.4-0.01, 0.4 + binwidth, binwidth)
        sns.histplot(displ_tmp, bins=bins, stat='density')
        bars = plt.gca().patches[0]
        xy_coords = np.array([[bars.get_x()+(bars.get_width()/2), bars.get_height()]\
                      for bars in plt.gca().patches]).T
        hist[state] = xy_coords
        plt.close()

    fig, axs = plt.subplots(1, figsize=(5, 5))
    axs.grid(alpha=0.5)

    for state, _ in enumerate(all_displ_tracklet):
        axs.plot(hist[state][0], hist[state][1], 'o-', color=parms['colors'][state],
                 label=f'state {state+1}', ms=5.5, lw=1)
    axs.set_ylabel('Probability Density')
    axs.set_xlabel(r'$\mathrm{\Delta r}$ $[\mu m]$')
    axs.set_title('Distribution of X and Y displacements')
    axs.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(f"{parms['result_path']}/fig_{parms['data_path'].name}_displ"+\
                f".{parms['fig_format']}",
                dpi=parms['fig_dpi'], transparent=parms['fig_transparent'])
    plt.close()

def compute_angles(tracklet_lists, parms, dtime=1):
    """Computes the angles for each tracklet state.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param dtime: Time interval between two data points [Default: 1]
    :type dtime: int
    :return: Dictionary of numpy arrays for each tracklet group containing the calculated angles.
    :rtype: dict
    """
    tracklet_angles = {}
    for state in tqdm(range(parms['num_states'])):
        angles_current_state = np.array([])
        for tracklet in tracklet_lists[state]:
            displ_x = tracklet[dtime:].reset_index()['x']-tracklet.reset_index()['x']
            displ_x = displ_x.to_numpy().astype(np.float64)
            displ_y = tracklet[dtime:].reset_index()['y']-tracklet.reset_index()['y']
            displ_y = displ_y.to_numpy().astype(np.float64)
            # calulate angles
            alpha_1 = np.arctan2(displ_y[:-1], displ_x[:-1])
            alpha_2 = np.arctan2(displ_y[1:], displ_x[1:]) + 2*np.pi
            angles = (alpha_2 - alpha_1) % (2*np.pi)
            angles[angles > np.pi] = angles[angles > np.pi] - (2*np.pi)
            angles = angles[~np.isnan(angles)]
            angles_current_state = np.append(angles_current_state, angles)
        tracklet_angles[state] = angles_current_state
    return tracklet_angles

def calculate_fold_180_0(angle_list):
    """Calculates the fold anisotropy for a given list of angles. ref doi:10.1038/s41589-019-0422-3

    :param angle_list: Numpy array containing the calculated angles for a given tracklet group.
    :type angle_list: dict
    :return: (measure fold anisopropy, total number of 180 +/-30° angles, total number of 0 +/-30° angles)
    :rtype:  (float, int, int)
    """
    num = {'180': 0, '0': 0}
    for angle in angle_list:
        # angle = angle + 2*np.pi
        if ((5*np.pi)/6 <= angle <= np.pi) or (-np.pi <= angle <= (-5*np.pi)/6):
            num['180'] += 1
        elif (0 <= angle <= np.pi/6) or (-np.pi/6 <= angle <= 0):
            num['0'] += 1
    try:
        fold_180_0 = num['180'] / num['0']
    except ZeroDivisionError:
        fold_180_0 = None
    return (fold_180_0, (num['180'], num['0']))

def plot_angles(tracklet_lists, parms, dtime=1):
    """Plots the angular distribution per tracklet state.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param dtime: Time interval between two data points [Default: 1]
    :type dtime: int
    """
    print(f'\nPlot angles with dt={dtime}...')
    tracklet_angles = compute_angles(tracklet_lists, parms, dtime=dtime)

    plt.close()
    hist = {}
    for state, _ in enumerate(tracklet_angles):
        sns.histplot(tracklet_angles[state], bins=25, stat='density',
                     color=parms['colors'][state], alpha=0.5)
        bars = plt.gca().patches[0]
        xy_coords = np.array([[bars.get_x()+(bars.get_width()/2), bars.get_height()]\
                      for bars in plt.gca().patches]).T
        hist[state] = xy_coords
        plt.close()

    fig, axs = plt.subplots(1, figsize=(5, 5))
    axs.grid(alpha=0.5)

    formula = r'$\mathrm{f_{180/0}}$'
    labels = []
    for state, _ in enumerate(tracklet_lists):
        axs.plot(hist[state][0], hist[state][1], 'o-', color=parms['colors'][state],
                 label=f'state {state+1}',
                 ms=5.5, lw=1)
        fold = calculate_fold_180_0(tracklet_angles[state])[0]
        labels.append(mlines.Line2D([], [], color=parms['colors'][state],
                                    label=f"{formula} = {fold:.2}", lw=1))
    axs.set_xlim(-np.pi, np.pi)
    # axs.set_ylim(0.05, 0.30)
    axs.set_xticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    axs.set_xticklabels([r'$\mathrm{-180^o}$', r'$\mathrm{-90^o}$', r'$\mathrm{0^o}$',
                         r'$\mathrm{90^o}$', r'$\mathrm{180^o}$'])
    axs.legend(handles=labels, loc='upper center')
    axs.set_xlabel(r'$\mathrm{\theta}$')
    axs.set_ylabel('Probability Density')
    axs.set_title('Angular distributions')
    fig.tight_layout()
    plt.savefig(f"{parms['result_path']}/fig_{parms['data_path'].name}_angles"+\
                f".{parms['fig_format']}",
                dpi=parms['fig_dpi'], transparent=parms['fig_transparent'])
    plt.close()

def compute_vac(tracklet_lists, parms, dtime=1, thr=1000):
    """Computes the velocity autocorrelation (VAC) curves for each tracklet state.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param dtime: Time interval between two data points [Default: 1]
    :type dtime: int
    :param thr: Threshold for the vac limiting the number of points calculated [Default: 1000].
    :type thr: int
    :return:  List of calculated VAC for each tracklet group.
    :rtype: list
    """
    time_frame_ms = parms['time_frame']*1000
    all_vac = [{dtime: None} for _ in range(len(tracklet_lists))]
    for state in tqdm(range(parms['num_states'])):
        vac_tmp = []
        for tracklet in tracklet_lists[state]:
            if len(tracklet) <= dtime:
                continue
            xcoord = tracklet['x'].to_numpy()
            vac_tmp.append(np.dot(((xcoord[dtime:]-xcoord[:-dtime])/dtime)*time_frame_ms,
                                  ((xcoord[dtime]-xcoord[0])/dtime)*time_frame_ms))
            ycoord = tracklet['y'].to_numpy()
            vac_tmp.append(np.dot(((ycoord[dtime:]-ycoord[:-dtime])/dtime)*time_frame_ms,
                                  ((ycoord[dtime]-ycoord[0])/dtime)*time_frame_ms))
        vac = np.zeros((len(vac_tmp), thr))
        vac[:] = np.nan
        for i, _ in enumerate(vac):
            vac[i][:len(vac_tmp[i])] = vac_tmp[i][:1000]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            all_vac[state][dtime] = np.nanmean(vac, axis=0)
    return all_vac

def plot_vac(tracklet_lists, parms, dtime=1):
    """Plots the velocity autocorrelation curves per tracklet state.

    :param tracklet_lists: List of tracklet groups. Each group consists of a dataframe of tracklets.
    :type tracklet_lists: list
    :param parms: Stored parameters containing global variables and instructions.
    :type parms: dict
    :param dtime: Time interval between two data points [Default: 1]
    :type dtime: int
    """
    print(f'\nPlot velocity autocorrelation with dt={dtime}...')
    all_vac = compute_vac(tracklet_lists, parms, dtime=dtime)

    fig, axs = plt.subplots(1, figsize=(5, 5))
    axs.grid(alpha=0.5)
    # plot theoretical curves
    xaxis = np.arange(0, 100)
    alphas = np.arange(0.1, 1.3, 0.1)
    cmap = cm.get_cmap("viridis_r")
    cmap = cmap(np.linspace(0, 1, len(alphas)))
    thr_prev = None
    for i, alpha_thr in enumerate(alphas[::-1]):
        thr = ((xaxis+1)**(alpha_thr) + abs(xaxis-1)**(alpha_thr) - 2*(xaxis**(alpha_thr)))/2
        if thr_prev is None:
            thr_prev = thr.copy()
        else:
            axs.fill_between(xaxis*parms['time_frame']*1000, thr, thr_prev, facecolor=cmap[i],
                             alpha=1)
            thr_prev = thr.copy()

    for state, _ in enumerate(tracklet_lists):
        x_axis_norm = np.array([i*parms['time_frame']*1000\
                               for i in range(len(all_vac[state][dtime]))])
        axs.plot(x_axis_norm, (all_vac[state][dtime]/all_vac[state][dtime][0]), 'o-',
                 color=parms['colors'][state], lw=1, ms=5.5, label=f'state {state}')
    axs.set_ylabel(r'$\mathrm{C^{(\delta=1)}_{v}(\Delta t)}$'+\
                   r'$\mathrm{/}$ $\mathrm{C^{(\delta=1)}_{v}(0)}$')
    axs.set_xlabel(r'$\mathrm{\Delta t}$')#' $[ms]$')
    axs.set_xlim([0, 30])
    axs.set_xticks(np.arange(0, parms['time_frame']*1000*5, parms['time_frame']*1000))
    axs.set_ylim([-0.5, 1])
    # axs.legend(loc='upper right', fontsize='x-small')
    thr_curves = plt.contourf([[0, 0], [0, 0]], alphas, cmap='viridis')
    fig.colorbar(thr_curves, ax=axs, ticks=alphas).\
                 set_label(label=r'$\mathrm{\alpha}$ value of theoretical fBm')
    axs.set_title('Velocity autocorrelation curves')
    fig.tight_layout()
    plt.savefig(f"{parms['result_path']}/fig_{parms['data_path'].name}_vac"+\
                f".{parms['fig_format']}",
                dpi=parms['fig_dpi'], transparent=parms['fig_transparent'])
    plt.close()
