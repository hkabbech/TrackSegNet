"""
.. module:: plot.py
   :synopsis: This module implements plot functions
"""

# Third-party modules
import os
import numpy as np
import pandas as pd
import pydotplus
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import rc
import colorcet as cc
import seaborn as sns

sns.set(color_codes=True)
rc('text', usetex=True)


def get_color_list(num_states):
    """
        Create list of colors
    """
    def get_hex(col_dict):
        return colors.cnames[col_dict['name']][1:]
    blue = {'name': 'darkblue', 'cmap': cc.cm.kbc, 'cmap_lim': (0.1, 0.9)}
    blue['hex'] = get_hex(blue)
    orange = {'name': 'darkorange', 'cmap': cm.autumn, 'cmap_lim': (0.2, 0.8)}
    orange['hex'] = get_hex(orange)
    red = {'name': 'red', 'cmap': cc.cm.CET_CBTL1, 'cmap_lim': (0.15, 0.65)}
    red['hex'] = get_hex(red)
    cyan = {'name': 'cyan', 'cmap': cc.cm.kbc, 'cmap_lim': (0.55, 1)}
    cyan['hex'] = get_hex(cyan)
    green = {'name': 'green', 'cmap': cc.cm.kgy, 'cmap_lim': (0.15, 0.9)}
    green['hex'] = get_hex(green)
    purple = {'name': 'darkviolet', 'cmap': cc.cm.CET_L7, 'cmap_lim': (0.1, 0.6)}
    purple['hex'] = get_hex(purple)

    if num_states == 2:
        return [blue, red]
    elif num_states == 3:
        return [blue, orange, red]
    elif num_states == 4:
        return [blue, orange, green, red]

def get_label_list(num_states):
    """
    """
    if num_states == 2:
        return ['immobile tracklets', 'mobile tracklets']
    elif num_states == 3:
        return ['immobile tracklets', 'slow tracklets', 'fast tracklets']
    elif num_states ==4:
        return ['immobile tracklets', 'slow tracklets', 'restricted fast tracklets', 'free fast tracklets']


def plot_tracklets(path, data, par, num=600, fformat='pdf', dpi=1000):
    """
        Plot tracks
    """
    # color_rb = cm.rainbow(np.linspace(0, 1, len(data.tracks.list[:num])))
    fig, ax = plt.subplots(1)
    # # Plot the tracks in the same window
    # for track in data.tracks.list[:num]:
    #     ax[0].plot(track.df['x'], track.df['y'], color=color_rb[track.id], lw=0.5)
    for state, tracklet_list in enumerate(data.tracklet_lists):
        for tracklet in tracklet_list[:num]:
            ax.plot(tracklet.table['x'], tracklet.table['y'], color=par['colors'][state]['name'], lw=0.5)
    fig.tight_layout()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.margins(0)    # plt.subplots_adjust(hspace=0.05)
    fig.savefig(path/'example_tracklets_plot.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_ptm_graph(path, ptm, par, fformat='png', dpi=500):
    """
        Plot the probability transition matrix graph
    """
    div = ptm['num'][par['num_states']-1]/0.5
    mult = 45
    space = int(ptm['num'][0]/div)
    if par['num_states'] == 3:
        port = [('nw', ':'), ('se', ':'), (':', 'sw')]
        pos = ["{}, 0!".format(space/2), "{}, -{}!".format(space, space), "0, -{}!".format(space)]
    elif par['num_states'] == 4:
        port = [('nw', ':'), (':', 'ne'), ('se', ':'), (':', 'sw')]
        pos = ["0, 0!", "{}, 0!".format(space), "{}, -{}!".format(space, space), "0, -{}!".format(space)]

    graph = pydotplus.Dot(graph_type='digraph', size="10, 10",  dpi=dpi)#, rankdir='LR')

    # Add nodes
    nodes = []
    for state in range(par['num_states']):
        name = "State {}".format(state)
        size = ptm['num'][state]/div
        color = par['colors'][state]['name']
        if color == 'green': color = 'darkgreen'
        new_node = pydotplus.Node(name, style="filled", fillcolor=color, fontcolor='white',
                                  pos=pos[state], height=size, width=size, shape="circle", color='transparent')
        nodes.append(new_node)
        graph.add_node(new_node)

        # Add the edge coming from and leaving from the same node
        label = '{:.3f}'.format(ptm[state][state])
        edge_loop = pydotplus.Edge(nodes[state], nodes[state], penwidth=6, label=label,
                       headport=port[state][0], tailport=port[state][1], fontcolor="black",
                       color="black", labeldistance=3, arrowhead='normal')
        graph.add_edge(edge_loop)

    # Add edges betweem nodes:
    for state_i, node_i in enumerate(nodes):
        for state_j, node_j in enumerate(nodes):
            if state_i == state_j:
                continue
            label = '{:.3f}'.format(ptm[state_i][state_j])
            penwidth = ptm[state_i][state_j]*mult
            color = par['colors'][state_i]['name']
            if color == 'green': color = 'darkgreen'
            edge = pydotplus.Edge(node_i, node_j, penwidth=penwidth, headlabel=label,
                                  fontcolor=color, color=color, arrowhead='lnormal', labeldistance=3,
                                  labelangle=-40)
            graph.add_edge(edge)

    # Save graph
    path = str(path/'PTM_graph.{}'.format(fformat))
    if fformat == 'png':
        graph.write_png(path, prog='neato')
    elif fformat == 'pdf':
        graph.write_pdf(path, prog='neato')      

def plot_power_figures(path, data, par, fformat='png', dpi=1000):
    """
        Plot MSS figures
    """
    _, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.fill_between(par['power_values'], par['power_values']/2.0, par['power_values'],
                    facecolor='darkslateblue', edgecolor='none',
                    alpha=0.1, label='Region from diffusion to linear motion')
    for state, gamma in enumerate(data.gamma):
        ax.plot(par['power_values'], gamma, '-o', color=par['colors'][state]['name'],
                markeredgecolor='black', label='{}, {}= {:.2}'\
                .format(par['labels'][state], r'$S_{MSS}$', data.smss[state]))
    plt.xlabel(r'$power$', labelpad=10, size='large')
    plt.ylabel(r'$\mathrm{\gamma_p}$', labelpad=10, size='x-large')
    ax.set_xlim(par['power_values'][0], par['power_values'][-1])
    ax.legend(loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig(path/'graph_power_gamma.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

    _, ax = plt.subplots(figsize=(6.4, 4.8))
    plt.subplots_adjust(right=1)
    for state, logc in enumerate(data.logc):
        ax.plot(par['power_values'], logc, '-o', color=par['colors'][state]['name'], markeredgecolor='black',
                label=par['labels'][state])
    ax.set_xlim(par['power_values'][0], par['power_values'][-1])
    plt.xlabel(r'$power$', labelpad=10, size='large')
    plt.ylabel(r'$\mathrm{log} \ C_\mathrm{p}$', labelpad=10, size='large')
    ax.legend(loc='upper left', ncol=1)
    plt.tight_layout()
    plt.savefig(path/'graph_power_logC.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

    _, ax = plt.subplots(figsize=(6.4, 4.8))
    # plt.subplots_adjust(right=1)
    for state, diffusion in enumerate(data.diffusion):
        plt.plot(par['power_values'], diffusion, '-o', markeredgecolor='black', color=par['colors'][state]['name'],
                 label=par['labels'][state])
    ax.set_xlim(par['power_values'][0], par['power_values'][-1])
    plt.xlabel(r'$power$', labelpad=10, size='large')
    plt.ylabel(r'$D \ \mathrm{[\mu m^2/s]}$', labelpad=10, size='large')
    ax.legend(loc='center left', ncol=1) # loc=(0.02, 0.7)
    plt.tight_layout()
    plt.savefig(path/'power_diffusion.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_smss_diffusion(path, data, par, alpha_1=0.2, size=20, nb_hist=50, n_levels=15,
                        fformat='png', dpi=1000, kde_plot=False):
    """
        Plot Smss/Diffusion figures
    """
    legends = [mpatches.Patch(color=par['colors'][state]['name'], alpha=0.5,\
                              label=par['labels'][state]) for state in range(par['num_states'])]

    all_diffusion = [[] for _ in range(par['num_states'])]
    all_smss = [[] for _ in range(par['num_states'])]
    for state, tracklet_list in enumerate(data.tracklet_lists):
        for tracklet in tracklet_list:
            all_diffusion[state].append(np.mean(tracklet.diffusion[par['where_power_2']]))
            all_smss[state].append(tracklet.smss)

    alpha_2 = 0.4
    # lim_diff = {'min': [], 'max': []}
    xlab, ylab = r'$D$ $\mathrm{[\mu m^2/s]}$', r'$S_{\mathrm{MSS}}$'
    dataframe = pd.DataFrame({xlab: all_diffusion[0], ylab: all_smss[0]})
    func = sns.JointGrid(x=xlab, y=ylab, data=dataframe)
    ax = func.ax_joint
    ax.set_xlabel(xlab, size='large')
    ax.set_ylabel(ylab, size='large')
    ax.set_xscale('log', basex=10)

    # func.plot_joint(plt.scatter, color=colors[init], alpha=alpha_1, edgecolor='none', s=size)
    for state in range(par['num_states']):
        try:
            # lim_diff['min'].append(min(diff))
            # lim_diff['max'].append(max(diff))
            ax.scatter(all_diffusion[state], all_smss[state], alpha=alpha_1,
                       color=par['colors'][state]['name'], edgecolor='none', s=size)
            centroid = {'diffusion': np.median(all_diffusion[state]),
                        'smss': np.median(all_smss[state])}
            ax.plot(centroid['diffusion'], centroid['smss'], 'o', color='0.2', alpha=1, ms=3)
            ax.plot(centroid['diffusion'], centroid['smss'], 'o', color='0.2', alpha=1, ms=1, mew=8)
            func.ax_marg_y.hist(all_smss[state], color=par['colors'][state]['name'],
                                edgecolor='none', alpha=alpha_2, orientation='horizontal',
                                bins=np.linspace(min(all_smss[state]),
                                                 max(all_smss[state]),
                                                 nb_hist))
            func.ax_marg_x.hist(all_diffusion[state], color=par['colors'][state]['name'],
                                edgecolor='none', alpha=alpha_2,
                                bins=np.logspace(np.log10(min(all_diffusion[state])),
                                                 np.log10(max(all_diffusion[state])), nb_hist))
        except:
            print('empty sequence(s)')
    ax.axhspan(0.4, 0.6, facecolor='darkslateblue', edgecolor='none', alpha=0.15)
    func.ax_marg_y.axhspan(0.4, 0.6, facecolor='slateblue', edgecolor='none', alpha=0.15)
    # ax.set_xlim(min(lim_diff['min']), max(lim_diff['max']))
    ax.set_xlim(10**-3, 10)
    ax.set_ylim(0, 1)
    ax.legend(handles=legends, loc='upper left')
    # ax.get_shared_x_axes().join(ax, ax)
    # ax.autoscale()
    func.fig.set_size_inches((6, 6))
    plt.savefig(path/'diffusion_smss_all.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

    if kde_plot:
        # Plot KDE:
        xlab, ylab = r'$D$ $\mathrm{[\mu m^2/s]}$', r'$S_{\mathrm{MSS}}$'

        ext_all_diffusion, ext_all_smss = [], []
        for state in range(par['num_states']):
            ext_all_diffusion.extend(all_diffusion[state])
            ext_all_smss.extend(all_smss[state])

        kdemap = colors.LinearSegmentedColormap.from_list('kdemap', ['white', 'midnightblue'], N=1000)

        func = sns.JointGrid(x=np.log10(ext_all_diffusion), y=ext_all_smss, ylim=(0, 1), xlim=(-3, 1))
        ax = func.ax_joint

        ax.set_xlabel(xlab, size='large')
        ax.set_ylabel(ylab, size='large')
        ax.set_facecolor('white')
        ax.grid(color='lavender')
        func.ax_marg_x.set_facecolor('white')
        func.ax_marg_x.grid(color='lavender')
        func.ax_marg_y.set_facecolor('white')
        func.ax_marg_y.grid(color='lavender')

        sns.kdeplot(np.log10(ext_all_diffusion), ext_all_smss, ax=ax, cmap=kdemap, shade=True,
                    shade_lowest=False, n_levels=n_levels, vmin=0, vmax=2.5)
        sns.kdeplot(np.log10(ext_all_diffusion), ext_all_smss, ax=ax, shade=False, alpha=0.5,
                    cmap=cm.gray, n_levels=n_levels)
        sns.kdeplot(np.log10(ext_all_diffusion), ax=func.ax_marg_x, shade=True, color='midnightblue', alpha=0.4)
        sns.kdeplot(np.log10(ext_all_diffusion), ax=func.ax_marg_x, shade=False, color='k', alpha=0.5)
        sns.kdeplot(ext_all_smss, ax=func.ax_marg_y, shade=True, color='midnightblue', alpha=0.4, vertical=True)
        sns.kdeplot(ext_all_smss, ax=func.ax_marg_y, shade=False, color='k', alpha=0.5, vertical=True)

        xlabels = ['$10^{'+str(label)+'}$' for label in range(-3, 2)]
        func.ax_joint.set_xticks(range(-3, 2))
        func.ax_joint.set_xticklabels(xlabels)

        func.fig.set_size_inches((6, 6))
        plt.savefig(path/'diffusion_smss_all_kde.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
        plt.close()

def plot_tracklets_max_projection(path, data, par, background='Greys_r', lw=0.3, fformat='png',
                                  dpi=1000):
    """
        Tracklets on top of max projection
    """
    os.makedirs(path/'max_projection_plots', exist_ok=True)

    cmaps = {folder_name.stem: {} for folder_name in par['folder_names']}
    for state in range(par['num_states']):
        for folder_name in par['folder_names']:
            cmaps[folder_name.stem][state] = par['colors'][state]['cmap']\
                                        (np.linspace(par['colors'][state]['cmap_lim'][0],
                                                     par['colors'][state]['cmap_lim'][1],
                                                     data.num_tracklet_df.loc[state][folder_name.stem]))
    os.makedirs(path, exist_ok=True)
    for folder_name in par['folder_names']:
        for file in os.listdir(folder_name):
            if file.endswith(".tif"):
                imarray = np.array(Image.open(folder_name/file))
                break
        fig, axs = plt.subplots(par['num_states']+2)
        for i, _ in enumerate(axs):
            axs[i].imshow(imarray, cmap=cm.get_cmap(background))
        fig.tight_layout()
        for ax in axs:
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.margins(0.05)
        plt.subplots_adjust(hspace=0.05)
        for state, tracklet_list in enumerate(data.tracklet_lists):
            p = 0
            for tracklet in tracklet_list:
                if tracklet.folder_name.stem == folder_name.stem:
                    axs[1].plot(tracklet.table['x'], tracklet.table['y'], color=par['colors'][state]['name'],
                                lw=lw, alpha=1)
                    axs[state+2].plot(tracklet.table['x'], tracklet.table['y'],
                                      color=cmaps[folder_name.stem][state][p], lw=lw, alpha=1)
                    p += 1
        fig.savefig(path/'max_projection_plots'/'{}_tracklets_{}.{}'.format(folder_name.stem, background, fformat),
                    transparent=True, bbox_inches='tight', pad_inches=0.05, dpi=dpi)
        plt.close(fig)

def plot_violinplots(path, data, par, fformat='png', dpi=200):
    """
        Plot boxplots
    """
    os.makedirs(path/'violinplots', exist_ok=True)

    all_diffusion = [[] for _ in range(par['num_states'])]
    all_smss = [[] for _ in range(par['num_states'])]
    for state, tracklet_list in enumerate(data.tracklet_lists):
        for tracklet in tracklet_list:
            all_diffusion[state].append(tracklet.diffusion)
            all_smss[state].append(tracklet.smss)

    keys = ['num_frames', 'num_frames_log', 'smss', 'diffusion', 'diffusion_log',
            'displ_x_1', 'displ_y_1', 'dist_1', 'mean_dist_1', 'mean_dist_2', 'angle_1']

    all_features = {}
    for key in keys:
        all_features[key] = [[] for _ in range(par['num_states'])]
    for state, tracklet_list in enumerate(data.tracklet_lists):
        all_features['diffusion'][state] = all_diffusion[state]
        all_features['smss'][state] = all_smss[state]
        for tracklet in tracklet_list:
            all_features['num_frames'][state].append(tracklet.num_frames)
            for key in ['displ_x_1', 'displ_y_1', 'dist_1', 'mean_dist_1', 'mean_dist_2', 'angle_1']:
                all_features[key][state].extend(tracklet.table[key][1:-1])
    all_features['diffusion_log'] = [np.log10(i) for i in all_features['diffusion']]
    all_features['num_frames_log'] = [np.log10(i) for i in all_features['num_frames']]

    xlabels = {
        'num_frames': r'$number$ $of$ $frames$',
        'num_frames_log': r'$log$ $number$ $of$ $frames$',
        # 'time': r'$time$ $\mathrm{[ms]}$',
        # 'time_log': r'$log$ $time$ $\mathrm{[ms]}$',
        'smss': r'$S_{\mathrm{MSS}}$',
        'diffusion': r'$D \ \mathrm{[\mu m^2/s]}$',
        'diffusion_log': r'$D \ \mathrm{[\mu m^2/s]}$',
        'displ_x_1': r'$x$ $displacement$',
        'displ_y_1': r'$y$ $displacement$',
        'dist_1': r'$distance$',
        'mean_dist_1': r'$mean$ $distances$ $(order$ $1)$',
        'mean_dist_2': r'$mean$ $distances$ $(order$ $2)$',
        'angle_1': r'$angle$'
    }

    for key in keys:
        ax = sns.violinplot(data=all_features[key], palette=[color['name'] for color in par['colors']],
                            whis="range", orient='h', scale='count', boxprops=dict(alpha=.6))
        ax.set_xlabel(xlabels[key], size='large')
        if key == 'diffusion_log':
            ax.set_xticks(range(-6, 2))
            ax.set_xticklabels(['$10^{'+str(label)+'}$' for label in range(-6, 2)])
        elif key == 'num_frames_log':
            ax.set_xticks(np.arange(0.5, 3, 0.5))
            ax.set_xticklabels(['$10^{'+str(label)+'}$' for label in np.arange(0.5, 3, 0.5)])
        # elif key == 'time_log':
        #     ax.set_xticks(np.arange(1.5, 4, 0.5))
        #     ax.set_xticklabels(['$10^{'+str(label)+'}$' for label in np.arange(1.5, 4, 0.5)])
        elif key == 'angle_1':
            ax.set_xticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
            ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        elif key == 'time' or key == 'num_frames':
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.set_yticklabels([par['labels'][s][:-10] for s in range(par['num_states'])], size='large')
        plt.tight_layout()
        plt.setp(ax.collections, alpha=.6)
        fig = ax.get_figure()
        fig.set_size_inches((6.4, 4.8))
        plt.savefig(path/'violinplots'/'{}.{}'.format(key, fformat), bbox_inches='tight', dpi=dpi)
        plt.close()

def plot_tracklet_proportions(path, data, par, fformat='png', dpi=1000):
    """
        Pie chart
    """
    proportion_list = [len(tracklet_list) for tracklet_list in data.tracklet_lists]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _, _, autotexts = ax.pie(proportion_list, labels=[par['labels'][state][:-10] for state in range(par['num_states'])],
                             colors=[color['name'] for color in par['colors']],
                             autopct=lambda pct: "{:.1f}\%".format(pct),
                             textprops=dict(size='larger'), wedgeprops={'alpha':.6})
    # for state in range(NUM_STATES):
    #     autotexts[state].set_color('white')
    plt.setp(autotexts, size=15, weight="bold")
    plt.axis('equal')
    plt.savefig(path/'tracklet_proportion_piechart.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, color='C0'):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    angles = (angles + np.pi) % (2*np.pi) - np.pi
    if start_zero:
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)
    count, bin = np.histogram(angles, bins=bins)
    widths = np.diff(bin)
    if density is None or density is True:
        area = count / angles.size
        radius = (area / np.pi)**.5
    else:
        radius = count
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='black', linewidth=1, color=color, alpha=.5)
    ax.set_theta_offset(offset)
    ax.set_yticks([])
    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                 r'$\pi=-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$']
        ax.set_xticklabels(label, fontsize=9)

def calculate_fold_180_0(angle_list):
    """
        Calculate fold(180/0)
    """
    num = {'180': 0, '0': 0}
    for angle in angle_list:
        angle = angle + 2*np.pi
        if (5*np.pi)/6 <= angle <= (7*np.pi)/6:
            num['180'] += 1
        elif angle <= np.pi/6 or angle >= (11*np.pi)/6:
            num['0'] += 1
    try:
        fold_180_0 = num['180'] / num['0']
    except:
        # if num['180'] == 0:
        #     fold_180_0 = 1
        # else:
        fold_180_0 = None
    # print('{:.2f} ({} / {})'.format(fold_180_0, num['180'], num['0']))
    return (fold_180_0, (num['180'], num['0']))

def plot_angles(path, data, par, fformat='png', dpi=1000):
    """
        Angle
    """
    angles = []
    for state, tracklet_list in enumerate(data.tracklet_lists):
        angles_state = np.array([])
        for tracklet in tracklet_list:
            angles_state = np.append(angles_state, tracklet.table['angle_1'][1:-1])
        angles.append(angles_state)
        # angles[state] = np.array(angles[state])[~pd.isnull(angles[state])].tolist()

    fig, axs = plt.subplots(1, par['num_states'], subplot_kw=dict(projection='polar'))
    for state in range(par['num_states']):
        rose_plot(axs[state], np.array(angles[state]), lab_unit="radians",
                  color=par['colors'][state]['name'])
    fig.tight_layout()
    plt.subplots_adjust(wspace=1)
    plt.savefig(path/'angle_circular.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

    labels = []
    for state in range(par['num_states']):
        # angles[state] = (np.array(angles[state])/(2*np.pi)).tolist()
        sns.distplot(np.array(angles[state]), color=par['colors'][state]['name'], kde_kws={"lw": 0},
                     hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1})
        labels.append(mlines.Line2D([], [], color=par['colors'][state]['name'],
                                    label='{}, {} = {:.3}'\
                                    .format(par['labels'][state], r'$f(180^o/0^o)$',
                                            calculate_fold_180_0(angles[state]+np.pi)[0])))
    plt.xlim(-np.pi, np.pi)
    plt.gca().set_xticks(np.arange(-np.pi, np.pi+0.01, np.pi/2))
    plt.gca().set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    plt.legend(handles=labels, loc='lower center')
    plt.xlabel(r'$angle$ $\mathrm{\theta}$', size='large')
    plt.ylabel(r'$probability$ $density$', size='large')
    plt.savefig(path/'angle_diagram.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_folds_180_0(path, data, par, step=50, lim=5, fformat='png', dpi=1000):
    """
        to do
    """
    os.makedirs(path/'angle_fold_plots', exist_ok=True)

    step_range = [i for i in np.arange(0, 800+step, step)]
    all_angle_distance = []
    all_fold_distance = []
    for state, tracklet_list in enumerate(data.tracklet_lists):
        angle_distance = [[] for _, _ in enumerate(step_range)]
        for tracklet in tracklet_list:
            D = ((tracklet.table['dist_1']*par['pixel_size']*1000)/step).astype(int)
            for p, d in enumerate(D):
                if d < len(step_range):
                    angle = tracklet.table['angle_1'][p]
                    angle_distance[d].append(angle)
        all_angle_distance.append(angle_distance)
        step_fold = []
        for i in angle_distance:
            fold = calculate_fold_180_0(i)[0]
            step_fold.append(fold)
        all_fold_distance.append(step_fold)

    _, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axhline(y=1, color='grey', linestyle=':')
    for state, _ in enumerate(mss_analysis):
        ax.plot(step_range, all_fold_distance[state], '-o', color=par['colors'][state]['name'],
                label=par['labels'][state], markeredgecolor='black')
    ax.set_xticklabels([i for i in range(-100, 900, 100)])
    plt.xlabel(r'$distance$ $[nm]$', labelpad=10, size='large')
    plt.ylabel(r'$fold\left(\frac{180^o\pm30^o}{0^o\pm30^o}\right)$', labelpad=10, size='large')
    ax.legend(loc='best', ncol=1)
    plt.tight_layout()
    plt.savefig(path/'angle_fold_plots'/'fold_distance.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()


    all_angle_lag_time = []
    all_fold_lagtime = []
    N = 10
    for state, mss in enumerate(mss_analysis):
        angle_lagtime = [[] for i in range(N)]
        for tracklet in mss.positive['tracklet']:
            if tracklet.num_frames >= N:
                L = tracklet.table['angle_1']
                for i, l in enumerate(L[:N]):
                    angle_lagtime[i].append(l)
        all_angle_lag_time.append(angle_lagtime)
        step_fold = []
        for i in angle_lagtime:
            step_fold.append(calculate_fold_180_0(i)[0])
        all_fold_lagtime.append(step_fold)

    _, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axhline(y=1, color='grey', linestyle=':')
    time = [i for i in range(0, N*7, 7)]
    for state, _ in enumerate(mss_analysis):
        ax.plot(time, all_fold_lagtime[state], '-o', color=par['colors'][state]['name'],
                label=par['labels'][state].format(state), markeredgecolor='black')
    # ax.set_xticklabels([i for i in range(-100, 900, 100)])
    plt.xlabel(r'$lag$ $time$ $[ms]$', labelpad=10, size='large')
    plt.ylabel(r'$fold\left(\frac{180^o\pm30^o}{0^o\pm30^o}\right)$', labelpad=10, size='large')
    ax.legend(loc='best', ncol=1)
    plt.tight_layout()
    plt.savefig(path/'angle_fold_plots'/'fold_lagtime.{}'.format(fformat), bbox_inches='tight', dpi=dpi)
    plt.close()

    cmap = cm.afmhot_r
    for state, mss in enumerate(mss_analysis):
        angle_displ = np.frompyfunc(list, 0, 1)(np.empty((len(step_range), len(step_range)),
                                                         dtype=object))
        for tracklet in mss.positive['tracklet']:
            D = ((tracklet.table['dist_1']*par['pixel_size']*1000)/step).astype(int)
            p = 1
            for i, j in zip(D[1:], D):
                if i < len(step_range) and j < len(step_range):
                    angle = tracklet.table['angle_1'][p]
                    angle_displ[i, j].append(angle)
                p += 1
        fold_array = np.empty((len(step_range), len(step_range)))
        num_180_array = np.empty((len(step_range), len(step_range)))
        num_0_array = np.empty((len(step_range), len(step_range))) 
        for i, _ in enumerate(step_range):
            for j, _ in enumerate(step_range):
                results = calculate_fold_180_0(angle_displ[i, j])
                num_180_array[i, j] = results[1][0]
                num_0_array[i, j] = results[1][1]
                fold_array[i, j] = results[0]

        im = plt.imshow(num_180_array, cmap=cmap, extent=[0, 800, 800, 0], interpolation=None, alpha=0.5)
        fig = im.get_figure()
        ax = fig.get_axes()[0]
        plt.grid(None)
        plt.xlabel(r'$second$ $displacement$ $[nm]$')
        plt.ylabel(r'$first$ $displacement$ $[nm]$')
        clb = plt.colorbar(im)
        for (j,i),label in np.ndenumerate(num_180_array):
            ax.text(i/num_180_array.shape[0]*800+25, j/num_180_array.shape[0]*800+25, int(label), ha='center', va='center')
        ax.set_title(r'$Number$ $of$ $angles$ $180^o\pm30^o$')
        plt.savefig(path/'angle_fold_plots'/'fold_displacement_state_{}_num180.{}'.format(state, fformat),
                    bbox_inches='tight')
        plt.close()

        im = plt.imshow(num_0_array, cmap=cmap, extent=[0, 800, 800, 0], interpolation=None, alpha=0.5)
        fig = im.get_figure()
        ax = fig.get_axes()[0]
        plt.grid(None)
        plt.xlabel(r'$second$ $displacement$ $[nm]$')
        plt.ylabel(r'$first$ $displacement$ $[nm]$')
        clb = plt.colorbar(im)
        for (j,i),label in np.ndenumerate(num_0_array):
            ax.text(i/num_0_array.shape[0]*800+25, j/num_0_array.shape[0]*800+25, int(label), ha='center', va='center')
        ax.set_title(r'$Number$ $of$ $angles$ $0^o\pm30^o$')
        plt.savefig(path/'angle_fold_plots'/'fold_displacement_state_{}_num0.{}'.format(state, fformat),
                    bbox_inches='tight')
        plt.close()

        im = plt.imshow(fold_array, cmap=cmap, extent=[0, 800, 800, 0], interpolation=None)
        plt.grid(None)
        plt.xlabel(r'$second$ $displacement$ $[nm]$')
        plt.ylabel(r'$first$ $displacement$ $[nm]$')
        clb = plt.colorbar(im)
        clb.ax.set_title(r'$fold\left(\frac{180^o\pm30^o}{0^o\pm30^o}\right)$')
        plt.savefig(path/'angle_fold_plots'/'fold_displacement_state_{}.{}'.format(state, fformat),
                    bbox_inches='tight', dpi=dpi)
        plt.close()

        # im = plt.imshow(fold_array, cmap=cmap, extent=[0, 800, 800, 0], interpolation=None)
        # plt.grid(None)
        # plt.xlabel(r'$second$ $displacement$ $[nm]$')
        # plt.ylabel(r'$first$ $displacement$ $[nm]$')
        # clb = plt.colorbar(im)
        # clb.ax.set_title(r'$fold(180^o/0^o)$')
        # plt.clim(0, lim)
        # plt.savefig(path/'angle_fold_displacement_state_{}_lim.{}'.format(state, fformat),
        #             bbox_inches='tight', dpi=dpi)
        # plt.close()
