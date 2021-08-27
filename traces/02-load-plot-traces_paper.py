# %%
# imports
import pacsltk
from IPython.display import display
from matplotlib.ticker import NullFormatter, ScalarFormatter
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import timedelta

import pacsltk.pacs_util as pacs_util
import pandas as pd
import numpy as np

# prepare for plots
pacs_util.prepare_matplotlib_cycler()
# To avoid type 3 fonts: http://phyletica.org/matplotlib-fonts/
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
register_matplotlib_converters()

import math
import matplotlib.ticker as mticker


print(pacsltk.__version__)

# %%
# Try loading autoscale data
autoscale_folder_path = './files/AutoScale/'
figs_folder = './figs/'
autoscale_trace_file_names = [
    # Big Spike, NLANR [nlanr1995]
    'trace_t2.txt',
    # Dual Phase, NLANR [nlanr1995]
    'trace_t4.txt',
    # Large variations, NLANR [nlanr1995]
    'trace_t5.txt',
    # worldcup, slowly varying [ita 1998]
    'trace_wc.txt',
]


def extract_raw_exp_name(v):
    return v.replace('.txt', '')


def get_plot_name(v):
    if '_t2' in v:
        return 'AS NLAR T2'
    elif '_t4' in v:
        return 'AS NLAR T4'
    elif '_t5' in v:
        return 'AS NLAR T5'
    elif '_wc' in v:
        return 'AS FIFA WC'
    return v.replace('.txt', '').replace('_', '')


def save_fig(figname, raw_exp_name, default_margins=True):
    # same margins for all so they fall in line with each other
    if default_margins:
        plt.gcf().subplots_adjust(left=0.14, bottom=0.17)
    plt.savefig(
        figs_folder + f"{raw_exp_name}_{figname}" + ".png", dpi=600)
    plt.savefig(figs_folder + f"{raw_exp_name}_{figname}" + ".pdf")


def fix_x_axis_timedelta(timescale=1e9):
    def timeTicks(x, pos):
        remaining_seconds = x / timescale
        hours = math.floor(remaining_seconds / 60 / 60)
        remaining_seconds -= hours * 60 * 60
        minutes = math.floor(remaining_seconds / 60)
        remaining_seconds -= minutes * 60
        seconds = math.floor(remaining_seconds)
        return f"{hours:02d}:{minutes:02d}"

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(base=20*60*1e9))
    plt.gca().xaxis.set_minor_locator(mticker.MultipleLocator(base=5*60*1e9))
    formatter = mticker.FuncFormatter(timeTicks)
    plt.gca().xaxis.set_major_formatter(formatter)


# plot the autoscale files
for trace_fname in autoscale_trace_file_names:
    # get the file path
    curr_file_path = autoscale_folder_path + trace_fname
    print('loading file:', curr_file_path)

    # load the file
    trace_arr = np.loadtxt(curr_file_path)
    trace_arr = trace_arr / trace_arr.max() * 100
    trace_len = len(trace_arr)
    time_since_start = [timedelta(seconds=i*60) for i in range(trace_len)]
    data = {
        'time_since_start': time_since_start,
        'arrival_rate': trace_arr,
    }
    df = pd.DataFrame(data=data)
    df.set_index('time_since_start', inplace=True)

    trace_name = get_plot_name(trace_fname)

    plt.figure(figsize=(4, 2.5))
    plt.plot(df['arrival_rate'])
    # plt.title(trace_name)
    fix_x_axis_timedelta()

    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Arrival Rate (req/s)')
    plt.grid()

    raw_exp_name = extract_raw_exp_name(trace_fname)
    save_fig('arrival_rate', raw_exp_name)
    print(raw_exp_name)
# %%
