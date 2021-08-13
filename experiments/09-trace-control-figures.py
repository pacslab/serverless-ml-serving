# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import matplotlib.ticker as mticker
from numpy.lib.function_base import disp
import pacsltk
from IPython.display import display
from matplotlib.ticker import NullFormatter, ScalarFormatter
import seaborn as sns
from scipy import stats
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import pacsltk.pacs_util as pacs_util
from datetime import timedelta
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# prepare for plots
pacs_util.prepare_matplotlib_cycler()
# To avoid type 3 fonts: http://phyletica.org/matplotlib-fonts/
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
register_matplotlib_converters()


print(pacsltk.__version__)

# %%
# Prepare for timedelta fix in plots


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

def fix_log_x_plot():
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

def fix_log_y_plot():
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# %%
# service_name = 'bentoml-keras-toxic-comments'
# exp_name = 'res-2021-06-28_22-09-08'
# slo_timeout = 500
# trace_name = 'trace2'

service_name = 'bentoml-pytorch-fashion-mnist'
trace_name = 'trace_trace_wc'
exp_name = 'res-2021-07-29_23-08-36_proxy'
exp_no_proxy_name = 'res-2021-07-29_18-18-28_proxy_no_controller'
slo_timeout = 1000


# service_name = 'bentoml-iris'
# trace_name = 'trace_trace_wc'
# experiments
# exp_name = 'res-2021-08-03_11-46-33_proxy' # max 50
# exp_name = 'res-2021-08-03_19-10-32_proxy'
# exp_name = 'res-2021-08-04_11-35-40_proxy'
# slo_timeout = 500
# exp_name = 'res-2021-08-04_15-54-03_proxy'
# exp_name = 'res-2021-08-04_18-12-27_proxy' # max 200
# slo_timeout = 200
# no proxy experiment
# exp_no_proxy_name = 'res-2021-08-03_16-39-31_proxy_no_controller' # max 50

# %%


def get_exp_paths(**kwargs):
    trace_name = kwargs.get('trace_name')
    service_name = kwargs.get('service_name')
    exp_name = kwargs.get('exp_name')

    exp_folder = f'results/{trace_name}/{service_name}'
    exp_proxy_filename = f'{exp_folder}/{exp_name}.csv'
    exp_reqs_filename = exp_proxy_filename.replace('proxy', 'reqs')

    return {
        'exp_folder': exp_folder,
        'exp_proxy_filename': exp_proxy_filename,
        'exp_reqs_filename': exp_reqs_filename,
    }


ret = get_exp_paths(**locals())
locals().update(ret)


# %%

def preprocess_exp_logs(**kwargs):
    exp_reqs_filename = kwargs.get('exp_reqs_filename')
    exp_proxy_filename = kwargs.get('exp_proxy_filename')

    df_proxy_stats = pd.read_csv(
        exp_proxy_filename, index_col=0, parse_dates=True)
    df_res = pd.read_csv(exp_reqs_filename, index_col=0, parse_dates=True)

    df_res['received_at'] = pd.to_datetime(df_res['received_at'])
    df_res['response_at'] = pd.to_datetime(df_res['response_at'])

    return {
        'df_proxy_stats': df_proxy_stats,
        'df_res': df_res,
    }


ret = preprocess_exp_logs(**locals())
locals().update(ret)

display(df_proxy_stats.head())
display(df_res.head())

# %%


def fix_log_time_since_start(**kwargs):
    df_proxy_stats = kwargs.get('df_proxy_stats')
    df_res = kwargs.get('df_res')

    proxy_stats_count = df_proxy_stats.shape[0]
    time_since_start = [timedelta(seconds=i*30)
                        for i in range(proxy_stats_count)]
    df_proxy_stats['time_since_start'] = time_since_start
    df_proxy_stats.set_index('time_since_start', inplace=True)

    df_res['time_since_start'] = df_res['received_at'] - \
        df_res['received_at'].min()
    df_res = df_res.set_index('time_since_start')

    # remove first and last 5 minutes
    df_proxy_stats = df_proxy_stats.loc[df_proxy_stats.index > timedelta(
        seconds=5 * 60), :]
    df_proxy_stats = df_proxy_stats.loc[df_proxy_stats.index <
                                        df_proxy_stats.index.max() - timedelta(seconds=5 * 60), :]
    df_res = df_res.loc[df_res.index > timedelta(seconds=5 * 60), :]
    df_res = df_res.loc[df_res.index <
                        df_res.index.max() - timedelta(seconds=5 * 60), :]

    df_res_resample = df_res.resample('T')

    return {
        'df_proxy_stats': df_proxy_stats,
        'df_res': df_res,
        'df_res_resample': df_res_resample,
    }


ret = fix_log_time_since_start(**locals())
locals().update(ret)

display(df_proxy_stats.head())
display(df_res.head())

# %%
plt.plot(df_proxy_stats['averageArrivalRate'])
fix_x_axis_timedelta()

# %%


def print_val_keys(**kwargs):
    vals = [k for k in kwargs]
    print(vals)
    return {}

def extract_no_control_params(**kwargs):
    # list of parameters to be extracted
    params_list = [
        'df_proxy_stats',
        'df_res',
        'df_res_resample',
    ]
    ret = {f'{k}_no_proxy': kwargs.get(k) for k in params_list}
    # update exp name to use main experiment now
    ret.update({
        'exp_name': kwargs.get('exp_name_main')
    })
    return ret

def plot_over_time_both(**kwargs):
    df_proxy_stats = kwargs.get('df_proxy_stats')
    df_proxy_stats_no_proxy = kwargs.get('df_proxy_stats_no_proxy')

    df_res = kwargs.get('df_res')
    df_res_no_proxy = kwargs.get('df_res_no_proxy')

    df_res_resample = kwargs.get('df_res_resample')
    df_res_resample_no_proxy = kwargs.get('df_res_resample_no_proxy')
    df_res_resample_mean = df_res_resample.mean()
    df_res_resample_no_proxy_mean = df_res_resample_no_proxy.mean()

    slo_timeout = kwargs.get('slo_timeout')

    plt.figure()
    plt.plot(df_proxy_stats['averageArrivalRate'])
    plt.plot(df_proxy_stats_no_proxy['averageArrivalRate'])
    fix_x_axis_timedelta()

    plt.figure()
    plt.plot(df_proxy_stats['reponseTimeP95'], label='RT95 (*)')
    plt.plot(df_proxy_stats_no_proxy['reponseTimeP95'], label='RT95')
    plt.axhline(y=slo_timeout, ls='--', c='r', label='SLA Timeout')
    fix_x_axis_timedelta()
    plt.legend()

    plt.figure()
    plt.plot(df_proxy_stats['currentReadyReplicaCount'], label='# of Cont (*)')
    plt.plot(df_proxy_stats_no_proxy['currentReadyReplicaCount'], label='# of Cont')
    fix_x_axis_timedelta()
    plt.legend()

    plt.figure()
    plt.plot(df_res_resample_mean['upstream_request_count'], label='Batch Size (*)')
    # plt.plot(df_res_resample_no_proxy_mean['upstream_request_count'], label='Batch Size')
    fix_x_axis_timedelta()
    plt.legend()

    # slo miss rates with proxy
    slo_miss_count = df_res_resample['response_time_ms_server'].apply(lambda x: np.sum(x > slo_timeout))
    resampled_request_count = df_res_resample['response_time_ms_server'].apply(lambda x: len(x))
    slo_miss_rates = slo_miss_count / resampled_request_count * 100
    slo_miss_rates_all = slo_miss_count.sum() / resampled_request_count.sum() * 100
    # slow miss rates without proxy
    slo_miss_count_no_proxy = df_res_resample_no_proxy['response_time_ms_server'].apply(lambda x: np.sum(x > slo_timeout))
    resampled_request_count_no_proxy = df_res_resample_no_proxy['response_time_ms_server'].apply(lambda x: len(x))
    slo_miss_rates_no_proxy = slo_miss_count_no_proxy / resampled_request_count_no_proxy * 100
    slo_miss_rates_no_proxy_all = slo_miss_count_no_proxy.sum() / resampled_request_count_no_proxy.sum() * 100
    # plot the slo miss rates
    plt.figure()
    plt.plot(slo_miss_rates, label="SLO Miss (*)")
    plt.plot(slo_miss_rates_no_proxy, label="SLO Miss")
    plt.axhline(y=5, ls='--', c='r')
    plt.ylabel('SLO Miss Rate (%)')
    plt.xlabel('Time (HH:MM)')
    fix_x_axis_timedelta()

    
    # ccdf min and max value
    ccdf_max_value = max(
        df_res['response_time_ms_server'].max(),
        df_res_no_proxy['response_time_ms_server'].max()
    )
    ccdf_max_value = min(ccdf_max_value, slo_timeout*2)
    ccdf_max_value = max(ccdf_max_value, slo_timeout*1.1)
    ccdf_min_value = min(
        df_res['response_time_ms_server'].min(),
        df_res_no_proxy['response_time_ms_server'].min()
    )
    # ccdf with proxy
    global ccdf_values
    global ccdf_freqs
    ccdf_values = np.linspace(ccdf_min_value, ccdf_max_value, 100)
    ccdf_freqs = pd.Series(ccdf_values).apply(lambda x: np.sum(df_res['response_time_ms_server'] < x))
    ccdf_freqs = 100 - (ccdf_freqs / ccdf_freqs.iloc[-1] * 100)
    ccdf_slo_loc = np.where(ccdf_values > slo_timeout)[0][0]
    ccdf_slo_tick = ccdf_freqs[ccdf_slo_loc]
    # ccdf without proxy
    ccdf_values_no_proxy = ccdf_values
    ccdf_freqs_no_proxy = pd.Series(ccdf_values_no_proxy).apply(lambda x: np.sum(df_res_no_proxy['response_time_ms_server'] < x))
    ccdf_freqs_no_proxy = 100 - (ccdf_freqs_no_proxy / ccdf_freqs_no_proxy.iloc[-1] * 100)
    ccdf_slo_loc_no_proxy = np.where(ccdf_values_no_proxy > slo_timeout)[0][0]
    ccdf_slo_tick_no_proxy = ccdf_freqs_no_proxy[ccdf_slo_loc_no_proxy]
    # plotting ccdf
    plt.figure(figsize=(4,2.5))
    plt.plot(ccdf_values, ccdf_freqs)
    plt.loglog(ccdf_values_no_proxy, ccdf_freqs_no_proxy)
    plt.axvline(x=slo_timeout, ls='--', c='r', lw=1)
    plt.xlim([ccdf_min_value*0.8, ccdf_max_value*1.2])
    plt.hlines(y=ccdf_slo_tick, xmin=0, xmax=slo_timeout, ls='--', color='k', lw=1)
    plt.hlines(y=ccdf_slo_tick_no_proxy, xmin=0, xmax=slo_timeout, ls='--', color='k', lw=1)
    fix_log_x_plot()
    fix_log_y_plot()
    plt.ylabel('CCDF (%)')
    plt.xlabel('Latency (ms)')
    plt.yticks([ccdf_slo_tick, ccdf_slo_tick_no_proxy, 100])

    # generate the stats needed for the paper
    exp_stats = {
        'ML Proxy': [True, False],
        'SLO Miss Rate': [
            slo_miss_rates_all,
            slo_miss_rates_no_proxy_all
        ]
    }
    exp_stats_df = pd.DataFrame(data=exp_stats)
    exp_stats_df['SLO Miss Improvement'] = 100 - exp_stats_df['SLO Miss Rate'] / slo_miss_rates_no_proxy_all * 100
    display(exp_stats_df.T)

    return {}

processing_pipeline = [
    # parsing the no control experiment
    ('Get Experiment Paths', get_exp_paths),
    ('Preprocess Experiment Logs', preprocess_exp_logs),
    ('Fix Log Timings', fix_log_time_since_start),
    ('Extract No Control Parameters', extract_no_control_params),
    # parsing the main experiment
    ('Get Experiment Paths', get_exp_paths),
    ('Preprocess Experiment Logs', preprocess_exp_logs),
    ('Fix Log Timings', fix_log_time_since_start),
    # print all keys
    ('Print Keys', print_val_keys),
    ('Plot For Both Proxy/Non-Proxy', plot_over_time_both),
]


def process_exp_path(vals, processing_pipeline):
    service_name = vals.get('service_name')
    trace_name = vals.get('trace_name')

    print('='*20, f'{service_name}({trace_name})', '='*20)

    inter = vals
    for step_name, step_func in processing_pipeline:
        print(f'[*] {step_name}')
        inter_update = step_func(**inter)
        inter.update(inter_update)
    return inter

# rel_avg_resp_time_all = None
# for exp_path in exp_paths:
#     vals = {
#         'exp_path': exp_path,
#         'service_name': service_name,
#         'trace_name': trace_name,
#     }


vals = {
    'exp_name_main': exp_name,
    'exp_name_no_proxy': exp_no_proxy_name,
    'exp_name': exp_no_proxy_name,
    'service_name': service_name,
    'trace_name': trace_name,
    'slo_timeout': slo_timeout,
    'exp_no_proxy_name': exp_no_proxy_name,
}
_ = process_exp_path(vals, processing_pipeline)

# %%
