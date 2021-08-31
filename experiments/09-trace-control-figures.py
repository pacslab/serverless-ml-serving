# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import matplotlib.ticker as mticker
import pacsltk
from IPython.display import display
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
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
trace_configs = {
    # trace_trace_wc ----------------------------------------------

    'trace_trace_wc': {
        'iris_max50': {
            'service_name': 'bentoml-iris',
            'trace_name': 'trace_trace_wc',
            'exp_name': 'res-2021-08-03_11-46-33_proxy',
            'exp_no_proxy_name': 'res-2021-08-03_16-39-31_proxy_no_controller',
            'slo_timeout': 500,
        },
        'iris_max200': {
            'service_name': 'bentoml-iris',
            'trace_name': 'trace_trace_wc',
            'exp_name': 'res-2021-08-04_18-12-27_proxy',
            'exp_no_proxy_name': 'res-2021-08-13_19-37-35_proxy_no_controller',
            'slo_timeout': 200,
        },
        'fashion_mnist_max30': {
            'service_name': 'bentoml-pytorch-fashion-mnist',
            'trace_name': 'trace_trace_wc',
            'exp_name': 'res-2021-07-29_23-08-36_proxy',
            'exp_no_proxy_name': 'res-2021-07-29_18-18-28_proxy_no_controller',
            'slo_timeout': 1000,
        },
        'fashion_mnist_max100': {
            'service_name': 'bentoml-pytorch-fashion-mnist',
            'trace_name': 'trace_trace_wc',
            'exp_name': 'res-2021-08-18_15-32-56_proxy',
            'exp_no_proxy_name': 'res-2021-08-18_20-13-19_proxy_no_controller',
            'slo_timeout': 1000,
        },
        'toxic_comments_max50': {
            'service_name': 'bentoml-keras-toxic-comments',
            'trace_name': 'trace_trace_wc',
            'exp_name': 'res-2021-07-30_17-38-19_proxy',
            'exp_no_proxy_name': 'res-2021-07-30_15-11-01_proxy_no_controller',
            'slo_timeout': 500,
        },
    },

    # trace_trace_t5 ----------------------------------------------

    'trace_trace_t5': {
        'iris_max200': {
            'service_name': 'bentoml-iris',
            'trace_name': 'trace_trace_t5',
            'exp_name': 'res-2021-08-10_12-24-59_proxy',
            'exp_no_proxy_name': 'res-2021-08-06_16-33-09_proxy_no_controller',
            'slo_timeout': 200,
        },
        'iris_max200_2': {
            'service_name': 'bentoml-iris',
            'trace_name': 'trace_trace_t5',
            'exp_name': 'res-2021-08-10_16-29-55_proxy',
            'exp_no_proxy_name': 'res-2021-08-06_16-33-09_proxy_no_controller',
            'slo_timeout': 500,
        },
        'toxic_comments_max50': {
            'service_name': 'bentoml-keras-toxic-comments',
            'trace_name': 'trace_trace_t5',
            'exp_name': 'res-2021-08-10_19-21-54_proxy',
            'exp_no_proxy_name': 'res-2021-08-11_13-25-09_proxy_no_controller',
            'slo_timeout': 500,
        },
        'fashion_mnist_max30': {
            'service_name': 'bentoml-pytorch-fashion-mnist',
            'trace_name': 'trace_trace_t5',
            'exp_name': 'res-2021-08-11_17-30-40_proxy',
            'exp_no_proxy_name': 'res-2021-08-11_19-47-33_proxy_no_controller',
            'slo_timeout': 1000,
        },
    },

    # trace_trace_t4 ----------------------------------------------

    'trace_trace_t4': {
        'fashion_mnist_max100': {
            'service_name': 'bentoml-pytorch-fashion-mnist',
            'trace_name': 'trace_trace_t4',
            'exp_name': 'res-2021-08-17_17-21-00_proxy',
            'exp_no_proxy_name': 'res-2021-08-17_19-45-56_proxy_no_controller',
            'slo_timeout': 1000,
        },
        'iris_max200': {
            'service_name': 'bentoml-iris',
            'trace_name': 'trace_trace_t4',
            'exp_name': 'res-2021-08-26_01-01-31_proxy',
            'exp_no_proxy_name': 'res-2021-08-25_18-42-06_proxy_no_controller',
            'slo_timeout': 200,
        },
        'toxic_comments_max50': {
            'service_name': 'bentoml-keras-toxic-comments',
            'trace_name': 'trace_trace_t4',
            'exp_name': 'res-2021-08-26_16-16-15_proxy',
            'exp_no_proxy_name': 'res-2021-08-26_11-29-56_proxy_no_controller',
            'slo_timeout': 500,
        },
    },
}

selected_trace_name = 'trace_trace_t4'
selected_config = 'toxic_comments_max50'

configs = trace_configs[selected_trace_name]
config = configs[selected_config]
service_name = config['service_name']
trace_name = config['trace_name']
exp_name = config['exp_name']
exp_no_proxy_name = config['exp_no_proxy_name']
slo_timeout = config['slo_timeout']


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

    config_name = kwargs.get('config_name')
    figs_folder = kwargs.get('figs_folder')

    def save_fig(figname, default_margins=True):
        # same margins for all so they fall in line with each other
        if default_margins:
            plt.gcf().subplots_adjust(left=0.20, bottom=0.17)
        plt.savefig(figs_folder + f"{config_name}_{figname}" + ".png", dpi=600)
        plt.savefig(figs_folder + f"{config_name}_{figname}" + ".pdf")

    slo_timeout = kwargs.get('slo_timeout')

    plt.figure(figsize=(4,2.5))
    plt.plot(df_proxy_stats['averageArrivalRate'], label='Arrival Rate (*)')
    plt.plot(df_proxy_stats_no_proxy['averageArrivalRate'], label='Arrival Rate')
    fix_x_axis_timedelta()
    plt.legend()
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Arrival Rate (req/s)')
    plt.grid()
    save_fig('arrival_rate')

    plt.figure(figsize=(4,2.5))
    plt.plot(df_proxy_stats['reponseTimeP95'], label='RT95 (*)')
    plt.plot(df_proxy_stats_no_proxy['reponseTimeP95'], label='RT95')
    plt.axhline(y=slo_timeout, ls='--', c='r', label='SLO Timeout')
    fix_x_axis_timedelta()
    plt.legend()
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('P95 of Resp Time (ms)')
    plt.grid()
    save_fig('rt95')
    

    plt.figure(figsize=(4,2.5))
    plt.plot(df_proxy_stats['currentReadyReplicaCount'], label='# of Cont (*)')
    plt.plot(df_proxy_stats_no_proxy['currentReadyReplicaCount'], label='# of Cont')
    fix_x_axis_timedelta()
    plt.legend()
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('# of Containers')
    plt.grid()
    save_fig('num_of_cont')

    plt.figure(figsize=(4,2.5))
    plt.plot(df_res_resample_mean['upstream_request_count'], label='Batch Size (*)')
    # plt.plot(df_res_resample_no_proxy_mean['upstream_request_count'], label='Batch Size')
    fix_x_axis_timedelta()
    plt.legend()
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Average Batch Size')
    plt.grid()
    save_fig('batch_size')

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
    plt.figure(figsize=(4,2.5))
    plt.plot(slo_miss_rates, label="SLO Miss (*)")
    plt.plot(slo_miss_rates_no_proxy, label="SLO Miss")
    plt.axhline(y=5, ls='--', c='r', label='SLO Thresh')
    plt.ylabel('SLO Miss Rate (%)')
    plt.xlabel('Time (HH:MM)')
    fix_x_axis_timedelta()
    plt.legend()
    plt.grid()
    save_fig('slo_miss_rate')

    
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
    ccdf_freqs = 100 - (ccdf_freqs / df_res['response_time_ms_server'].shape[0] * 100)
    ccdf_slo_loc = np.where(ccdf_values > slo_timeout)[0][0]
    ccdf_slo_tick = ccdf_freqs[ccdf_slo_loc]
    # ccdf without proxy
    ccdf_values_no_proxy = ccdf_values
    ccdf_freqs_no_proxy = pd.Series(ccdf_values_no_proxy).apply(lambda x: np.sum(df_res_no_proxy['response_time_ms_server'] < x))
    ccdf_freqs_no_proxy = 100 - (ccdf_freqs_no_proxy / df_res_no_proxy['response_time_ms_server'].shape[0] * 100)
    ccdf_slo_loc_no_proxy = np.where(ccdf_values_no_proxy > slo_timeout)[0][0]
    ccdf_slo_tick_no_proxy = ccdf_freqs_no_proxy[ccdf_slo_loc_no_proxy]
    # plotting ccdf
    plt.figure(figsize=(4,2.5))
    plt.plot(ccdf_values, ccdf_freqs, label='CCDF (*)')
    plt.loglog(ccdf_values_no_proxy, ccdf_freqs_no_proxy, label='CCDF')
    plt.axvline(x=slo_timeout, ls='--', c='r', lw=1)
    plt.xlim([ccdf_min_value*0.8, ccdf_max_value*1.2])
    plt.hlines(y=ccdf_slo_tick, xmin=0, xmax=slo_timeout, ls='--', color='k', lw=1)
    plt.hlines(y=ccdf_slo_tick_no_proxy, xmin=0, xmax=slo_timeout, ls='--', color='k', lw=1)
    fix_log_x_plot()
    fix_log_y_plot()
    plt.ylabel('CCDF (%)')
    plt.xlabel('Latency (ms)')
    yticks = [100, ccdf_slo_tick]
    # if the two ticks are further than some percentage away
    if np.abs((ccdf_slo_tick_no_proxy - ccdf_slo_tick) / ccdf_slo_tick) * 100 > 5:
        yticks += [ccdf_slo_tick_no_proxy]
    plt.yticks(yticks)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    save_fig('ccdf')

    # generate the stats needed for the paper
    average_replica_count = df_proxy_stats['currentReadyReplicaCount'].mean()
    average_replica_count_no_proxy = df_proxy_stats_no_proxy['currentReadyReplicaCount'].mean()
    exp_stats = {
        'ML Proxy': [True, False],
        'SLO Miss Rate': [
            slo_miss_rates_all,
            slo_miss_rates_no_proxy_all
        ],
        'Max RPS': [
            df_proxy_stats['averageArrivalRate'].max(),
            df_proxy_stats_no_proxy['averageArrivalRate'].max(),
        ],
        'Average Replica Count': [
            average_replica_count,
            average_replica_count_no_proxy,
        ],
        # 'Average Batch Size (over reqs)': [
        #     df_res_resample_mean['upstream_request_count'].mean(),
        #     1,
        # ],
        'Average Batch Size (over time)': [
            df_proxy_stats['averageActualBatchSize'].mean(),
            1
        ],
        'Average Max Buffer Size': [
            df_proxy_stats['averageMaxBufferSize'].mean(),
            1,
        ],
        'Average RT95': [
            df_proxy_stats['reponseTimeP95'].mean(),
            df_proxy_stats_no_proxy['reponseTimeP95'].mean(),
        ],
    }
    exp_stats_df = pd.DataFrame(data=exp_stats)
    exp_stats_df['SLO Miss Improvement'] = 100 - exp_stats_df['SLO Miss Rate'] / slo_miss_rates_no_proxy_all * 100
    exp_stats_df['Average Replica Improvement'] = 100 - exp_stats_df['Average Replica Count'] / average_replica_count_no_proxy * 100
    exp_stats_df = exp_stats_df.T
    exp_stats_df.to_csv(figs_folder + f"{config_name}_summary.csv")
    display(exp_stats_df)

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

for t in trace_configs:
    configs = trace_configs[t]
    for k in configs:
        config_name = k
        config = configs[k]
        exp_name = config['exp_name']
        exp_no_proxy_name = config['exp_no_proxy_name']
        service_name = config['service_name']
        trace_name = config['trace_name']
        slo_timeout = config['slo_timeout']
        print('Config Name:', config_name)

        figs_folder = f"./figs/{trace_name}/"
        # create the figs folder
        get_ipython().system('mkdir -p {figs_folder}')

        vals = {
            'config_name': config_name,
            'figs_folder': figs_folder,
            'exp_name_main': exp_name,
            'exp_name_no_proxy': exp_no_proxy_name,
            'exp_name': exp_no_proxy_name,
            'service_name': service_name,
            'trace_name': trace_name,
            'slo_timeout': slo_timeout,
            'exp_no_proxy_name': exp_no_proxy_name,
        }
        process_exp_path(vals, processing_pipeline)

# %%
