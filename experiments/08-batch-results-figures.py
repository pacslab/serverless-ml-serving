# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import glob
import json
from datetime import timedelta

import pacsltk.pacs_util as pacs_util
import pandas as pd
import numpy as np

# prepare for plots
pacs_util.prepare_matplotlib_cycler()
# To avoid type 3 fonts: http://phyletica.org/matplotlib-fonts/
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy import stats
import seaborn as sns

from matplotlib.ticker import NullFormatter, ScalarFormatter

from IPython.display import display

import pacsltk
print(pacsltk.__version__)

# %%
exp_group_name = 'batch_experiments_default'
exp_folder = f"./results/{exp_group_name}/"
figs_folder = f"./figs/{exp_group_name}/"

# create the figs folder
get_ipython().system('mkdir -p {figs_folder}')

exp_paths = glob.glob(f'{exp_folder}/*.csv')
exp_paths


# %%
def extract_raw_exp_name(v):
    return v.split('/')[-1].split('.')[0]

def get_exp_name(v):
    if 'tfserving-mobilenetv1' in v:
        return 'MobileNet-v1'
    elif 'bentoml-iris' in v:
        return 'Iris'
    elif 'tfserving-resnetv2' in v:
        return 'ResNet-v2'
    elif 'bentoml-onnx-resnet50_' in v:
        return 'ResNet50-v2'
    elif 'bentoml-pytorch-fashion-mnist' in v:
        return 'Fashion MNIST'
    elif 'bentoml-keras-toxic-comments' in v:
        return 'Toxic Comments'
    else:
        return v.split('/')[-1].split('_')[0]

display([get_exp_name(v) for v in exp_paths])
display([extract_raw_exp_name(v) for v in exp_paths])


# %%
# the experiment path that we will use for developing our plots here
# exp_path = [f for f in exp_paths if 'onnx-resnet50' in f][0]
exp_path = exp_paths[0]
print(exp_path)
exp_name = get_exp_name(exp_path)
print(exp_name)
raw_exp_name = extract_raw_exp_name(exp_path)
print(raw_exp_name)

# processing the csv file to extract info
def post_process(df):
    convert_resp_time_lambda = lambda row: np.array(json.loads(row['response_times_ms'])).astype(np.float)
    df['response_times_ms'] = df.apply(convert_resp_time_lambda, axis=1)
    df['resp_time_avg'] = df.apply(lambda x: np.mean(x['response_times_ms']), axis=1)
    for percentile in [5,50,90,95,99]:
        df[f'resp_time_p{percentile}'] = df.apply(lambda x: np.percentile(x['response_times_ms'], percentile), axis=1)
    df[f'request_count'] = df.apply(lambda x: len(x['response_times_ms']), axis=1)
    return df

df = pd.read_csv(exp_path, index_col=0, parse_dates=True)
df = post_process(df)
df


# %%
# prepare for plotting
def save_fig(figname, raw_exp_name):
    if raw_exp_name is not None:
        plt.savefig(figs_folder + f"{raw_exp_name}_{figname}" + ".png", dpi=600)
        plt.savefig(figs_folder + f"{raw_exp_name}_{figname}" + ".pdf")

# fixed x log plots
def fix_log_x_plot():
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

# %%
def prepare_batch_plot_info(**kwargs):
    df = kwargs.get('df')
    raw_exp_name = kwargs.get('raw_exp_name')

    base_row = df.loc[df['batch_size']==1,:].iloc[0]
    exp_name = kwargs.get('exp_name')

    rel_avg_resp_time = df.loc[:,['batch_size', 'resp_time_avg']]
    rel_avg_resp_time['rel_resp_time_avg'] = df['resp_time_avg'] / base_row['resp_time_avg']

    rel_avg_resp_time['time_per_inference'] = rel_avg_resp_time['resp_time_avg'] / rel_avg_resp_time['batch_size']
    rel_avg_resp_time['rel_time_per_inference'] = rel_avg_resp_time['time_per_inference'] / base_row['resp_time_avg']


    return {
        'rel_avg_resp_time': rel_avg_resp_time
    }

ret = prepare_batch_plot_info(df=df, exp_name=exp_name, raw_exp_name=raw_exp_name)
rel_avg_resp_time = ret['rel_avg_resp_time']
display(rel_avg_resp_time)

# %%

rel_avg_resp_time_all = None
def stack_batch_plot_info(**kwargs):
    global rel_avg_resp_time_all

    raw_exp_name = kwargs.get('raw_exp_name')
    exp_name = kwargs.get('exp_name')
    rel_avg_resp_time = kwargs.get('rel_avg_resp_time')

    rel_avg_resp_time['raw_exp_name'] = raw_exp_name
    rel_avg_resp_time['exp_name'] = exp_name

    if rel_avg_resp_time_all is None:
        rel_avg_resp_time_all = rel_avg_resp_time
    else:
        rel_avg_resp_time_all = rel_avg_resp_time_all.append(rel_avg_resp_time)

    return kwargs
    

_ = stack_batch_plot_info(df=df, exp_name=exp_name, raw_exp_name=raw_exp_name, rel_avg_resp_time=rel_avg_resp_time)


# %%
processing_pipeline = [
    ('Extract Experiment Name', lambda vals: {
        'exp_name': get_exp_name(vals['exp_path']),
        'raw_exp_name': extract_raw_exp_name(vals['exp_path']),
    }),
    ('Read CSV', lambda vals: {'df': pd.read_csv(vals['exp_path'], index_col=0, parse_dates=True)}),
    ('Post Processing', lambda vals: {'df': post_process(vals['df'])} ),
    ('Prepare Batch Plot Info', lambda vals: prepare_batch_plot_info(**vals)),
    ('Stack Batch Plot Info', lambda vals: stack_batch_plot_info(**vals)),
]

def process_exp_path(vals, processing_pipeline):
    exp_path = vals['exp_path']
    print('='*20, extract_raw_exp_name(exp_path), '='*20)

    inter = vals
    for step_name, step_func in processing_pipeline:
        print(f'[*] {step_name}')
        inter_update = step_func(inter)
        inter.update(inter_update)
    return inter

rel_avg_resp_time_all = None
for exp_path in exp_paths:
    vals = {
        'exp_path': exp_path
    }
    process_exp_path(vals, processing_pipeline)


# %%
# Create overall plots

# Relative response time plot
plt.figure(figsize=(4,3))
all_bs = np.sort(rel_avg_resp_time_all['batch_size'].unique())
plt.plot(all_bs, all_bs, ls='--', c='r', label='Linear Baseline')
for raw_exp_name in rel_avg_resp_time_all['raw_exp_name'].unique():
    sub_df = rel_avg_resp_time_all.loc[rel_avg_resp_time_all['raw_exp_name'] == raw_exp_name ,:]
    exp_name = sub_df.loc[0, 'exp_name']
    plt.plot(sub_df['batch_size'], sub_df['rel_resp_time_avg'], label=exp_name)

plt.ylim([0, rel_avg_resp_time_all['rel_resp_time_avg'].max()])
plt.legend()
plt.xlabel('Batch Size')
plt.ylabel('Relative Response Time')
plt.gcf().subplots_adjust(left=0.10, bottom=0.17)
save_fig('relative_batch_response_time', 'all')

# %%
# Relative Time Per Inference
plt.figure(figsize=(4,3))
all_bs = np.sort(rel_avg_resp_time_all['batch_size'].unique())
plt.plot(all_bs, [1 for _ in all_bs], ls='--', c='r', label='Linear Baseline')
for raw_exp_name in rel_avg_resp_time_all['raw_exp_name'].unique():
    sub_df = rel_avg_resp_time_all.loc[rel_avg_resp_time_all['raw_exp_name'] == raw_exp_name ,:]
    exp_name = sub_df.loc[0, 'exp_name']
    plt.semilogx(sub_df['batch_size'], sub_df['rel_time_per_inference'], label=exp_name)

plt.ylim([-0.1, 1.1])
plt.legend()
plt.xlabel('Batch Size')
plt.ylabel('Relative Time Per Inference')
fix_log_x_plot()
plt.gcf().subplots_adjust(left=0.14, bottom=0.17)
save_fig('relative_time_per_inference', 'all')

# %%

# extract values for tables
resp_time_base_results_df = rel_avg_resp_time_all.loc[rel_avg_resp_time_all['batch_size'] == 1, ['resp_time_avg', 'exp_name']]
display(resp_time_base_results_df)














# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
def create_relative_batch_plots(**kwargs):
    df = kwargs.get('df')
    raw_exp_name = kwargs.get('raw_exp_name')

    base_row = df.loc[df['batch_size']==1,:].iloc[0]
    exp_name = kwargs.get('exp_name')

    plt.figure()
    plt.plot(df['batch_size'], df['resp_time_avg']/base_row['resp_time_avg'], label='Rel Avg RT', marker='x')
    plt.plot(df['batch_size'], df['resp_time_p95']/base_row['resp_time_p95'], label='Rel P95 RT', marker='x')
    plt.plot(df['batch_size'], df['resp_time_p99']/base_row['resp_time_p99'], label='Rel P99 RT', marker='x')
    prev_ylim = plt.gca().get_ylim()
    plt.plot(df['batch_size'], df['batch_size'], ls='--', label='Linear Baseline')
    plt.ylim(prev_ylim)
    plt.legend()
    plt.title(exp_name)
    plt.xlabel('Batch Size')
    plt.ylabel('Relative Response Time')

    save_fig('relative_batch_response_time', raw_exp_name)

    return kwargs

_ = create_relative_batch_plots(df=df, exp_name=exp_name, raw_exp_name=raw_exp_name)


# %%
def create_resp_time_hist_plots(**kwargs):
    df = kwargs.get('df')
    raw_exp_name = kwargs.get('raw_exp_name')

    # min_x_plt = (df['resp_time_p5']/df['resp_time_avg']).min()
    # max_x_plt = (df['resp_time_p95']/df['resp_time_avg']).max()
    min_x_plt = (df['resp_time_p5']/df['batch_size']).min()
    max_x_plt = (df['resp_time_p95']/df['batch_size']).max()


    data = []
    batch_sizes = df['batch_size'].unique()
    for batch_size in batch_sizes:
        base_row = df.loc[df['batch_size']==batch_size,:].iloc[0]

        exp_name = kwargs.get('exp_name')
        fig_title = f"{exp_name} (BS={batch_size})"
        rt_hist_bins = kwargs.get('rt_hist_bins', 100)

        # hist_data = base_row['response_times_ms']/base_row['resp_time_avg']
        hist_data = base_row['response_times_ms']/base_row['batch_size']
        gkde = stats.gaussian_kde(dataset=hist_data)

        # for violin plot
        data.append(hist_data)

        # plt.figure()
        # hist, bins, _ = plt.hist(hist_data, bins=rt_hist_bins, range=(min_x_plt, max_x_plt), density=True, alpha=0.6, color='#607c8e')
        # bins_kde = gkde(bins)
        # plt.plot(bins, bins_kde, ls='--', color='k')
        # plt.xlim((min_x_plt,max_x_plt))
        # plt.title(fig_title)
        # plt.grid(alpha=0.75)
        # plt.xlabel('Time Per Prediction (ms)')
        # plt.ylabel('Density')
        # save_fig(f'resp_time_hist_bs{batch_size}', raw_exp_name)

    labels = [f"BS={bs}" for bs in batch_sizes]
    plt.figure()
    sns.boxplot(data=data, showfliers=False)
    plt.gca().set_xticks(np.arange(0, len(labels)))
    plt.gca().set_xticklabels(labels)
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('Time Per Inference (ms)')
    plt.xlabel('Batch Size')
    plt.title(exp_name)
    # save_fig(f'resp_time_boxplot_bs', raw_exp_name)

    return kwargs

_ = create_resp_time_hist_plots(df=df, exp_name=exp_name, raw_exp_name=raw_exp_name)


# %%
processing_pipeline = [
    ('Extract Experiment Name', lambda vals: {
        'exp_name': get_exp_name(vals['exp_path']),
        'raw_exp_name': extract_raw_exp_name(vals['exp_path']),
    }),
    ('Read CSV', lambda vals: {'df': pd.read_csv(vals['exp_path'], index_col=0, parse_dates=True)}),
    ('Post Processing', lambda vals: {'df': post_process(vals['df'])} ),
    ('Relative Batch Plots', lambda vals: create_relative_batch_plots(**vals)),
    ('Relative Response Time Histograms', lambda vals: create_resp_time_hist_plots(**vals)),
]

def process_exp_path(vals, processing_pipeline):
    exp_path = vals['exp_path']
    print('='*20, exp_path, '='*20)

    inter = vals
    for step_name, step_func in processing_pipeline:
        print(f'[*] {step_name}')
        inter_update = step_func(inter)
        inter.update(inter_update)
    return inter


for exp_path in exp_paths:
    vals = {
        'exp_path': exp_path
    }
    process_exp_path(vals, processing_pipeline)

# %% [markdown]
# # Creating Plots
# 
# In the previous section, we analyzed the performance of the system under different circumistances using the default configuration.
# Here, our goal is to analyze the performance under differenet configurations.

# %%
