# %%
# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Try loading autoscale data
autoscale_folder_path = './files/AutoScale/'
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

# plot the autoscale files
for trace_fname in autoscale_trace_file_names:
  # get the file path
  curr_file_path = autoscale_folder_path + trace_fname
  print('loading file:', curr_file_path)

  # load the file
  trace_arr = np.loadtxt(curr_file_path)
  plt.figure()
  plt.plot(trace_arr)
  plt.title(trace_fname)
# %%
