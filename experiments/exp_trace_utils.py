# imports
import os
import time
from datetime import datetime
import pytz
from collections import deque
from threading import Thread

# library imports
import requests
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set logging
import logging
logging.basicConfig(
    filename='controller.log', 
    filemode='w', level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

# config
my_timezone = os.getenv('PY_TZ', 'America/Toronto')

# small functions
def get_time_with_tz():
    return datetime.now().astimezone(pytz.timezone(my_timezone))

from_js_timestamp = lambda x: datetime.utcfromtimestamp(x/1000).astimezone(pytz.timezone(my_timezone))

class SmartProxyController:
    @staticmethod
    def calculate_new_bs(curr_bs, inc=True, config=None):
        """
        Static function that helps calculate new batch size
        """
        if config is None:
            config = {
                'max_bs': 100,
                'min_bs': 20,
                'inc_step': 1,
                'dec_mult': 0.7,
            }

        if inc:
            new_bs = curr_bs + config['inc_step']
            # at least one step is made
            new_bs = max(new_bs, curr_bs+1)
        else:
            new_bs = curr_bs * config['dec_mult']
            new_bs = int(new_bs)
            # at least one step is made
            new_bs = min(new_bs, curr_bs-1)

        new_bs = min(new_bs, config['max_bs'])
        new_bs = max(new_bs, config['min_bs'])

        return new_bs

    def __init__(self, service_name, slo_timeout, initial_batch_size, bs_config, **kwargs):
        # parameters with defaults
        self.server_address = kwargs.get('server_address', 'http://localhost:3000')
        self.average_timeout_ratio_threshold = kwargs.get('average_timeout_ratio_threshold', 0.5)
        self.upstream_rt_max_len = kwargs.get('upstream_rt_max_len', 1000)
        # calculated default parameters
        self.slo_target = kwargs.get('slo_target', slo_timeout * 0.8)

        # necessary parameters
        self.service_name = service_name
        self.slo_timeout = slo_timeout
        self.initial_batch_size = initial_batch_size
        self.bs_config = bs_config

        # empty stuff
        self.batch_rt_values = {}
        self.control_thread = None
        self.stop_control_thread_signal = False

        self.set_initial_config()

    def control_loop(self):
        while self.stop_control_thread_signal == False:
            logging.debug('running control loop')

            # get and save proxy stats
            proxy_stats = self.get_proxy_stats()
            self.update_batch_rt_values(proxy_stats=proxy_stats)
            self.acc_proxy_stats.append(proxy_stats)

            shouldIncrease = True
            if proxy_stats['reponseTimeP95'] is not None:
                if proxy_stats['reponseTimeP95'] >= self.slo_target:
                    logging.info(f"reponseTimeP95 ({proxy_stats['reponseTimeP95']}) >= slo_target ({self.slo_target})")
                    shouldIncrease = False
            if proxy_stats['averageTimeoutRatio'] is not None:
                if proxy_stats['averageTimeoutRatio'] >= self.average_timeout_ratio_threshold:
                    logging.info(f"averageTimeoutRatio({proxy_stats['averageTimeoutRatio']}) >= average_timeout_ratio_threshold({self.average_timeout_ratio_threshold})")
                    shouldIncrease = False

            new_bs = SmartProxyController.calculate_new_bs(self.curr_bs, shouldIncrease, self.bs_config)
            self.set_proxy_config({
                'maxBufferSize': new_bs,
                'maxBufferTimeoutMs': self.slo_target,
            })
            logging.info(f"Changing BS from {self.curr_bs} to {new_bs}")
            self.curr_bs = new_bs

            time.sleep(30)

        logging.info('stopping control thread')

    def stop_control_thread(self):
        while self.control_thread.is_alive():
            self.stop_control_thread_signal = True
            time.sleep(1)
        logging.info('Control thread stopped successfully')

    def start_control_thread(self):
        if self.control_thread is None or not(self.control_thread.is_alive()):
            # create the thread
            logging.info('starting control thread...')
            self.stop_control_thread_signal = False
            self.control_thread = Thread(target=self.control_loop, args=(), daemon=True)
            self.control_thread.start()
            # wait for initial results to appear before returning to do something else
            time.sleep(1)
        else:
            # thread already exists
            logging.info('control thread already started')

    def set_initial_config(self):
        logging.info(f'Initializing Config, BS={self.initial_batch_size}')
        self.curr_bs = self.initial_batch_size
        # accumulated proxy stats
        self.acc_proxy_stats = []
        return self.set_proxy_config({
            'maxBufferSize': self.initial_batch_size,
            'maxBufferTimeoutMs': self.slo_target,
        })

    def get_batch_rt_values(self):
        return self.batch_rt_values

    def fetch_raw_server_proxy_stats(self):
        url = f'{self.server_address}/proxy-monitor/{self.service_name}'
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_proxy_stats(self):
        raw_stats = self.fetch_raw_server_proxy_stats()
        return {
            'maxBufferSize': raw_stats['maxBufferSize'],
            'averageMaxBufferSize': raw_stats['windowedHistoryValues']['maxBufferSize']['average'],
            'averageActualBatchSize': raw_stats['windowedUpstream']['batchSizes']['average'],
            'maxBufferTimeoutMs': raw_stats['maxBufferTimeoutMs'],
            'currentReplicaCount': raw_stats['currentMonitorStatus']['currentReplicaCount'],
            'currentReadyReplicaCount': raw_stats['currentMonitorStatus']['currentReplicaCount'],
            'currentConcurrency': raw_stats['currentMonitorStatus']['currentConcurrency'],
            'averageConcurrency': raw_stats['windowedHistoryValues']['concurrency']['average'],
            'averageArrivalRate': raw_stats['windowedHistoryValues']['arrival']['rate'],
            'averageDepartureRate': raw_stats['windowedHistoryValues']['departure']['rate'],
            'averageDispatchRate': raw_stats['windowedHistoryValues']['dispatch']['rate'],
            'averageErrorRate': raw_stats['windowedHistoryValues']['error']['rate'],
            'averageTimeoutRatio': raw_stats['windowedHistoryValues']['timeoutRatio']['average'],
            'reponseTimeAverage': raw_stats['responseTimes']['stats']['average'],
            'reponseTimeP50': raw_stats['responseTimes']['stats']['q50'],
            'reponseTimeP95': raw_stats['responseTimes']['stats']['q95'],
            'batchResponseTimeStats': raw_stats['windowedUpstream']['responseTimes'],
        }

    def set_proxy_config(self, update_config):
        url = f'{self.server_address}/proxy-config/{self.service_name}'
        response = requests.post(url, json=update_config)
        response.raise_for_status()
        return response.json()

    def update_batch_rt_values(self, proxy_stats=None):
        # update if stats are not passed
        if proxy_stats is None:
            proxy_stats = self.get_proxy_stats()
        for batch_size_str in proxy_stats['batchResponseTimeStats']:
            batch_size = int(batch_size_str)
            if not(batch_size in self.batch_rt_values):
                self.batch_rt_values[batch_size] = deque(maxlen=self.upstream_rt_max_len)
            # concatenate arrays
            self.batch_rt_values[batch_size].extend(proxy_stats['batchResponseTimeStats'][batch_size_str]['values'])

        return self.batch_rt_values

