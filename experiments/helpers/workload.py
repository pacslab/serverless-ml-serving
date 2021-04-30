#! /usr/local/bin/python

import time
import threading
from collections import deque

import numpy as np

from pacswg.timer import TimerClass


class WorkloadLogger:
    conc_count = 0
    conc_lock = threading.Lock()
    threads_stop_signal = False
    recorded_data = {}

    def __init__(self, get_ready_cb, cc_average_count=60, monitor_timeout=1, record_timeout=2):
        super().__init__()
        self.most_recent_cc_queue = deque(maxlen=cc_average_count)
        self.get_ready_cb = get_ready_cb
        self.monitor_timeout = monitor_timeout
        self.record_timeout = record_timeout
        
    def get_recorded_data(self):
        return self.recorded_data

    def start_capturing(self):
        self.monitoring_thread = threading.Thread(target=self.monitor_conc_loop, args=(self.monitor_timeout,), daemon=True)
        self.record_thread = threading.Thread(target=self.record_conc_loop, args=(self.record_timeout,), daemon=True)
        # clear recorded data
        self.recorded_data.clear()
        # start capturing
        print('starting threads')
        self.threads_stop_signal = False
        self.monitoring_thread.start()
        self.record_thread.start()

    def stop_capturing(self):
        print('stopping threads...')
        self.threads_stop_signal = True
        # wait for threads to stop running
        if self.monitoring_thread is not None:
            self.monitoring_thread.join()
        if self.record_thread is not None:
            self.record_thread.join()
        print('Done.')

    def get_conc(self):
        return self.conc_count

    def inc_conc(self):
        with self.conc_lock:
            self.conc_count += 1

    def dec_conc(self):
        with self.conc_lock:
            self.conc_count -= 1

    def reset_conc(self):
        with self.conc_lock:
            self.conc_count = 0

    def monitor_conc_loop(self, timeout=1):
        timer = TimerClass()
        while not(self.threads_stop_signal):
            timer.tic()
            # record concurrency into self.most_recent_cc_queue
            ready_count = self.get_ready_cb()
            if ready_count > 0:
                # cc = tcc / N
                cc = self.get_conc() / ready_count
            else:
                # assume we have one ready, but it doesn't still show
                # ready remains one until health checks complete
                # but sometimes container is up before that
                cc = self.get_conc()

            self.most_recent_cc_queue.append(cc)
            while timer.toc() < timeout:
                time.sleep(0.01)

    def get_cc_window_average(self):
        if len(self.most_recent_cc_queue) > 0:
            return np.mean(self.most_recent_cc_queue)
        return -1

    def record_data(self, data):
        for k in data:
            if k not in self.recorded_data:
                self.recorded_data[k] = []

            self.recorded_data[k].append(data[k])
            

    def record_conc_loop(self, timeout=2):
        timer = TimerClass()
        while not(self.threads_stop_signal):
            timer.tic()
            # record concurrency value and others
            self.record_data({
                'ready_count': self.get_ready_cb(),
                'total_conc': self.get_conc(),
                'conc_window_average': self.get_cc_window_average(),
                'time': time.time(),
            })
            while timer.toc() < timeout:
                time.sleep(0.01)

    def worker_func(self, user_func):
        self.inc_conc()
        try:
            start_conc = self.conc_count
            start_ready_count = self.get_ready_cb()
            client_start_time = time.time()
            # send the request and check if success
            success = user_func()
            client_end_time = time.time()
            end_ready_count = self.get_ready_cb()
            end_conc = self.conc_count
        except:
            client_start_time = -1
            client_end_time = -1
            end_conc = -1
            end_ready_count = -1
            success = False
        finally:
            self.dec_conc()

        return {
            'client_start_time': client_start_time,
            'client_end_time': client_end_time,
            'client_elapsed_time': client_end_time - client_start_time,
            'start_conc': start_conc,
            'end_conc': end_conc,
            'success': success,
            'start_ready_count': start_ready_count,
            'end_ready_count': end_ready_count,
        }