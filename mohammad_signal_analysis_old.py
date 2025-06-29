import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.integrate import cumtrapz

import glob
import os
import cv2
import scipy.signal as signal
import scipy
import time
from scipy.stats import ttest_ind_from_stats
from scipy.stats import sem, t
from sklearn.linear_model import LinearRegression
from scipy.stats import variation
from scipy.interpolate import RegularGridInterpolator, interp1d


class Signal_Analysis_Mohammad:
    def __init__(self):
        self.signal = None
        self.task_type = None
        self.signal_after_butterworth = None
        self.signal_filtered_derivative = None
        self.signal_second_derivative = None
        self.file_name = None
        self.video_file_location = None
        self.results_location = None
        self.sensor_data_location = None
        self.sensor_video_file_location = None
        self.fps = None
        self.signal_timing = None
        self.max_peaks = None
        self.min_peaks = None
        self.prominences = None
        self.contour_heights = None
        self.top_peaks_idx_max = None
        self.top_peaks_idx_min = None
        self.filtered_max = None
        self.prominences_idx = None
        self.patient_title = None
        self.WIDTH_NUM = 2
        self.cali_list = None
        self.ang_v = None
        self.signal_end_peak = None
        self.signal_start_peak = None
        self.power_max_peaks = None
        self.power_min_peaks = None

        # Phenotypes
        self.FLEXION_PHENOTYPE = None
        self.FLEXION_TIME = None
        self.EXTENSION_TIME = None
        self.FLEXION_TIME_PLUS_EXTENSION_TIME = None
        self.FLEXION_EXTENSION_RATE = None
        self.N_FLEXION_EXTENSION = None
        self.ANGULAR_VELOCITY_RANGE = None
        self.RISING_TIME_AVG = None
        self.FALLING_TIME_AVG = None
        self.RISING_TIME_PLUS_FALLING_TIME = None
        self.POWER = None
        self.POWER_DECLINE_PERCENTAGE = None
        self.FRAILTY_INDEX = None

        # Dual Task Cost
        self.ST_OUTPUT_FILE = None
        self.DT_OUTPUT_FILE = None
        self.DT_Cost = None

        # Innovative ones
        self.COGNITIVE_ROM_DECLINE_RATE_1 = None
        self.COGNITIVE_ROM_LR_RESIDUAL_SUM = None
        self.COGNITIVE_ROM_LR_SLOPE = None

        self.COGNITIVE_IRREGULAR_PAUSES_RATE_1 = None
        self.COGNITIVE_IRREGULAR_PAUSES_LR_RESIDUAL_SUM = None
        self.COGNITIVE_IRREGULAR_PAUSES_LR_SLOPE = None

        # Sensor data
        self.sensory_run = True

        # Angular Acceleration
        self.angular_acc_folder_path = None

    def get_signal(self):
        print('*' * 100)
        print(self.file_name)
        self.signal = pd.read_excel(self.file_name, sheet_name='Sheet1', header=None)
        self.patient_title = self.file_name.split('\\')[-1].split('.')[0]
        self.signal = self.signal.to_numpy().squeeze()

    def reverse_signal(self):
        return -self.signal

    def critical_points(self):
        ''' Max peaks '''
        self.max_peaks, sth = find_peaks(self.signal, width=self.WIDTH_NUM)
        output = peak_prominences(self.signal, self.max_peaks)
        self.prominences = output[0]
        self.prominences_idx = output[1]
        self.top_peaks_idx_max = np.where(self.prominences > 60)[0]
        self.prominences = self.prominences[self.top_peaks_idx_max]
        self.prominences_idx = self.prominences_idx[self.top_peaks_idx_max]

        ''' Min peaks '''
        reversed_signal = self.reverse_signal()
        self.min_peaks, sth = find_peaks(reversed_signal, width=self.WIDTH_NUM)
        output = peak_prominences(reversed_signal, self.min_peaks)
        self.prominences = output[0]
        self.prominences_idx = output[1]
        self.top_peaks_idx_min = np.where(self.prominences > 60)[0]
        self.prominences = self.prominences[self.top_peaks_idx_min]
        self.prominences_idx = self.prominences_idx[self.top_peaks_idx_min]
        self.infls = None

    def peaks_cleaner(self):
        ''' Remove redundant minimum points between two Max points '''
        redundant_min_idx = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            try:
                first_val = self.max_peaks[self.top_peaks_idx_max[i]]
                second_val = self.max_peaks[self.top_peaks_idx_max[i + 1]]
                boolean_array = np.logical_and(self.min_peaks[self.top_peaks_idx_min] >= first_val,
                                               self.min_peaks[self.top_peaks_idx_min] <= second_val)
                in_range_indices = np.where(boolean_array)[0]
                if len(in_range_indices) > 1:
                    global_min = np.argmin(self.signal[self.min_peaks[self.top_peaks_idx_min[in_range_indices]]])

                    for j in range(len(in_range_indices)):
                        redundant_min_idx.append(in_range_indices[j])
                    index = np.argwhere(redundant_min_idx == in_range_indices[global_min])
                    redundant_min_idx = list(np.delete(redundant_min_idx, index))
                # if len(in_range_indices) < 1:
                #     if first_val > second_val:

            except Exception as e:
                print(str(e))
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, redundant_min_idx)

        ''' Remove redundant Maximum points between two min points '''
        redundant_max_idx = []
        for i in range(len(self.top_peaks_idx_min) - 1):
            try:
                first_val = self.min_peaks[self.top_peaks_idx_min[i]]
                second_val = self.min_peaks[self.top_peaks_idx_min[i + 1]]
                boolean_array = np.logical_and(self.max_peaks[self.top_peaks_idx_max] >= first_val,
                                               self.max_peaks[self.top_peaks_idx_max] <= second_val)
                in_range_indices = np.where(boolean_array)[0]
                if len(in_range_indices) > 1:
                    global_max = np.argmax(self.signal[self.max_peaks[self.top_peaks_idx_max[in_range_indices]]])

                    for j in range(len(in_range_indices)):
                        redundant_max_idx.append(in_range_indices[j])
                    index = np.argwhere(redundant_max_idx == in_range_indices[global_max])
                    redundant_max_idx = list(np.delete(redundant_max_idx, index))
            except Exception as e:
                print(str(e))
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, redundant_max_idx)

        ''' Remove redundant Max and Min points of beginning and ending of the signal '''
        redundant_min_idx = []
        for i in range(len(self.top_peaks_idx_min) - 1):
            first_val = self.min_peaks[self.top_peaks_idx_min[i]]
            second_val = self.min_peaks[self.top_peaks_idx_min[i + 1]]
            boolean_array = np.logical_and(self.max_peaks[self.top_peaks_idx_max] >= first_val,
                                           self.max_peaks[self.top_peaks_idx_max] <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            if len(in_range_indices) < 1:
                redundant_min_idx.append(self.top_peaks_idx_min[i])
            else:
                self.top_peaks_idx_min = np.setdiff1d(self.top_peaks_idx_min, redundant_min_idx)
                break
        redundant_max_idx = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            first_val = self.max_peaks[self.top_peaks_idx_max[i]]
            second_val = self.max_peaks[self.top_peaks_idx_max[i + 1]]
            boolean_array = np.logical_and(self.min_peaks[self.top_peaks_idx_min] >= first_val,
                                           self.min_peaks[self.top_peaks_idx_min] <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            if len(in_range_indices) < 1:
                redundant_max_idx.append(self.top_peaks_idx_max[i])
            else:
                self.top_peaks_idx_max = np.setdiff1d(self.top_peaks_idx_max, redundant_max_idx)
                break

    def peaks_remove_outliers(self):
        ''' Remove peaks greater or lower than a threshold '''
        ''' Remove max outliers '''
        avg_max = np.mean(self.signal[self.max_peaks[self.top_peaks_idx_max]])
        difference = avg_max - self.signal[self.min_peaks[self.top_peaks_idx_min]]
        redundant_min_idx = []
        for i in range(len(difference)):
            if difference[i] < 36:
                redundant_min_idx.append(i)
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, redundant_min_idx)
        ''' Remove min outliers '''
        avg_min = np.mean(self.signal[self.min_peaks[self.top_peaks_idx_min]])
        difference = np.abs(avg_min - self.signal[self.max_peaks[self.top_peaks_idx_max]])
        redundant_max_idx = []
        for i in range(len(difference)):
            if difference[i] < 36:
                redundant_max_idx.append(i)
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, redundant_max_idx)

        ''' Detect x max gaps '''
        gaps = np.array([self.cali_list[self.max_peaks[self.top_peaks_idx_max]][i + 1] -
                         self.cali_list[self.max_peaks[self.top_peaks_idx_max]][i] for i in
                         range(len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]]) - 1)])
        # print(gaps)
        gap_idx = np.where(gaps < 0.25)[0]
        # print(gap_idx)
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, gap_idx)

        ''' Detect x min gaps '''
        # writing this section is optional
        gaps = np.array([self.cali_list[self.min_peaks[self.top_peaks_idx_min]][i + 1] -
                         self.cali_list[self.min_peaks[self.top_peaks_idx_min]][i] for i in
                         range(len(self.cali_list[self.min_peaks[self.top_peaks_idx_min]]) - 1)])
        # print(gaps)
        gap_idx = np.where(gaps < 0.25)[0]
        # print(gap_idx)
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, gap_idx)

    def detect_start_point(self):
        '''
        In this function second five peaks mean is calculated
        and compared with first five peaks
        '''
        ''' Remove max first outliers '''
        # print('1')
        # print(self.signal[self.max_peaks[self.top_peaks_idx_max]])
        redundant_max_idx = self.detect_first_outliers_func(self.signal[self.max_peaks[self.top_peaks_idx_max]])
        # print(redundant_max_idx)
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, redundant_max_idx)

        # Gap check for Max points
        redundant_gap_idx = self.detect_first_outliers_func(self.cali_list[self.max_peaks[self.top_peaks_idx_max]])
        # print(redundant_gap_idx)
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, redundant_gap_idx)

        ''' Remove min first outliers '''
        redundant_min_idx = self.detect_first_outliers_func(self.signal[self.min_peaks[self.top_peaks_idx_min]])
        # print(redundant_min_idx)
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, redundant_min_idx)

        # Gap check for min points
        redundant_gap_idx = self.detect_first_outliers_func(self.cali_list[self.min_peaks[self.top_peaks_idx_min]])
        # print(redundant_gap_idx)
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, redundant_gap_idx)

    def detect_first_outliers_func(self, values):
        redundant_idx = []
        values_mean = np.mean(values[3:8])
        value_sem = sem(values[3:8])
        h = value_sem * t.ppf((1 + 0.999) / 2., len(values[3:8]) - 1)
        for i in range(4):
            if values_mean - h < values[i] < values_mean + h:
                pass
            else:
                redundant_idx.append(i)
        return redundant_idx

    def detect_inflcs(self):
        ''' Detect inflection points '''
        smooth_d2 = np.gradient(np.gradient(self.signal))
        self.infls = np.where(np.diff(np.sign(smooth_d2)))[0]

        # Detect first peak
        if self.min_peaks[self.top_peaks_idx_min][0] > self.max_peaks[self.top_peaks_idx_max][0]:
            first_peak = self.max_peaks[self.top_peaks_idx_max][0]
        else:
            first_peak = self.min_peaks[self.top_peaks_idx_min][0]

        # Detect last peak
        if self.min_peaks[self.top_peaks_idx_min][-1] < self.max_peaks[self.top_peaks_idx_max][-1]:
            last_peak = self.max_peaks[self.top_peaks_idx_max][-1]
        else:
            last_peak = self.min_peaks[self.top_peaks_idx_min][-1]

        # Filter start and end inflection points
        self.infls = self.infls[(self.infls <= last_peak)]
        self.infls = self.infls[(self.infls >= first_peak)]

        # Detect the Mid inflection point
        all_peaks = np.concatenate((self.max_peaks[self.top_peaks_idx_max], self.min_peaks[self.top_peaks_idx_min]))
        all_peaks.sort(kind='mergesort')
        infls_list = []
        for i in range(len(all_peaks) - 1):
            first_val = all_peaks[i]
            second_val = all_peaks[i + 1]
            boolean_array = np.logical_and(self.infls >= first_val, self.infls <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            idx = (np.abs(self.signal[self.infls[in_range_indices]] - (
                    np.abs(self.signal[first_val] + self.signal[second_val]) / 2))).argmin()
            infls_list.append(self.infls[in_range_indices[idx]])
        self.infls = np.array(infls_list)

    def get_fps(self):
        video_list = []
        for root, dirs, files in os.walk(self.video_file_location):
            for file in files:
                # print(file)
                if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".3gp") or file.endswith(".3GP"):
                    video_list.append(os.path.join(root, file))
                elif file.endswith(".MOV") or file.endswith(".mov"):
                    video_list.append(os.path.join(root, file))
                else:
                    pass
        video_dict = {}
        for i in range(len(video_list)):
            video_dict[video_list[i].split('\\')[-1].split('.')[0]] = video_list[i]
        video = cv2.VideoCapture(video_dict[self.patient_title])
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.time_calibration()

    def phenotypes(self):
        ''' Calculate all Phenotypes '''
        self.start_end_signal_peak()

        ''' Flexion '''
        if self.signal_end_peak is 'min':
            end_adjuster = 0
        else:
            end_adjuster = 1

        flx_list = []
        for i in range(len(self.top_peaks_idx_max) - end_adjuster):
            flexion = self.signal[self.max_peaks[self.top_peaks_idx_max[i + end_adjuster]]] - self.signal[
                self.min_peaks[self.top_peaks_idx_min[i]]]
            flx_list.append(flexion)
        self.FLEXION_PHENOTYPE = np.mean(flx_list)

        # print(flx_list)
        # print('Flexion: ', self.FLEXION_PHENOTYPE)

        ''' Flexion time '''
        if self.signal_start_peak == 'Max':
            rst_s_adjuster = 1
        else:
            rst_s_adjuster = 0

        rst_list = []
        for i in range(len(self.top_peaks_idx_max) - rst_s_adjuster):
            rise_time = self.cali_list[self.max_peaks[self.top_peaks_idx_max[i + rst_s_adjuster]]] - self.cali_list[
                self.min_peaks[self.top_peaks_idx_min[i]]]
            rst_list.append(rise_time)

        # print(rst_list)
        self.FLEXION_TIME = np.mean(rst_list)

        ''' Extension time '''
        if self.signal_start_peak is 'Max':
            fall_adj = 0
        else:
            fall_adj = 1
        ft_list = []
        for i in range(len(self.top_peaks_idx_min) - fall_adj):
            rise_time = self.cali_list[self.min_peaks[self.top_peaks_idx_min[i + fall_adj]]] - self.cali_list[
                self.max_peaks[self.top_peaks_idx_max[i]]]
            ft_list.append(rise_time)
        # print(ft_list)
        self.EXTENSION_TIME = np.mean(ft_list)

        if len(ft_list) > len(rst_list):
            counter = len(rst_list)
        else:
            counter = len(ft_list)
        fet_list = []
        for i in range(counter):
            fet_list.append(rst_list[i] + ft_list[i])
        fet_list = np.array(fet_list)

        self.FLEXION_TIME_PLUS_EXTENSION_TIME = self.FLEXION_TIME + self.EXTENSION_TIME

        ''' Flexion/Extension Rate '''
        duration = self.cali_list[self.max_peaks[self.top_peaks_idx_max[-1]]] - self.cali_list[
            self.max_peaks[self.top_peaks_idx_max[0]]]
        self.N_FLEXION_EXTENSION = len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]])
        # print('self.N_FLEXION_EXTENSION', self.N_FLEXION_EXTENSION)
        self.N_FLEXION_EXTENSION = int(
            np.round(len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]]) * (20.0 / duration)))
        # print(duration)
        # print((20.0 / duration))
        # print('self.N_FLEXION_EXTENSION', self.N_FLEXION_EXTENSION)
        self.FLEXION_EXTENSION_RATE = (self.N_FLEXION_EXTENSION / duration) * 60
        # print(self.N_FLEON_EXTEXION_EXTENSION)
        #         # print(self.FLEXINSION_RATE)

        ''' Angular Velocity Range '''
        ang_r = []
        for i in range(len(self.infls)):
            x_l, y_l = self.cali_list[self.infls[i] - 1], self.signal[self.infls[i] - 1]
            x_h, y_h = self.cali_list[self.infls[i] + 1], self.signal[self.infls[i] + 1]
            derivative = (y_h - y_l) / (x_h - x_l)
            ang_r.append(derivative)
        ang_r_ = []
        # self.ANGULAR_VELOCITY_RANGE = np.mean(ang_r)
        if self.signal_start_peak is 'Max':
            ang_v = True
        else:
            ang_v = False
        if ang_v:
            for i in range(0, len(ang_r) - 1, 2):
                ang_r_.append(ang_r[i + 1] - ang_r[i])
        else:
            for i in range(0, len(ang_r) - 1, 2):
                ang_r_.append(ang_r[i] - ang_r[i + 1])

        # print(ang_r)
        # print(ang_r_)
        self.ANGULAR_VELOCITY_RANGE = np.mean(ang_r_)
        # print(self.ANGULAR_VELOCITY_RANGE)
        # print('here')

        ''' Rising time '''
        rise_time_list = []
        if self.signal_start_peak is 'Max':
            rst_adj = 0
        else:
            rst_adj = 1
        if self.signal_end_peak is 'Max':
            rst_m_adj = 1
        else:
            rst_m_adj = 0
        for i in range(len(self.top_peaks_idx_max) - rst_m_adj):
            rst = self.cali_list[self.infls[2 * (i + rst_adj) - rst_adj]] - self.cali_list[
                self.max_peaks[self.top_peaks_idx_max[i]]]
            rise_time_list.append(rst)
        # print(rise_time_list)
        # print(len(rise_time_list))
        self.RISING_TIME_AVG = np.mean(rise_time_list)
        # print(self.RISING_TIME_AVG)

        ''' Falling time '''
        ftv_list = []
        if self.signal_end_peak is 'min':
            ft_adj = 1
        else:
            ft_adj = 0
        if self.signal_start_peak is 'min':
            ft_f_adj = 0
        else:
            ft_f_adj = 1
        for i in range(len(self.top_peaks_idx_min) - ft_adj):
            ft_v = self.cali_list[self.infls[2 * (i + ft_f_adj) - ft_f_adj]] - self.cali_list[
                self.min_peaks[self.top_peaks_idx_min[i]]]
            ftv_list.append(ft_v)
        # print(ftv_list)
        # print(len(ftv_list))
        self.FALLING_TIME_AVG = np.mean(ftv_list)

        if len(ftv_list) > len(rise_time_list):
            counter = len(rise_time_list)
        else:
            counter = len(ftv_list)
        rft_list = []
        for i in range(counter):
            rft_list.append(rise_time_list[i] + ftv_list[i])
        rft_list = np.array(rft_list)

        self.RISING_TIME_PLUS_FALLING_TIME = self.RISING_TIME_AVG + self.FALLING_TIME_AVG

        ''' Power '''
        # First derivative - Angular Velocity
        self.smooth_func(tresh=0.19)
        self.signal_filtered_derivative = np.diff(self.signal_after_butterworth) / np.diff(self.cali_list)
        self.signal_filtered_derivative = np.append(self.signal_filtered_derivative,
                                                    self.signal_filtered_derivative[-1])

        # Second derivative - Angular Acceleration
        self.signal_second_derivative = np.diff(self.signal_filtered_derivative) / np.diff(self.cali_list)
        self.signal_second_derivative = np.append(self.signal_second_derivative, self.signal_second_derivative[-1])

        # print(self.signal_filtered_derivative)
        # print(self.signal_second_derivative)
        self.POWER = self.signal_filtered_derivative * self.signal_second_derivative

        ''' Speed plot '''
        plt.figure(figsize=(8, 2))
        plt.plot(self.cali_list[110:780], self.signal_filtered_derivative[110:780],
                 label='original signal', linewidth=2.5)
        plt.xlabel('Time (sec)')
        plt.ylabel('Angular Velocity\n(deg\sec)')
        plt.grid(False)
        # Remove top and right borders
        ax = plt.gca()  # Get current axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        # save_path = r"C:\Users\u241915\OneDrive - Baylor College of Medicine\Lee, Myeounggon - Lee, Myeounggon's files\Image-based\Mohammad_figures\velocity1.png"
        # plt.savefig(save_path, dpi=500, bbox_inches='tight')
        # plt.show()

        ''' Update Power Calculations '''
        self.POWER = self.POWER / 25
        # print('self.POWER', self.POWER)
        # print(self.POWER[self.infls])

        ''' Frailty Index '''
        # Ph1
        ph1_rigidity = self.FLEXION_PHENOTYPE
        print('Rigidity: ', ph1_rigidity)

        # Ph2
        print('+' * 100)
        mid_idx = len(self.signal[self.max_peaks[self.top_peaks_idx_max]]) // 2
        mid_idx = self.max_peaks[self.top_peaks_idx_max[mid_idx]]
        self.power_max_peaks, sth = find_peaks(self.POWER, width=self.WIDTH_NUM)
        self.power_min_peaks, sth = find_peaks(-self.POWER, width=self.WIDTH_NUM)
        power_first_max_peaks, power_second_max_peaks = [self.power_max_peaks[self.power_max_peaks < mid_idx],
                                                         self.power_max_peaks[self.power_max_peaks >= mid_idx]]
        power_first_min_peaks, power_second_min_peaks = [self.power_min_peaks[self.power_min_peaks < mid_idx],
                                                         self.power_min_peaks[self.power_min_peaks >= mid_idx]]
        ''' Remove first outliers '''
        # print(self.min_peaks[self.top_peaks_idx_min])
        # print(power_first_max_peaks)
        idx = (np.abs(power_first_max_peaks - self.min_peaks[self.top_peaks_idx_min][1])).argmin()
        # print(power_first_max_peaks[idx])
        power_first_max_peaks = power_first_max_peaks[idx:]
        power_first_min_peaks = power_first_min_peaks[idx:]
        # print(power_first_max_peaks)

        k = np.min([len(power_first_max_peaks),
                    len(power_second_max_peaks),
                    len(power_first_min_peaks),
                    len(power_second_min_peaks)])

        # k = len(power_first_max_peaks)
        top_k_max_first_peaks = np.argpartition(self.POWER[power_first_max_peaks], -k)[-k:]
        # k = len(power_second_max_peaks)
        top_k_max_second_peaks = np.argpartition(self.POWER[power_second_max_peaks], -k)[-k:]

        # k = len(power_first_min_peaks)
        top_k_min_first_peaks = np.argpartition(-self.POWER[power_first_min_peaks], -k)[-k:]
        # k = len(power_second_min_peaks)
        top_k_min_second_peaks = np.argpartition(-self.POWER[power_second_min_peaks], -k)[-k:]

        power_range_first_section = np.mean(self.POWER[power_first_max_peaks[top_k_max_first_peaks]]) - np.mean(
            self.POWER[power_first_min_peaks[top_k_min_first_peaks]])
        power_range_second_section = np.mean(self.POWER[power_second_max_peaks[top_k_max_second_peaks]]) - np.mean(
            self.POWER[power_second_min_peaks[top_k_min_second_peaks]])
        print(power_range_first_section, power_range_second_section)
        self.POWER_DECLINE_PERCENTAGE = np.round((1 - (power_range_second_section / power_range_first_section)) * 1,
                                                 decimals=3)
        self.PowR_PD_new = ((power_range_second_section - power_range_first_section) / power_range_first_section) * 100
        ph2_exhaustion = self.POWER_DECLINE_PERCENTAGE

        print('Exhaustion: ', ph2_exhaustion)

        # Ph3
        ph3_slowness = self.FLEXION_TIME * 1000
        print('Slowness: ', ph3_slowness)

        # Ph4
        ph4_steadiness_lack_flexion = np.std(rst_list) / np.mean(rst_list)
        print('Steadiness_lack_flexion: ', ph4_steadiness_lack_flexion)

        # Ph5
        ph5_steadiness_lack_extension = np.std(ft_list) / np.mean(ft_list)
        print('Steadiness_lack_extension: ', ph5_steadiness_lack_extension)

        b = 0.24495
        a1 = -1.7357 * 0.001
        a2 = -1.2026 * 0.001
        a3 = 0.36848 * 0.001
        a4 = -0.49396
        a5 = 0.48974
        self.FRAILTY_INDEX = b + a1 * ph1_rigidity + a2 * ph2_exhaustion + a3 * ph3_slowness + \
                             a4 * ph4_steadiness_lack_flexion + a5 * ph5_steadiness_lack_extension
        print('FRAILTY_INDEX: ', self.FRAILTY_INDEX)

        ''' Cognitive ROM Decline 1 '''

        ''' Report '''
        # Major Frailty Phenotypes
        self.FI = self.FRAILTY_INDEX
        self.AngR_Mean = ph1_rigidity
        self.PowR_PD = ph2_exhaustion
        self.FlexT_Mean = ph3_slowness
        self.FlexT_CV = ph4_steadiness_lack_flexion
        self.ExT_CV = ph5_steadiness_lack_extension

        # Video-based Frailty Metrics
        self.AngV_Mean = self.ANGULAR_VELOCITY_RANGE
        self.AngV_SD = np.std(ang_r_)
        self.AngV_CV = variation(ang_r_, axis=0)
        self.AngV_PD = self.pd_calculation(ang_r_)

        self.AngR_Mean = self.FLEXION_PHENOTYPE
        self.AngR_SD = np.std(flx_list)
        self.AngR_CV = variation(flx_list, axis=0)
        self.AngR_PD = self.pd_calculation(flx_list)

        self.PowR_Mean = np.mean([power_range_first_section, power_range_second_section])
        print('self.PowR_Mean: ', self.PowR_Mean)
        power_ranges = np.concatenate((self.POWER[power_first_max_peaks[top_k_max_first_peaks]] -
                                       self.POWER[power_first_min_peaks[top_k_min_first_peaks]],
                                       self.POWER[power_second_max_peaks[top_k_max_second_peaks]] -
                                       self.POWER[power_second_min_peaks[top_k_min_second_peaks]]), axis=0)
        self.PowR_SD = np.std(power_ranges)
        self.PowR_CV = variation(power_ranges)
        self.PowR_PD = self.pd_calculation(power_ranges)

        self.RiseT_Mean = self.RISING_TIME_AVG
        self.RiseT_SD = np.std(rise_time_list)
        self.RiseT_CV = variation(rise_time_list)
        self.RiseT_PD = self.pd_calculation(rise_time_list)

        self.FallT_Mean = self.FALLING_TIME_AVG
        self.FallT_SD = np.std(ftv_list)
        self.FallT_CV = variation(ftv_list)
        self.FallT_PD = self.pd_calculation(ftv_list)

        self.RFT_Mean = self.RISING_TIME_PLUS_FALLING_TIME
        self.RFT_SD = np.std(rft_list)
        self.RFT_CV = variation(rft_list)
        self.RFT_PD = self.pd_calculation(rft_list)

        self.FlexT_Mean = self.FLEXION_TIME
        self.FlexT_SD = np.std(rst_list)
        self.FlexT_CV = variation(rst_list)
        self.FlexT_PD = self.pd_calculation(rst_list)

        self.ExT_Mean = self.EXTENSION_TIME
        self.ExT_SD = np.std(ft_list)
        self.ExT_CV = variation(ft_list)
        self.ExT_PD = self.pd_calculation(ft_list)

        self.FET_Mean = self.FLEXION_TIME_PLUS_EXTENSION_TIME
        self.FET_SD = np.std(fet_list)
        self.FET_CV = variation(fet_list)
        self.FET_PD = self.pd_calculation(fet_list)
        self.FER = self.FLEXION_EXTENSION_RATE
        self.FEN = self.N_FLEXION_EXTENSION

        # New parameters
        self.cognitive_parameters()
        # Cognitive parameter - Range of Motion (Flexibility) Outliers Rate
        self.CROR = self.COGNITIVE_ROM_DECLINE_RATE_1
        # Cognitive parameter - Range of Motion (Flexibility) Linear Regression Residuals Sum Per Iteration
        self.CRLRSR = self.COGNITIVE_ROM_LR_RESIDUAL_SUM
        # Cognitive parameter - Range of Motion (Flexibility) Linear Regression Slope
        self.CRLS = self.COGNITIVE_ROM_LR_SLOPE

        # Cognitive parameter - Pauses Outliers Rate
        self.CPOR = self.COGNITIVE_IRREGULAR_PAUSES_RATE_1
        # Cognitive parameter - Pauses Linear Regression Residuals Sum Per Iteration
        self.CPLRSR = self.COGNITIVE_IRREGULAR_PAUSES_LR_RESIDUAL_SUM
        # Cognitive parameter - Pauses Linear Regression Slope
        self.CPLS = self.COGNITIVE_IRREGULAR_PAUSES_LR_SLOPE

        '''Angular Acceleration update'''
        ############## Angular Acceleration update ###############
        print('FER Rate:')
        print(self.FER)
        adjusted_threshold = 0.1
        if self.FER >= 100:
            adjusted_threshold = 0.2
        elif self.FER <= 50:
            adjusted_threshold = 0.1
        else:
            adjusted_threshold = 0.1 + 0.002 * (self.FER - 50)
        print('adjusted_threshold')
        print(adjusted_threshold)
        N = 3  # Filter order
        Wn = float(adjusted_threshold)  # Cutoff frequency
        B, A = signal.butter(N, Wn, 'low')
        # print(B)
        # print(A)
        # print(self.signal)
        signal_denoised = signal.filtfilt(B, A, self.signal)
        velocity_vector = np.diff(signal_denoised) / np.diff(self.cali_list)
        velocity_vector = np.append(velocity_vector, velocity_vector[-1])

        # Second derivative - Angular Acceleration
        acceleration_vector = np.diff(velocity_vector) / np.diff(self.cali_list)
        acceleration_vector = np.append(acceleration_vector, acceleration_vector[-1])

        # plt.figure(figsize=(8, 2))
        # plt.plot(self.cali_list, velocity_vector)
        # plt.title('Velocity Vector')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Angular Velocity\n(deg\sec)')
        # plt.grid(False)
        # plt.tight_layout()
        # plt.show()
        #
        # plt.figure(figsize=(8, 2))
        # plt.plot(self.cali_list, acceleration_vector)
        # plt.title('Acceleration Vector')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Angular Acceleration\n(deg\sec^2)')
        # plt.grid(False)
        # plt.tight_layout()
        # plt.show()
        # print('Acceleration vector mean:')
        # print(np.mean(acceleration_vector))
        #
        # # Create a DataFrame
        # df = pd.DataFrame({
        #     'Time': self.cali_list,
        #     'Angular Velocity': velocity_vector
        # })
        #
        # # Save to Excel
        # df.to_excel('output.xlsx', index=False)
        #
        # # Calculate indices for the 20th and 80th percentiles
        # p20_index = int(np.floor(20 / 100.0 * len(acceleration_vector)))
        # p80_index = int(np.floor(80 / 100.0 * len(acceleration_vector)))
        # # Sort the array
        # sorted_array = np.sort(acceleration_vector)
        # # Slice the array from 20th to 80th percentile
        # sliced_array = sorted_array[p20_index:p80_index]
        # # Calculate the average
        # average = np.mean(sliced_array)
        # print('Acceleration vector mean 20 to 80 percentile:')
        # print(average)
        # print('Acceleration vector median:')
        # print(np.median(acceleration_vector))

        # ''' Flexion acceleration peaks '''
        # plt.figure(figsize=(8, 2))
        # plt.title('Acceleration Vector')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Angular Acceleration\n(deg\sec^2)')
        # plt.plot(self.cali_list, acceleration_vector, label='Acceleration signal')
        # ''' Max peaks '''
        # plt.plot(self.cali_list[self.max_peaks[self.top_peaks_idx_max]],
        #          acceleration_vector[self.max_peaks[self.top_peaks_idx_max]], 'gv',
        #          label='Max peaks')
        # plt.scatter(self.cali_list[self.max_peaks[self.top_peaks_idx_max]],
        #             acceleration_vector[self.max_peaks[self.top_peaks_idx_max]], s=88,
        #             facecolors='none', edgecolors='r')
        #
        # ''' Min peaks '''
        # plt.plot(self.cali_list[self.min_peaks[self.top_peaks_idx_min]],
        #          acceleration_vector[self.min_peaks[self.top_peaks_idx_min]], 'b^',
        #          label='Min peaks')
        # plt.scatter(self.cali_list[self.min_peaks[self.top_peaks_idx_min]],
        #             acceleration_vector[self.min_peaks[self.top_peaks_idx_min]], s=88,
        #             facecolors='none', edgecolors='r')
        # plt.tight_layout()
        # plt.show()
        #
        # ''' Velocity '''
        # plt.figure(figsize=(8, 2))
        # plt.title('Velocity Vector')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Angular Acceleration\n(deg\sec^2)')
        # plt.plot(self.cali_list, velocity_vector, label='Acceleration signal')
        # ''' Max peaks '''
        # plt.plot(self.cali_list[self.max_peaks[self.top_peaks_idx_max]],
        #          velocity_vector[self.max_peaks[self.top_peaks_idx_max]], 'gv',
        #          label='Max peaks')
        # plt.scatter(self.cali_list[self.max_peaks[self.top_peaks_idx_max]],
        #             velocity_vector[self.max_peaks[self.top_peaks_idx_max]], s=88,
        #             facecolors='none', edgecolors='r')
        #
        # ''' Min peaks '''
        # plt.plot(self.cali_list[self.min_peaks[self.top_peaks_idx_min]],
        #          velocity_vector[self.min_peaks[self.top_peaks_idx_min]], 'b^',
        #          label='Min peaks')
        # plt.scatter(self.cali_list[self.min_peaks[self.top_peaks_idx_min]],
        #             velocity_vector[self.min_peaks[self.top_peaks_idx_min]], s=88,
        #             facecolors='none', edgecolors='r')
        # plt.tight_layout()
        # plt.show()

        ''' Average between angular acceleration for flexion '''
        # print('PEAKS')
        roi_velocity_vector = velocity_vector[
                              self.max_peaks[self.top_peaks_idx_max[0]]:self.max_peaks[self.top_peaks_idx_max[-1]]]
        roi_acceleration_vector = acceleration_vector[
                                  self.max_peaks[self.top_peaks_idx_max[0]]:self.max_peaks[self.top_peaks_idx_max[-1]]]
        negative = roi_velocity_vector < 0

        # Find transitions from positive to negative and negative to positive
        transitions = np.diff(negative.astype(int))

        # Start indexes: where transition goes from False to True (0 to 1)
        start_indexes = np.where(transitions == 1)[0] + 1

        # End indexes: where transition goes from True to False (1 to 0)
        end_indexes = np.where(transitions == -1)[0]

        # Check if the first negative is at the start of the array
        if negative[0]:
            start_indexes = np.insert(start_indexes, 0, 0)

        # Check if the array ends while still being negative
        if negative[-1]:
            end_indexes = np.append(end_indexes, len(roi_velocity_vector) - 1)

        # Match starts and ends, making sure each cycle is complete
        cycles = []
        for start in start_indexes:
            end = end_indexes[end_indexes > start]
            if len(end) > 0:
                cycles.append((start, end[0]))
        average_acc_flexion = []
        for start_idx, end_idx in cycles:
            # The purpose of start_idx - 1: Because when you make diff, you get one element less in your array.
            average_acc_flexion.append(np.mean(roi_acceleration_vector[start_idx - 1:end_idx]))
        # print(average_acc_flexion)
        import math
        average_acc_flexion = [x for x in average_acc_flexion if not math.isnan(x)]
        # print(np.mean(average_acc_flexion))
        # print(np.median(average_acc_flexion))

        # Create a DataFrame
        data = {'Subject ID': [self.patient_title],
                'Mean of Angular acceleration of flexion (Deg/S2)': [np.mean(average_acc_flexion)],
                'Median of Angular acceleration of flexion (Deg/S2)': [np.median(average_acc_flexion)]}
        dfaa = pd.DataFrame(data)

        # Save DataFrame to Excel file
        excel_file_path = rf"{self.angular_acc_folder_path}/{self.patient_title}_angular_acc_flexion.xlsx"  # Specify the path where you want to save the Excel file
        dfaa.to_excel(excel_file_path, index=False)

    def dt_cost_calculation(self):
        names_array = str(
            'self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD, self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD, self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD, self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD, self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD, self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD, self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD, self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD, self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD, self.FER, self.FEN, self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS')
        self.var_names = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]
        if self.DT_OUTPUT_FILE is not None and self.ST_OUTPUT_FILE is not None:
            self.file_name = self.DT_OUTPUT_FILE
            self.run_command(plot=True)
            dt_AngV_Mean = self.AngV_Mean
            self.results_array_DT = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                     self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                     self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                     self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                     self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                     self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                     self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                     self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                     self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                     self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                     self.FER, self.FEN,
                                     self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                     ]
            self.task_type = 'DT'
            self.var_names_DT = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]

            self.file_name = self.ST_OUTPUT_FILE
            self.run_command(plot=True)
            st_AngV_Mean = self.AngV_Mean
            self.results_array_ST = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                     self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                     self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                     self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                     self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                     self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                     self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                     self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                     self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                     self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                     self.FER, self.FEN,
                                     self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                     ]
            self.task_type = 'ST'
            self.var_names_ST = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]
            self.var_names = self.var_names_ST + self.var_names_DT + ['DT_Cost']

            self.DT_Cost = (1 - (dt_AngV_Mean / st_AngV_Mean)) * 100

            self.results_array = self.results_array_ST + self.results_array_DT + [self.DT_Cost]
            # print(dt_AngV_Mean)
            # print(st_AngV_Mean)
            # print(self.DT_Cost)
        else:
            self.run_command(plot=True)
            self.results_array = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                  self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                  self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                  self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                  self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                  self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                  self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                  self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                  self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                  self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                  self.FER, self.FEN,
                                  self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                  ]

    def run_and_save_results(self):
        self.dt_cost_calculation()

        ''' Save the results '''

        # if self.DT_Cost is not None:
        #
        # if self.DT_Cost is not None:
        #     results_array.append(self.DT_Cost)
        #     var_names.append('DT_Cost')
        # print(var_names)
        results_dict = {}
        results_list = []
        for i in range(len(self.var_names)):
            results_dict[self.var_names[i]] = self.results_array[i]
            results_list.append([self.var_names[i], self.results_array[i]])
        # print(results_dict)

        results_list = np.array(results_list).T
        # print(results_list)
        df = pd.DataFrame(results_list[1].reshape(1, -1), columns=list(results_list).pop(0))
        # df.index = self.patient_title
        print(df)
        if self.DT_OUTPUT_FILE is not None and self.ST_OUTPUT_FILE is not None:
            df.to_excel('{}//results_all_{}.xlsx'.format(self.results_location, self.patient_title))
        else:
            df.to_excel('{}//results_{}.xlsx'.format(self.results_location, self.patient_title))

    def pd_calculation(self, arrr):
        arrr = np.array(arrr)
        arr1 = arrr[:(len(arrr) // 2)]
        arr2 = arrr[(len(arrr) // 2):]
        pdd = np.round((1 - (np.mean(arr2) / np.mean(arr1))) * 100, decimals=3)
        return pdd

    def cognitive_parameters(self):
        gaps_vertical = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            gaps = np.abs(self.signal[self.max_peaks[self.top_peaks_idx_max[i + 1]]] - \
                          self.signal[self.max_peaks[self.top_peaks_idx_max[i]]])
            gaps_vertical.append(gaps)
        gaps_vertical = np.array(gaps_vertical)
        outliers = 0
        for i in range(len(gaps_vertical)):
            if gaps_vertical[i] >= 15:
                outliers += 1
        self.COGNITIVE_ROM_DECLINE_RATE_1 = outliers / len(gaps_vertical)

        # print(self.COGNITIVE_ROM_DECLINE_RATE_1)

        gaps_horizontal = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            gaps = np.abs(self.cali_list[self.max_peaks[self.top_peaks_idx_max[i + 1]]] - \
                          self.cali_list[self.max_peaks[self.top_peaks_idx_max[i]]])
            gaps_horizontal.append(gaps)
        gaps_horizontal = np.array(gaps_horizontal)
        interval = t.interval(0.9999, len(gaps_horizontal) - 1, loc=np.mean(gaps_horizontal),
                              scale=sem(gaps_horizontal))

        outliers = 0
        for i in range(len(gaps_horizontal)):
            if gaps_horizontal[i] < interval[1]:
                pass
            else:
                outliers += 1
        # print(outliers)
        self.COGNITIVE_IRREGULAR_PAUSES_RATE_1 = outliers / len(gaps_horizontal)
        # print(self.COGNITIVE_IRREGULAR_PAUSES_RATE_1)

        # Linear Regression
        X = gaps_vertical.reshape(-1, 1)
        y = np.arange(len(X))
        reg = LinearRegression().fit(X, y)
        self.COGNITIVE_ROM_LR_RESIDUAL_SUM = reg.score(X, y) / self.N_FLEXION_EXTENSION
        self.COGNITIVE_ROM_LR_SLOPE = reg.coef_[0]
        # print(self.COGNITIVE_ROM_LR_RESIDUAL_SUM)
        # print(self.COGNITIVE_ROM_LR_SLOPE)

        # Horizontal
        X = gaps_horizontal.reshape(-1, 1)
        y = np.arange(len(X))
        reg = LinearRegression().fit(X, y)
        self.COGNITIVE_IRREGULAR_PAUSES_LR_RESIDUAL_SUM = reg.score(X, y) / self.N_FLEXION_EXTENSION
        self.COGNITIVE_IRREGULAR_PAUSES_LR_SLOPE = reg.coef_[0]
        # print(self.COGNITIVE_IRREGULAR_PAUSES_LR_RESIDUAL_SUM)
        # print(self.COGNITIVE_IRREGULAR_PAUSES_LR_SLOPE)

    def start_end_signal_peak(self):
        if self.cali_list[self.max_peaks[self.top_peaks_idx_max[0]]] < self.cali_list[
            self.min_peaks[self.top_peaks_idx_min[0]]]:
            self.signal_start_peak = 'Max'
        else:
            self.signal_start_peak = 'min'

        if self.cali_list[self.max_peaks[self.top_peaks_idx_max[-1]]] > self.cali_list[
            self.min_peaks[self.top_peaks_idx_min[-1]]]:
            self.signal_end_peak = 'Max'
        else:
            self.signal_end_peak = 'min'

    def sensor_video_validation(self):
        df = pd.read_excel(self.sensor_data_location, sheet_name='Raw Data', header=None, skiprows=1)
        df = df.to_numpy().reshape(1, -1)[0]

        ''' Manipulation '''
        # filename = r"Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\TEL007\BL\FM\DT.txt"
        # # file = np.loadtxt(filename)
        # # print(file)
        #
        # with open(filename) as file:
        #     lines = file.readlines()
        #     lines = [line.rstrip().split(' ') for line in lines]
        #     lines = [list(filter(None, line))[0] for line in lines]
        #     lines = [i.split(',') for i in lines]
        #     lines = [float(list(filter(None, line))[0]) for line in lines]
        # df = np.array(lines).reshape(1, -1)[0]

        x_axis = np.arange(len(df)) * (1 / 25) - 0.7
        # x_axis *= 1.001
        B, A = signal.butter(5, 0.2, 'low')
        df = signal.filtfilt(B, A, df)

        # print(x_axis)

        # idx_f = (np.abs(self.cali_list - x_axis[0])).argmin()
        # idx_l = (np.abs(self.cali_list - x_axis[-1])).argmin()
        # print(video_signal.shape)
        self.smooth_func(tresh=0.105)
        self.signal_filtered_derivative = np.diff(self.signal_after_butterworth) / np.diff(self.cali_list)
        self.signal_filtered_derivative = np.append(self.signal_filtered_derivative,
                                                    self.signal_filtered_derivative[-1])

        new_x = np.arange(x_axis[0], x_axis[-1], (x_axis[-1] - x_axis[0]) / 575)
        new_y = np.zeros(len(new_x))
        for i in range(len(new_x)):
            lp_ixd = np.where(self.cali_list - new_x[i] > 0, self.cali_list - new_x[i], np.inf).argmin()
            if lp_ixd != 0:
                x2 = self.cali_list[lp_ixd]
                y2 = self.signal_filtered_derivative[lp_ixd]
                x1 = self.cali_list[lp_ixd - 1]
                y1 = self.signal_filtered_derivative[lp_ixd - 1]
                new_y[i] = ((y2 - y1) / (x2 - x1)) * (new_x[i] - x1) + y1
            else:
                idx_value = (np.abs(self.cali_list - new_x[i])).argmin()
                new_y[i] = self.signal_filtered_derivative[idx_value]
            # print(self.cali_list[idx_value])
        print(new_x)
        print(new_y)
        df_t = pd.DataFrame()
        df_t['Time'] = x_axis
        df_t['Sensor_Signal'] = -df
        df_t['Video_Signal'] = new_y
        df_t.to_excel('{}//SV_{}.xlsx'.format(self.sensor_video_file_location, self.patient_title))
        print(df_t)

        # # print(x_axis)
        # plt.figure(figsize=(12, 6))
        # # plt.ylim([-500, 400])
        # # plt.xlim([0, 1600])
        # plt.title(self.patient_title)
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Angle (Degrees$^\circ$ / Sec)')
        # # plt.plot(x_axis, -df, '--', label='Sensor signal')
        #
        # plt.plot(new_x, -df, '--', label='Sensor signal')
        # # plt.plot(self.cali_list[idx_f:idx_l], self.signal_filtered_derivative[idx_f:idx_l], label='Video signal')
        #
        # plt.plot(new_x, new_y, label='Video signal')
        #
        # plt.legend()
        # plt.show()

    def run_sensor_analysis(self):
        df = pd.read_excel(self.sensor_data_location, sheet_name='Raw Data', header=None, skiprows=1)
        df = df.to_numpy().reshape(1, -1)[0]
        self.signal = df
        x = np.arange(len(self.signal))
        z = cumtrapz(self.signal, x, initial=0) + 0
        # plt.plot(x, self.signal)
        # plt.plot(x, z)
        # plt.show()

    def sensor_based(self):
        df = pd.read_excel(self.sensor_data_location, sheet_name='Raw Data', header=None, skiprows=1)
        df = df.to_numpy().reshape(1, -1)[0]

        ''' Manipulation '''
        # filename = "Z:\Projects BCM\H-43917 Image based Frailty\Phase II\Data\Raw Sensor\TEL034\BL\FM\ST.txt"
        # # file = np.loadtxt(filename)
        # # print(file)
        #
        # with open(filename) as file:
        #     lines = file.readlines()
        #     lines = [line.rstrip().split(' ') for line in lines]
        #     lines = [list(filter(None, line))[0] for line in lines]
        #     lines = [i.split(',') for i in lines]
        #     lines = [float(list(filter(None, line))[0]) for line in lines]
        # df = np.array(lines).reshape(1, -1)[0]

        # self.critical_points()
        # x_axis *= 1.001
        # B, A = signal.butter(5, 0.4, 'low')
        # df = signal.filtfilt(B, A, df)
        # print(df)

        self.signal = -df
        self.fps = 25
        self.time_calibration()

        ''' Max peaks '''
        self.max_peaks, sth = find_peaks(self.signal, width=self.WIDTH_NUM)
        output = peak_prominences(self.signal, self.max_peaks)
        self.prominences = output[0]
        self.prominences_idx = output[1]
        self.top_peaks_idx_max = np.where(self.prominences > 60)[0]
        self.prominences = self.prominences[self.top_peaks_idx_max]
        self.prominences_idx = self.prominences_idx[self.top_peaks_idx_max]

        ''' Remove max lower than 100 '''
        updated_list_idx = []
        for i in range(len(self.top_peaks_idx_max)):
            if self.signal[self.max_peaks[self.top_peaks_idx_max[i]]] <= 100:
                updated_list_idx.append(i)
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, updated_list_idx)

        ''' Min peaks '''
        reversed_signal = self.reverse_signal()
        self.min_peaks, sth = find_peaks(reversed_signal, width=self.WIDTH_NUM)
        output = peak_prominences(reversed_signal, self.min_peaks)
        self.prominences = output[0]
        self.prominences_idx = output[1]
        self.top_peaks_idx_min = np.where(self.prominences > 60)[0]
        self.prominences = self.prominences[self.top_peaks_idx_min]
        self.prominences_idx = self.prominences_idx[self.top_peaks_idx_min]
        self.infls = None

        ''' Remove min higher than -100 '''
        updated_list_idx = []
        for i in range(len(self.top_peaks_idx_min)):
            if self.signal[self.min_peaks[self.top_peaks_idx_min[i]]] > -100:
                updated_list_idx.append(i)
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, updated_list_idx)

        ''' Remove redundant minimum points between two Max points '''
        redundant_min_idx = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            try:
                first_val = self.max_peaks[self.top_peaks_idx_max[i]]
                second_val = self.max_peaks[self.top_peaks_idx_max[i + 1]]
                boolean_array = np.logical_and(self.min_peaks[self.top_peaks_idx_min] >= first_val,
                                               self.min_peaks[self.top_peaks_idx_min] <= second_val)

                in_range_indices = np.where(boolean_array)[0]

                if len(in_range_indices) > 1:
                    global_min = np.argmin(self.signal[self.min_peaks[self.top_peaks_idx_min[in_range_indices]]])
                    for j in range(len(in_range_indices)):
                        redundant_min_idx.append(in_range_indices[j])
                    index = np.argwhere(redundant_min_idx == in_range_indices[global_min])
                    redundant_min_idx = list(np.delete(redundant_min_idx, index))
                # if len(in_range_indices) < 1:
                #     if first_val > second_val:

            except Exception as e:
                print(str(e))
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, redundant_min_idx)

        ''' Remove redundant Maximum points between two min points '''
        redundant_max_idx = []
        for i in range(len(self.top_peaks_idx_min) - 1):
            try:
                first_val = self.min_peaks[self.top_peaks_idx_min[i]]
                second_val = self.min_peaks[self.top_peaks_idx_min[i + 1]]
                boolean_array = np.logical_and(self.max_peaks[self.top_peaks_idx_max] >= first_val,
                                               self.max_peaks[self.top_peaks_idx_max] <= second_val)
                in_range_indices = np.where(boolean_array)[0]
                if len(in_range_indices) > 1:
                    global_max = np.argmax(self.signal[self.max_peaks[self.top_peaks_idx_max[in_range_indices]]])

                    for j in range(len(in_range_indices)):
                        redundant_max_idx.append(in_range_indices[j])
                    index = np.argwhere(redundant_max_idx == in_range_indices[global_max])
                    redundant_max_idx = list(np.delete(redundant_max_idx, index))
            except Exception as e:
                print(str(e))
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, redundant_max_idx)

        ''' Remove redundant Max and Min points of beginning and ending of the signal '''
        redundant_min_idx = []
        for i in range(len(self.top_peaks_idx_min) - 1):
            first_val = self.min_peaks[self.top_peaks_idx_min[i]]
            second_val = self.min_peaks[self.top_peaks_idx_min[i + 1]]
            boolean_array = np.logical_and(self.max_peaks[self.top_peaks_idx_max] >= first_val,
                                           self.max_peaks[self.top_peaks_idx_max] <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            if len(in_range_indices) < 1:
                redundant_min_idx.append(self.top_peaks_idx_min[i])
            else:
                self.top_peaks_idx_min = np.setdiff1d(self.top_peaks_idx_min, redundant_min_idx)
                break
        redundant_max_idx = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            first_val = self.max_peaks[self.top_peaks_idx_max[i]]
            second_val = self.max_peaks[self.top_peaks_idx_max[i + 1]]
            boolean_array = np.logical_and(self.min_peaks[self.top_peaks_idx_min] >= first_val,
                                           self.min_peaks[self.top_peaks_idx_min] <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            if len(in_range_indices) < 1:
                redundant_max_idx.append(self.top_peaks_idx_max[i])
            else:
                self.top_peaks_idx_max = np.setdiff1d(self.top_peaks_idx_max, redundant_max_idx)
                break

        ''' Detect zero crossing points '''
        all_critical_points = np.concatenate(
            (self.max_peaks[self.top_peaks_idx_max], self.min_peaks[self.top_peaks_idx_min]))
        # print(all_critical_points)
        all_critical_points = np.sort(all_critical_points)
        all_inf_t_vals = []
        for i in range(len(all_critical_points) - 1):
            arr = self.signal[all_critical_points[i]:all_critical_points[i + 1]]
            arr_time = self.cali_list[all_critical_points[i]:all_critical_points[i + 1]]
            # print(arr)
            # print(arr_time)
            idx1 = np.where(arr < 0, arr, -np.inf).argmax()
            idx2 = np.where(arr > 0, arr, np.inf).argmin()
            if idx1 == idx2:
                x1 = arr_time[idx1]
                all_inf_t_vals.append(x1)
            else:
                x1 = arr_time[idx1]
                x2 = arr_time[idx2]
                y1 = arr[idx1]
                y2 = arr[idx2]
                slope = ((y2 - y1) / (x2 - x1))
                # y - y1 = slope * (x - x1)
                x = ((0 - y1) / slope) + x1
                x = np.round(x, decimals=4)
                all_inf_t_vals.append(x)

        ''' Fix min-max indices '''
        for index, item in enumerate(all_inf_t_vals):
            indices = np.searchsorted(self.cali_list, item)
            self.cali_list = np.insert(self.cali_list, indices, item)
            self.signal = np.insert(self.signal, indices, 0)
            # if self.signal[indices + 1] > 0:
            # print('*' * 100)
            # print(self.min_peaks[[self.top_peaks_idx_min]])
            # print(self.min_peaks[[self.top_peaks_idx_min]][index + 1:])
            if self.max_peaks[self.top_peaks_idx_max[0]] < self.min_peaks[self.top_peaks_idx_min[0]]:
                self.min_peaks[[self.top_peaks_idx_min[index:]]] += 1
                self.min_peaks[[self.top_peaks_idx_min[index + 1:]]] += 1
                self.max_peaks[[self.top_peaks_idx_max[index + 1:]]] += 2
            else:
                self.max_peaks[[self.top_peaks_idx_max[index:]]] += 1
                self.max_peaks[[self.top_peaks_idx_max[index + 1:]]] += 1
                self.min_peaks[[self.top_peaks_idx_min[index + 1:]]] += 2

        all_idx = []
        for index, item in enumerate(all_inf_t_vals):
            indices = np.where(item == self.cali_list)[0][0]
            all_idx.append(indices)
        all_idx = np.array(all_idx)
        self.infls = np.copy(all_idx)

        ''' Phenotypes '''
        self.start_end_signal_peak()

        ''' Angular Velocity Range '''
        if self.signal_end_peak is 'min':
            end_adjuster = 0
        else:
            end_adjuster = 1

        flx_list = []
        for i in range(len(self.top_peaks_idx_max) - end_adjuster):
            flexion = self.signal[self.max_peaks[self.top_peaks_idx_max[i + end_adjuster]]] - self.signal[
                self.min_peaks[self.top_peaks_idx_min[i]]]
            flx_list.append(flexion)
        self.AngV_Mean = np.mean(flx_list)
        self.AngV_SD = np.std(flx_list)
        self.AngV_CV = variation(flx_list, axis=0)
        self.AngV_PD = self.pd_calculation(flx_list)
        # print(self.AngV_Mean)
        # print(self.AngV_SD)
        # print(self.AngV_CV)
        # print(self.AngV_PD)
        # print('*' * 100)

        ''' Flexion '''
        # print(self.cali_list[self.infls])
        a = 2
        b = 3
        flxs = []
        for i in range(len(self.infls) - 1):
            a = i
            b = i + 1
            angles = np.trapz(self.signal[self.infls[a]:self.infls[b]], x=self.cali_list[self.infls[a]:self.infls[b]])
            flxs.append(np.abs(angles))
        # print('Flexion: ', np.mean(flxs))
        # print(flxs)

        self.FLEXION_PHENOTYPE = np.mean(flxs)

        self.AngR_Mean = self.FLEXION_PHENOTYPE
        self.AngR_SD = np.std(flxs)
        self.AngR_CV = variation(flxs, axis=0)
        self.AngR_PD = self.pd_calculation(flxs)
        # print(self.AngR_Mean)
        # print(self.AngR_SD)
        # print(self.AngR_CV)
        # print(self.AngR_PD)

        ''' Frailty Index '''
        # Ph1
        ph1_rigidity = self.FLEXION_PHENOTYPE
        print('Rigidity: (self.AngR_Mean)', ph1_rigidity)

        ''' Power '''
        self.smooth_func(tresh=0.19)
        self.signal_filtered_derivative = np.diff(self.signal_after_butterworth) / np.diff(self.cali_list)
        self.signal_filtered_derivative = np.append(self.signal_filtered_derivative,
                                                    self.signal_filtered_derivative[-1])

        self.POWER = self.signal * self.signal_filtered_derivative

        ''' Update Power Calculations '''
        self.POWER = self.POWER / 25

        # print(self.POWER)

        # Ph2
        # print('+' * 100)
        mid_idx = len(self.signal[self.max_peaks[self.top_peaks_idx_max]]) // 2
        mid_idx = self.max_peaks[self.top_peaks_idx_max[mid_idx]]
        self.power_max_peaks, sth = find_peaks(self.POWER, width=self.WIDTH_NUM)
        self.power_min_peaks, sth = find_peaks(-self.POWER, width=self.WIDTH_NUM)
        # print(self.POWER)

        power_first_max_peaks, power_second_max_peaks = [self.power_max_peaks[self.power_max_peaks < mid_idx],
                                                         self.power_max_peaks[self.power_max_peaks >= mid_idx]]
        power_first_min_peaks, power_second_min_peaks = [self.power_min_peaks[self.power_min_peaks < mid_idx],
                                                         self.power_min_peaks[self.power_min_peaks >= mid_idx]]

        # print('+' * 100)
        # print(self.POWER)
        # print(power_first_max_peaks)
        # print(power_second_max_peaks)

        ''' Remove first outliers '''
        # print(self.min_peaks[self.top_peaks_idx_min])
        # print(power_first_max_peaks)
        idx = (np.abs(power_first_max_peaks - self.min_peaks[self.top_peaks_idx_min][1])).argmin()
        # print(power_first_max_peaks[idx])
        power_first_max_peaks = power_first_max_peaks[idx:]
        power_first_min_peaks = power_first_min_peaks[idx:]
        # print(power_first_max_peaks)

        k = np.min([len(power_first_max_peaks),
                    len(power_second_max_peaks),
                    len(power_first_min_peaks),
                    len(power_second_min_peaks)])
        # print(k)
        # k = len(power_first_max_peaks)
        top_k_max_first_peaks = np.argpartition(self.POWER[power_first_max_peaks], -k)[-k:]
        # print(top_k_max_first_peaks)
        # k = len(power_second_max_peaks)
        top_k_max_second_peaks = np.argpartition(self.POWER[power_second_max_peaks], -k)[-k:]

        # k = len(power_first_min_peaks)
        top_k_min_first_peaks = np.argpartition(-self.POWER[power_first_min_peaks], -k)[-k:]
        # k = len(power_second_min_peaks)
        top_k_min_second_peaks = np.argpartition(-self.POWER[power_second_min_peaks], -k)[-k:]

        power_range_first_section = np.mean(self.POWER[power_first_max_peaks[top_k_max_first_peaks]]) - np.mean(
            self.POWER[power_first_min_peaks[top_k_min_first_peaks]])
        power_range_second_section = np.mean(self.POWER[power_second_max_peaks[top_k_max_second_peaks]]) - np.mean(
            self.POWER[power_second_min_peaks[top_k_min_second_peaks]])
        # print(power_range_first_section, power_range_second_section)
        self.POWER_DECLINE_PERCENTAGE = np.round((1 - (power_range_second_section / power_range_first_section)) * 1,
                                                 decimals=3)
        ph2_exhaustion = self.POWER_DECLINE_PERCENTAGE

        print('Exhaustion: (self.PowR_PD)', ph2_exhaustion)

        self.PowR_Mean = np.mean([power_range_first_section, power_range_second_section])
        # print('self.PowR_Mean: ', self.PowR_Mean)
        power_ranges = np.concatenate((self.POWER[power_first_max_peaks[top_k_max_first_peaks]] -
                                       self.POWER[power_first_min_peaks[top_k_min_first_peaks]],
                                       self.POWER[power_second_max_peaks[top_k_max_second_peaks]] -
                                       self.POWER[power_second_min_peaks[top_k_min_second_peaks]]), axis=0)
        self.PowR_SD = np.std(power_ranges)
        self.PowR_CV = variation(power_ranges)
        self.PowR_PD = self.pd_calculation(power_ranges)
        # print('PowR_SD: ', self.PowR_SD)
        # print('*' * 100)
        # print(self.PowR_Mean)
        # print(self.PowR_SD)
        # print(self.PowR_CV)
        # print(self.PowR_PD)

        ''' Rising time '''
        # self.max_peaks[self.top_peaks_idx_max], self.min_peaks[self.top_peaks_idx_min]
        # all_critical_points
        max_first = False
        if self.max_peaks[self.top_peaks_idx_max][0] < self.min_peaks[self.top_peaks_idx_min][0]:
            max_first = True
        # print(max_first)

        rise_time_list = []
        if max_first:
            for i in range(len(self.max_peaks[self.top_peaks_idx_max]) - 1):
                value = self.cali_list[self.max_peaks[self.top_peaks_idx_max][i + 1]] - self.cali_list[
                    self.infls[2 * i + 1]]
                rise_time_list.append(value)
        else:
            for i in range(len(self.max_peaks[self.top_peaks_idx_max]) - 1):
                value = self.cali_list[self.max_peaks[self.top_peaks_idx_max][i]] - self.cali_list[self.infls[2 * i]]
                rise_time_list.append(value)

        self.RiseT_Mean = np.mean(rise_time_list)
        self.RiseT_SD = np.std(rise_time_list)
        self.RiseT_CV = variation(rise_time_list)
        self.RiseT_PD = self.pd_calculation(rise_time_list)
        # print('*' * 50)
        # print('RiseT_Mean: ', self.RiseT_Mean)
        # print(self.RiseT_SD)
        # print(self.RiseT_CV)
        # print(self.RiseT_PD)

        ''' Falling time '''
        ftv_list = []
        if not max_first:
            for i in range(len(self.min_peaks[self.top_peaks_idx_min]) - 1):
                # print(i)
                value = self.cali_list[self.min_peaks[self.top_peaks_idx_min][i + 1]] - self.cali_list[
                    self.infls[2 * i + 1]]
                ftv_list.append(value)
        else:
            for i in range(len(self.min_peaks[self.top_peaks_idx_min]) - 1):
                value = self.cali_list[self.min_peaks[self.top_peaks_idx_min][i]] - self.cali_list[self.infls[2 * i]]
                ftv_list.append(value)
        # print(ftv_list)

        self.FallT_Mean = np.mean(ftv_list)
        self.FallT_SD = np.std(ftv_list)
        self.FallT_CV = variation(ftv_list)
        self.FallT_PD = self.pd_calculation(ftv_list)

        ''' Rising + Falling time'''
        if len(ftv_list) > len(rise_time_list):
            counter = len(rise_time_list)
        else:
            counter = len(ftv_list)
        rft_list = []
        for i in range(counter):
            rft_list.append(rise_time_list[i] + ftv_list[i])
        rft_list = np.array(rft_list)

        self.RISING_TIME_PLUS_FALLING_TIME = self.RiseT_Mean + self.FallT_Mean

        self.RFT_Mean = self.RISING_TIME_PLUS_FALLING_TIME
        self.RFT_SD = np.std(rft_list)
        self.RFT_CV = variation(rft_list)
        self.RFT_PD = self.pd_calculation(rft_list)

        ''' Flexion time '''
        flex_list = []
        exten_list = []
        swing_flag = None
        if max_first:
            swing_flag = False
        else:
            swing_flag = True
        for i in range(len(self.infls) - 1):
            value = self.cali_list[self.infls[i + 1]] - self.cali_list[self.infls[i]]
            # print(value)
            if swing_flag:
                flex_list.append(value)
                swing_flag = False
            else:
                exten_list.append(value)
                swing_flag = True
        # print(flex_list)
        # print(exten_list)
        self.FlexT_Mean = np.mean(flex_list)
        self.FlexT_SD = np.std(flex_list)
        self.FlexT_CV = variation(flex_list)
        self.FlexT_PD = self.pd_calculation(flex_list)

        self.ExT_Mean = np.mean(exten_list)
        self.ExT_SD = np.std(exten_list)
        self.ExT_CV = variation(exten_list)
        self.ExT_PD = self.pd_calculation(exten_list)

        if len(exten_list) > len(flex_list):
            counter = len(flex_list)
        else:
            counter = len(exten_list)
        fet_list = []
        for i in range(counter):
            fet_list.append(flex_list[i] + exten_list[i])
        fet_list = np.array(fet_list)

        self.FET_Mean = self.FlexT_Mean + self.ExT_Mean
        self.FET_SD = np.std(fet_list)
        self.FET_CV = variation(fet_list)
        self.FET_PD = self.pd_calculation(fet_list)

        # print(self.FET_Mean)
        # print(self.FET_SD)
        # print(self.FET_CV)
        # print(self.FET_PD)

        ''' Flexion/Extension Rate '''
        duration = self.cali_list[self.max_peaks[self.top_peaks_idx_max[-1]]] - self.cali_list[
            self.max_peaks[self.top_peaks_idx_max[0]]]
        self.N_FLEXION_EXTENSION = len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]])
        # print('self.N_FLEXION_EXTENSION', self.N_FLEXION_EXTENSION)
        self.N_FLEXION_EXTENSION = int(
            np.round(len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]]) * (20.0 / duration)))
        # print(duration)
        # print((20.0 / duration))
        # print('self.N_FLEXION_EXTENSION', self.N_FLEXION_EXTENSION)
        self.FLEXION_EXTENSION_RATE = (self.N_FLEXION_EXTENSION / duration) * 60

        self.FER = self.FLEXION_EXTENSION_RATE
        self.FEN = self.N_FLEXION_EXTENSION

        # New parameters
        # self.cognitive_parameters()
        self.CROR = 0
        self.CRLRSR = 0
        self.CRLS = 0
        self.CPOR = 0
        self.CPLRSR = 0
        self.CPLS = 0

        ''' Frailty Index calc'''
        ph3_slowness = self.FlexT_Mean * 1000
        print('Slowness: (self.FlexT_Mean)', ph3_slowness)

        ph4_steadiness_lack_flexion = self.FlexT_CV
        print('Steadiness_lack_flexion: (self.FlexT_CV)', ph4_steadiness_lack_flexion)

        ph5_steadiness_lack_extension = self.ExT_CV
        print('Steadiness_lack_extension: (self.ExT_CV)', ph5_steadiness_lack_extension)

        b = 0.24495
        a1 = -1.7357 * 0.001
        a2 = -1.2026 * 0.001
        a3 = 0.36848 * 0.001
        a4 = -0.49396
        a5 = 0.48974
        self.FRAILTY_INDEX = b + a1 * ph1_rigidity + a2 * ph2_exhaustion + a3 * ph3_slowness + \
                             a4 * ph4_steadiness_lack_flexion + a5 * ph5_steadiness_lack_extension
        print('FRAILTY_INDEX: ', self.FRAILTY_INDEX)
        self.FI = self.FRAILTY_INDEX
        # # Cognitive parameter - Range of Motion (Flexibility) Outliers Rate
        # self.CROR = self.COGNITIVE_ROM_DECLINE_RATE_1
        # # Cognitive parameter - Range of Motion (Flexibility) Linear Regression Residuals Sum Per Iteration
        # self.CRLRSR = self.COGNITIVE_ROM_LR_RESIDUAL_SUM
        # # Cognitive parameter - Range of Motion (Flexibility) Linear Regression Slope
        # self.CRLS = self.COGNITIVE_ROM_LR_SLOPE
        #
        # # Cognitive parameter - Pauses Outliers Rate
        # self.CPOR = self.COGNITIVE_IRREGULAR_PAUSES_RATE_1
        # # Cognitive parameter - Pauses Linear Regression Residuals Sum Per Iteration
        # self.CPLRSR = self.COGNITIVE_IRREGULAR_PAUSES_LR_RESIDUAL_SUM
        # # Cognitive parameter - Pauses Linear Regression Slope
        # self.CPLS = self.COGNITIVE_IRREGULAR_PAUSES_LR_SLOPE

        ''' Plot '''
        # # x_axis = np.arange(len(df)) * (1 / 25)
        # # df_t = pd.DataFrame()
        # # df_t['Time'] = x_axis
        # # df_t['Sensor_Signal'] = -df
        # plt.title(self.patient_title)
        # plt.xlabel('Time')
        # plt.ylabel('Angle (Degrees$^\circ$ / Sec)')
        # # plt.plot(x_axis, -df, '--', label='Sensor signal')
        #
        # # plt.plot(x_axis, -df, label='Sensor signal')
        #
        # plt.plot(self.cali_list, self.signal, label='Sensor signal')
        # plt.legend()
        # plt.show()
        self.plot_signal()

    def run_and_save_results_sensor(self):

        self.sensor_based()

        self.dt_cost_calculation_sensor()

        ''' Save the results '''

        # if self.DT_Cost is not None:
        #
        # if self.DT_Cost is not None:
        #     results_array.append(self.DT_Cost)
        #     var_names.append('DT_Cost')
        # print(var_names)
        results_dict = {}
        results_list = []
        for i in range(len(self.var_names)):
            results_dict[self.var_names[i]] = self.results_array[i]
            results_list.append([self.var_names[i], self.results_array[i]])
        # print(results_dict)

        results_list = np.array(results_list).T
        # print(results_list)
        df = pd.DataFrame(results_list[1].reshape(1, -1), columns=list(results_list).pop(0))
        # df.index = self.patient_title
        # print(df)
        # print('Yes' * 100)
        # print(self.file_name)
        self.patient_title = self.file_name.split('\\')[-1].split('.')[0]

        if self.DT_OUTPUT_FILE is not None and self.ST_OUTPUT_FILE is not None:
            df.to_excel('{}//sensor_results_all_{}.xlsx'.format(self.results_location, self.patient_title))
        else:
            df.to_excel('{}//sensor_results_{}.xlsx'.format(self.results_location, self.patient_title))

    def dt_cost_calculation_sensor(self):
        names_array = str(
            'self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD, self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD, self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD, self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD, self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD, self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD, self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD, self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD, self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD, self.FER, self.FEN, self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS')
        self.var_names = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]
        if self.DT_OUTPUT_FILE is not None and self.ST_OUTPUT_FILE is not None:
            self.file_name = self.DT_OUTPUT_FILE

            # self.run_command(plot=True)
            self.sensor_data_location = self.file_name
            self.sensor_based()

            dt_AngV_Mean = self.AngV_Mean
            self.results_array_DT = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                     self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                     self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                     self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                     self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                     self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                     self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                     self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                     self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                     self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                     self.FER, self.FEN,
                                     self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                     ]
            self.task_type = 'DT'
            self.var_names_DT = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]

            self.file_name = self.ST_OUTPUT_FILE
            self.sensor_data_location = self.file_name
            # self.run_command(plot=True)
            self.sensor_based()

            st_AngV_Mean = self.AngV_Mean
            self.results_array_ST = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                     self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                     self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                     self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                     self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                     self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                     self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                     self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                     self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                     self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                     self.FER, self.FEN,
                                     self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                     ]
            self.task_type = 'ST'
            self.var_names_ST = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]
            self.var_names = self.var_names_ST + self.var_names_DT + ['DT_Cost']

            self.DT_Cost = (1 - (dt_AngV_Mean / st_AngV_Mean)) * 100

            self.results_array = self.results_array_ST + self.results_array_DT + [self.DT_Cost]
            # print(dt_AngV_Mean)
            # print(st_AngV_Mean)
            # print(self.DT_Cost)
        else:
            # self.run_command(plot=True)
            self.file_name = self.sensor_data_location
            self.sensor_based()

            self.results_array = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                  self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                  self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                  self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                  self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                  self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                  self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                  self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                  self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                  self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                  self.FER, self.FEN,
                                  self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                  ]
        print(self.results_array)

    def time_calibration(self):
        # Time Calibration
        self.cali_list = np.arange(len(self.signal)) * (1 / self.fps)

    def smooth_func(self, tresh=0.1):
        N = 5  # Filter order
        Wn = float(tresh)  # Cutoff frequency
        B, A = signal.butter(N, Wn, 'low')
        # print(B)
        # print(A)
        # print(self.signal)
        self.signal_after_butterworth = signal.filtfilt(B, A, self.signal)
        # self.signal = self.signal_after_butterworth

    def plot_signal(self):
        pass
        # plt.figure(figsize=(9, 6))
        # # plt.ylim([-500, 400])
        # # plt.xlim([0, 1600])
        # plt.title(self.patient_title)
        # # plt.title('Signal analysis')
        # plt.xlabel('Time')
        # plt.ylabel('Angle (Degrees$^\circ$)')
        # plt.plot(self.cali_list, self.signal, label='original signal')
        # # self.smooth_func()
        # # plt.plot(self.cali_list, self.smooth_signal, '--', label='smoothed')
        #
        # ''' Max peaks '''
        # plt.plot(self.cali_list[self.max_peaks[self.top_peaks_idx_max]],
        #          self.signal[self.max_peaks[self.top_peaks_idx_max]], 'gv',
        #          label='Max peaks')
        # plt.scatter(self.cali_list[self.max_peaks[self.top_peaks_idx_max]],
        #             self.signal[self.max_peaks[self.top_peaks_idx_max]], s=88,
        #             facecolors='none', edgecolors='r')
        #
        # ''' Min peaks '''
        # plt.plot(self.cali_list[self.min_peaks[self.top_peaks_idx_min]],
        #          self.signal[self.min_peaks[self.top_peaks_idx_min]], 'b^',
        #          label='Min peaks')
        # plt.scatter(self.cali_list[self.min_peaks[self.top_peaks_idx_min]],
        #             self.signal[self.min_peaks[self.top_peaks_idx_min]], s=88,
        #             facecolors='none', edgecolors='r')
        #
        # ''' Inflection points '''
        # # plt.plot(self.cali_list[self.infls], self.signal[self.infls], 'o')
        #
        # ''' Signal first derivative (Angular Speed) '''
        # # plt.plot(self.cali_list, self.signal_filtered_derivative + 120, label='Angular Velocity')
        # # plt.plot(self.cali_list, self.signal_second_derivative + 120, label='Angular Acceleration')
        #
        # # ''' Angular velocity '''
        # # plt.plot(self.cali_list[:len(self.cali_list) - 1], self.ang_v, '--')
        #
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        ''' Power figure '''
        # plt.figure()
        # plt.plot(self.cali_list, self.POWER, label='Angular Velocity')
        # plt.plot(self.cali_list[self.power_max_peaks], self.POWER[self.power_max_peaks], 'g*', label='Max peaks')
        # plt.plot(self.cali_list[self.power_min_peaks], self.POWER[self.power_min_peaks], 'b*', label='Min peaks')
        # # plt.plot(self.cali_list, self.signal_second_derivative, label='Angular Acceleration')
        # plt.title(self.patient_title)
        # plt.ylabel('Power')
        # plt.xlabel('Time')
        # plt.legend()
        # plt.show()

    def run_command(self, plot=False):
        self.get_signal()
        self.get_fps()
        self.smooth_func(tresh=0.1)

        # signal cleaner
        self.critical_points()
        self.peaks_cleaner()
        self.peaks_remove_outliers()
        self.detect_start_point()
        self.peaks_cleaner()
        self.detect_inflcs()

        ''' Calculate phenotypes '''
        self.phenotypes()

        # Plot
        if plot:
            self.plot_signal()

    def video_sensor_like(self):
        self.get_signal()
        self.get_fps()
        # print('done')
        self.smooth_func(tresh=0.105)
        self.signal_filtered_derivative = np.diff(self.signal_after_butterworth) / np.diff(self.cali_list)
        self.signal_filtered_derivative = np.append(self.signal_filtered_derivative,
                                                    self.signal_filtered_derivative[-1])

        self.signal = self.signal_filtered_derivative
        ''' Max peaks '''
        self.max_peaks, sth = find_peaks(self.signal, width=self.WIDTH_NUM)
        output = peak_prominences(self.signal, self.max_peaks)
        self.prominences = output[0]
        self.prominences_idx = output[1]
        self.top_peaks_idx_max = np.where(self.prominences > 60)[0]
        self.prominences = self.prominences[self.top_peaks_idx_max]
        self.prominences_idx = self.prominences_idx[self.top_peaks_idx_max]

        ''' Remove max lower than 100 '''
        updated_list_idx = []
        for i in range(len(self.top_peaks_idx_max)):
            if self.signal[self.max_peaks[self.top_peaks_idx_max[i]]] <= 100:
                updated_list_idx.append(i)
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, updated_list_idx)

        ''' Min peaks '''
        reversed_signal = self.reverse_signal()
        self.min_peaks, sth = find_peaks(reversed_signal, width=self.WIDTH_NUM)
        output = peak_prominences(reversed_signal, self.min_peaks)
        self.prominences = output[0]
        self.prominences_idx = output[1]
        self.top_peaks_idx_min = np.where(self.prominences > 60)[0]
        self.prominences = self.prominences[self.top_peaks_idx_min]
        self.prominences_idx = self.prominences_idx[self.top_peaks_idx_min]
        self.infls = None

        ''' Remove min higher than -100 '''
        updated_list_idx = []
        for i in range(len(self.top_peaks_idx_min)):
            if self.signal[self.min_peaks[self.top_peaks_idx_min[i]]] > -100:
                updated_list_idx.append(i)
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, updated_list_idx)

        ''' Remove redundant minimum points between two Max points '''
        redundant_min_idx = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            try:
                first_val = self.max_peaks[self.top_peaks_idx_max[i]]
                second_val = self.max_peaks[self.top_peaks_idx_max[i + 1]]
                boolean_array = np.logical_and(self.min_peaks[self.top_peaks_idx_min] >= first_val,
                                               self.min_peaks[self.top_peaks_idx_min] <= second_val)

                in_range_indices = np.where(boolean_array)[0]

                if len(in_range_indices) > 1:
                    global_min = np.argmin(self.signal[self.min_peaks[self.top_peaks_idx_min[in_range_indices]]])
                    for j in range(len(in_range_indices)):
                        redundant_min_idx.append(in_range_indices[j])
                    index = np.argwhere(redundant_min_idx == in_range_indices[global_min])
                    redundant_min_idx = list(np.delete(redundant_min_idx, index))
                # if len(in_range_indices) < 1:
                #     if first_val > second_val:

            except Exception as e:
                print(str(e))
        self.top_peaks_idx_min = np.delete(self.top_peaks_idx_min, redundant_min_idx)

        ''' Remove redundant Maximum points between two min points '''
        redundant_max_idx = []
        for i in range(len(self.top_peaks_idx_min) - 1):
            try:
                first_val = self.min_peaks[self.top_peaks_idx_min[i]]
                second_val = self.min_peaks[self.top_peaks_idx_min[i + 1]]
                boolean_array = np.logical_and(self.max_peaks[self.top_peaks_idx_max] >= first_val,
                                               self.max_peaks[self.top_peaks_idx_max] <= second_val)
                in_range_indices = np.where(boolean_array)[0]
                if len(in_range_indices) > 1:
                    global_max = np.argmax(self.signal[self.max_peaks[self.top_peaks_idx_max[in_range_indices]]])

                    for j in range(len(in_range_indices)):
                        redundant_max_idx.append(in_range_indices[j])
                    index = np.argwhere(redundant_max_idx == in_range_indices[global_max])
                    redundant_max_idx = list(np.delete(redundant_max_idx, index))
            except Exception as e:
                print(str(e))
        self.top_peaks_idx_max = np.delete(self.top_peaks_idx_max, redundant_max_idx)

        ''' Remove redundant Max and Min points of beginning and ending of the signal '''
        redundant_min_idx = []
        for i in range(len(self.top_peaks_idx_min) - 1):
            first_val = self.min_peaks[self.top_peaks_idx_min[i]]
            second_val = self.min_peaks[self.top_peaks_idx_min[i + 1]]
            boolean_array = np.logical_and(self.max_peaks[self.top_peaks_idx_max] >= first_val,
                                           self.max_peaks[self.top_peaks_idx_max] <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            if len(in_range_indices) < 1:
                redundant_min_idx.append(self.top_peaks_idx_min[i])
            else:
                self.top_peaks_idx_min = np.setdiff1d(self.top_peaks_idx_min, redundant_min_idx)
                break
        redundant_max_idx = []
        for i in range(len(self.top_peaks_idx_max) - 1):
            first_val = self.max_peaks[self.top_peaks_idx_max[i]]
            second_val = self.max_peaks[self.top_peaks_idx_max[i + 1]]
            boolean_array = np.logical_and(self.min_peaks[self.top_peaks_idx_min] >= first_val,
                                           self.min_peaks[self.top_peaks_idx_min] <= second_val)
            in_range_indices = np.where(boolean_array)[0]
            if len(in_range_indices) < 1:
                redundant_max_idx.append(self.top_peaks_idx_max[i])
            else:
                self.top_peaks_idx_max = np.setdiff1d(self.top_peaks_idx_max, redundant_max_idx)
                break

        ''' Detect zero crossing points '''
        all_critical_points = np.concatenate(
            (self.max_peaks[self.top_peaks_idx_max], self.min_peaks[self.top_peaks_idx_min]))
        # print(all_critical_points)
        all_critical_points = np.sort(all_critical_points)
        all_inf_t_vals = []
        for i in range(len(all_critical_points) - 1):
            arr = self.signal[all_critical_points[i]:all_critical_points[i + 1]]
            arr_time = self.cali_list[all_critical_points[i]:all_critical_points[i + 1]]
            # print(arr)
            # print(arr_time)
            idx1 = np.where(arr < 0, arr, -np.inf).argmax()
            idx2 = np.where(arr > 0, arr, np.inf).argmin()
            if idx1 == idx2:
                x1 = arr_time[idx1]
                all_inf_t_vals.append(x1)
            else:
                x1 = arr_time[idx1]
                x2 = arr_time[idx2]
                y1 = arr[idx1]
                y2 = arr[idx2]
                slope = ((y2 - y1) / (x2 - x1))
                # y - y1 = slope * (x - x1)
                x = ((0 - y1) / slope) + x1
                x = np.round(x, decimals=4)
                all_inf_t_vals.append(x)

        ''' Fix min-max indices '''
        for index, item in enumerate(all_inf_t_vals):
            indices = np.searchsorted(self.cali_list, item)
            self.cali_list = np.insert(self.cali_list, indices, item)
            self.signal = np.insert(self.signal, indices, 0)
            # if self.signal[indices + 1] > 0:
            # print('*' * 100)
            # print(self.min_peaks[[self.top_peaks_idx_min]])
            # print(self.min_peaks[[self.top_peaks_idx_min]][index + 1:])
            if self.max_peaks[self.top_peaks_idx_max[0]] < self.min_peaks[self.top_peaks_idx_min[0]]:
                self.min_peaks[[self.top_peaks_idx_min[index:]]] += 1
                self.min_peaks[[self.top_peaks_idx_min[index + 1:]]] += 1
                self.max_peaks[[self.top_peaks_idx_max[index + 1:]]] += 2
            else:
                self.max_peaks[[self.top_peaks_idx_max[index:]]] += 1
                self.max_peaks[[self.top_peaks_idx_max[index + 1:]]] += 1
                self.min_peaks[[self.top_peaks_idx_min[index + 1:]]] += 2

        all_idx = []
        for index, item in enumerate(all_inf_t_vals):
            indices = np.where(item == self.cali_list)[0][0]
            all_idx.append(indices)
        all_idx = np.array(all_idx)
        self.infls = np.copy(all_idx)

        ''' Phenotypes '''
        self.start_end_signal_peak()

        ''' Angular Velocity Range '''
        if self.signal_end_peak is 'min':
            end_adjuster = 0
        else:
            end_adjuster = 1

        flx_list = []
        for i in range(len(self.top_peaks_idx_max) - end_adjuster):
            flexion = self.signal[self.max_peaks[self.top_peaks_idx_max[i + end_adjuster]]] - self.signal[
                self.min_peaks[self.top_peaks_idx_min[i]]]
            flx_list.append(flexion)
        self.AngV_Mean = np.mean(flx_list)
        self.AngV_SD = np.std(flx_list)
        self.AngV_CV = variation(flx_list, axis=0)
        self.AngV_PD = self.pd_calculation(flx_list)
        # print(self.AngV_Mean)
        # print(self.AngV_SD)
        # print(self.AngV_CV)
        # print(self.AngV_PD)
        # print('*' * 100)

        ''' Flexion '''
        # print(self.cali_list[self.infls])
        a = 2
        b = 3
        flxs = []
        for i in range(len(self.infls) - 1):
            a = i
            b = i + 1
            angles = np.trapz(self.signal[self.infls[a]:self.infls[b]], x=self.cali_list[self.infls[a]:self.infls[b]])
            flxs.append(np.abs(angles))
        # print('Flexion: ', np.mean(flxs))
        # print(flxs)

        self.FLEXION_PHENOTYPE = np.mean(flxs)

        self.AngR_Mean = self.FLEXION_PHENOTYPE
        self.AngR_SD = np.std(flxs)
        self.AngR_CV = variation(flxs, axis=0)
        self.AngR_PD = self.pd_calculation(flxs)
        # print(self.AngR_Mean)
        # print(self.AngR_SD)
        # print(self.AngR_CV)
        # print(self.AngR_PD)

        ''' Frailty Index '''
        # Ph1
        ph1_rigidity = self.FLEXION_PHENOTYPE
        print('Rigidity: (self.AngR_Mean)', ph1_rigidity)

        ''' Power '''
        self.smooth_func(tresh=0.19)
        self.signal_filtered_derivative = np.diff(self.signal_after_butterworth) / np.diff(self.cali_list)
        self.signal_filtered_derivative = np.append(self.signal_filtered_derivative,
                                                    self.signal_filtered_derivative[-1])

        self.POWER = self.signal * self.signal_filtered_derivative

        ''' Update Power Calculations '''
        self.POWER = self.POWER / 25

        # print(self.POWER)

        # Ph2
        # print('+' * 100)
        mid_idx = len(self.signal[self.max_peaks[self.top_peaks_idx_max]]) // 2
        mid_idx = self.max_peaks[self.top_peaks_idx_max[mid_idx]]
        self.power_max_peaks, sth = find_peaks(self.POWER, width=self.WIDTH_NUM)
        self.power_min_peaks, sth = find_peaks(-self.POWER, width=self.WIDTH_NUM)
        # print(self.POWER)

        power_first_max_peaks, power_second_max_peaks = [self.power_max_peaks[self.power_max_peaks < mid_idx],
                                                         self.power_max_peaks[self.power_max_peaks >= mid_idx]]
        power_first_min_peaks, power_second_min_peaks = [self.power_min_peaks[self.power_min_peaks < mid_idx],
                                                         self.power_min_peaks[self.power_min_peaks >= mid_idx]]

        # print('+' * 100)
        # print(self.POWER)
        # print(power_first_max_peaks)
        # print(power_second_max_peaks)

        ''' Remove first outliers '''
        # print(self.min_peaks[self.top_peaks_idx_min])
        # print(power_first_max_peaks)
        idx = (np.abs(power_first_max_peaks - self.min_peaks[self.top_peaks_idx_min][1])).argmin()
        # print(power_first_max_peaks[idx])
        power_first_max_peaks = power_first_max_peaks[idx:]
        power_first_min_peaks = power_first_min_peaks[idx:]
        # print(power_first_max_peaks)

        k = np.min([len(power_first_max_peaks),
                    len(power_second_max_peaks),
                    len(power_first_min_peaks),
                    len(power_second_min_peaks)])
        # print(k)
        # k = len(power_first_max_peaks)
        top_k_max_first_peaks = np.argpartition(self.POWER[power_first_max_peaks], -k)[-k:]
        # print(top_k_max_first_peaks)
        # k = len(power_second_max_peaks)
        top_k_max_second_peaks = np.argpartition(self.POWER[power_second_max_peaks], -k)[-k:]

        # k = len(power_first_min_peaks)
        top_k_min_first_peaks = np.argpartition(-self.POWER[power_first_min_peaks], -k)[-k:]
        # k = len(power_second_min_peaks)
        top_k_min_second_peaks = np.argpartition(-self.POWER[power_second_min_peaks], -k)[-k:]

        power_range_first_section = np.mean(self.POWER[power_first_max_peaks[top_k_max_first_peaks]]) - np.mean(
            self.POWER[power_first_min_peaks[top_k_min_first_peaks]])
        power_range_second_section = np.mean(self.POWER[power_second_max_peaks[top_k_max_second_peaks]]) - np.mean(
            self.POWER[power_second_min_peaks[top_k_min_second_peaks]])
        # print(power_range_first_section, power_range_second_section)
        self.POWER_DECLINE_PERCENTAGE = np.round((1 - (power_range_second_section / power_range_first_section)) * 1,
                                                 decimals=3)
        ph2_exhaustion = self.POWER_DECLINE_PERCENTAGE

        print('Exhaustion: (self.PowR_PD)', ph2_exhaustion)

        self.PowR_Mean = np.mean([power_range_first_section, power_range_second_section])
        # print('self.PowR_Mean: ', self.PowR_Mean)
        power_ranges = np.concatenate((self.POWER[power_first_max_peaks[top_k_max_first_peaks]] -
                                       self.POWER[power_first_min_peaks[top_k_min_first_peaks]],
                                       self.POWER[power_second_max_peaks[top_k_max_second_peaks]] -
                                       self.POWER[power_second_min_peaks[top_k_min_second_peaks]]), axis=0)
        self.PowR_SD = np.std(power_ranges)
        self.PowR_CV = variation(power_ranges)
        self.PowR_PD = self.pd_calculation(power_ranges)
        # print('PowR_SD: ', self.PowR_SD)
        # print('*' * 100)
        # print(self.PowR_Mean)
        # print(self.PowR_SD)
        # print(self.PowR_CV)
        # print(self.PowR_PD)

        ''' Rising time '''
        # self.max_peaks[self.top_peaks_idx_max], self.min_peaks[self.top_peaks_idx_min]
        # all_critical_points
        max_first = False
        if self.max_peaks[self.top_peaks_idx_max][0] < self.min_peaks[self.top_peaks_idx_min][0]:
            max_first = True
        # print(max_first)

        rise_time_list = []
        if max_first:
            for i in range(len(self.max_peaks[self.top_peaks_idx_max]) - 1):
                value = self.cali_list[self.max_peaks[self.top_peaks_idx_max][i + 1]] - self.cali_list[
                    self.infls[2 * i + 1]]
                rise_time_list.append(value)
        else:
            for i in range(len(self.max_peaks[self.top_peaks_idx_max]) - 1):
                value = self.cali_list[self.max_peaks[self.top_peaks_idx_max][i]] - self.cali_list[self.infls[2 * i]]
                rise_time_list.append(value)

        self.RiseT_Mean = np.mean(rise_time_list)
        self.RiseT_SD = np.std(rise_time_list)
        self.RiseT_CV = variation(rise_time_list)
        self.RiseT_PD = self.pd_calculation(rise_time_list)
        # print('*' * 50)
        # print('RiseT_Mean: ', self.RiseT_Mean)
        # print(self.RiseT_SD)
        # print(self.RiseT_CV)
        # print(self.RiseT_PD)

        ''' Falling time '''
        ftv_list = []
        if not max_first:
            for i in range(len(self.min_peaks[self.top_peaks_idx_min]) - 1):
                # print(i)
                value = self.cali_list[self.min_peaks[self.top_peaks_idx_min][i + 1]] - self.cali_list[
                    self.infls[2 * i + 1]]
                ftv_list.append(value)
        else:
            for i in range(len(self.min_peaks[self.top_peaks_idx_min]) - 1):
                value = self.cali_list[self.min_peaks[self.top_peaks_idx_min][i]] - self.cali_list[self.infls[2 * i]]
                ftv_list.append(value)
        # print(ftv_list)

        self.FallT_Mean = np.mean(ftv_list)
        self.FallT_SD = np.std(ftv_list)
        self.FallT_CV = variation(ftv_list)
        self.FallT_PD = self.pd_calculation(ftv_list)

        ''' Rising + Falling time'''
        if len(ftv_list) > len(rise_time_list):
            counter = len(rise_time_list)
        else:
            counter = len(ftv_list)
        rft_list = []
        for i in range(counter):
            rft_list.append(rise_time_list[i] + ftv_list[i])
        rft_list = np.array(rft_list)

        self.RISING_TIME_PLUS_FALLING_TIME = self.RiseT_Mean + self.FallT_Mean

        self.RFT_Mean = self.RISING_TIME_PLUS_FALLING_TIME
        self.RFT_SD = np.std(rft_list)
        self.RFT_CV = variation(rft_list)
        self.RFT_PD = self.pd_calculation(rft_list)

        ''' Flexion time '''
        flex_list = []
        exten_list = []
        swing_flag = None
        if max_first:
            swing_flag = False
        else:
            swing_flag = True
        for i in range(len(self.infls) - 1):
            value = self.cali_list[self.infls[i + 1]] - self.cali_list[self.infls[i]]
            # print(value)
            if swing_flag:
                flex_list.append(value)
                swing_flag = False
            else:
                exten_list.append(value)
                swing_flag = True
        # print(flex_list)
        # print(exten_list)
        self.FlexT_Mean = np.mean(flex_list)
        self.FlexT_SD = np.std(flex_list)
        self.FlexT_CV = variation(flex_list)
        self.FlexT_PD = self.pd_calculation(flex_list)

        self.ExT_Mean = np.mean(exten_list)
        self.ExT_SD = np.std(exten_list)
        self.ExT_CV = variation(exten_list)
        self.ExT_PD = self.pd_calculation(exten_list)

        if len(exten_list) > len(flex_list):
            counter = len(flex_list)
        else:
            counter = len(exten_list)
        fet_list = []
        for i in range(counter):
            fet_list.append(flex_list[i] + exten_list[i])
        fet_list = np.array(fet_list)

        self.FET_Mean = self.FlexT_Mean + self.ExT_Mean
        self.FET_SD = np.std(fet_list)
        self.FET_CV = variation(fet_list)
        self.FET_PD = self.pd_calculation(fet_list)

        # print(self.FET_Mean)
        # print(self.FET_SD)
        # print(self.FET_CV)
        # print(self.FET_PD)

        ''' Flexion/Extension Rate '''
        duration = self.cali_list[self.max_peaks[self.top_peaks_idx_max[-1]]] - self.cali_list[
            self.max_peaks[self.top_peaks_idx_max[0]]]
        self.N_FLEXION_EXTENSION = len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]])
        # print('self.N_FLEXION_EXTENSION', self.N_FLEXION_EXTENSION)
        self.N_FLEXION_EXTENSION = int(
            np.round(len(self.cali_list[self.max_peaks[self.top_peaks_idx_max]]) * (20.0 / duration)))
        # print(duration)
        # print((20.0 / duration))
        # print('self.N_FLEXION_EXTENSION', self.N_FLEXION_EXTENSION)
        self.FLEXION_EXTENSION_RATE = (self.N_FLEXION_EXTENSION / duration) * 60

        self.FER = self.FLEXION_EXTENSION_RATE
        self.FEN = self.N_FLEXION_EXTENSION

        # New parameters
        # self.cognitive_parameters()
        self.CROR = 0
        self.CRLRSR = 0
        self.CRLS = 0
        self.CPOR = 0
        self.CPLRSR = 0
        self.CPLS = 0

        ''' Frailty Index calc'''
        ph3_slowness = self.FlexT_Mean * 1000
        print('Slowness: (self.FlexT_Mean)', ph3_slowness)

        ph4_steadiness_lack_flexion = self.FlexT_CV
        print('Steadiness_lack_flexion: (self.FlexT_CV)', ph4_steadiness_lack_flexion)

        ph5_steadiness_lack_extension = self.ExT_CV
        print('Steadiness_lack_extension: (self.ExT_CV)', ph5_steadiness_lack_extension)

        b = 0.24495
        a1 = -1.7357 * 0.001
        a2 = -1.2026 * 0.001
        a3 = 0.36848 * 0.001
        a4 = -0.49396
        a5 = 0.48974
        self.FRAILTY_INDEX = b + a1 * ph1_rigidity + a2 * ph2_exhaustion + a3 * ph3_slowness + \
                             a4 * ph4_steadiness_lack_flexion + a5 * ph5_steadiness_lack_extension
        print('FRAILTY_INDEX: ', self.FRAILTY_INDEX)
        self.FI = self.FRAILTY_INDEX
        # # Cognitive parameter - Range of Motion (Flexibility) Outliers Rate
        # self.CROR = self.COGNITIVE_ROM_DECLINE_RATE_1
        # # Cognitive parameter - Range of Motion (Flexibility) Linear Regression Residuals Sum Per Iteration
        # self.CRLRSR = self.COGNITIVE_ROM_LR_RESIDUAL_SUM
        # # Cognitive parameter - Range of Motion (Flexibility) Linear Regression Slope
        # self.CRLS = self.COGNITIVE_ROM_LR_SLOPE
        #
        # # Cognitive parameter - Pauses Outliers Rate
        # self.CPOR = self.COGNITIVE_IRREGULAR_PAUSES_RATE_1
        # # Cognitive parameter - Pauses Linear Regression Residuals Sum Per Iteration
        # self.CPLRSR = self.COGNITIVE_IRREGULAR_PAUSES_LR_RESIDUAL_SUM
        # # Cognitive parameter - Pauses Linear Regression Slope
        # self.CPLS = self.COGNITIVE_IRREGULAR_PAUSES_LR_SLOPE

        ''' Plot '''
        # # x_axis = np.arange(len(df)) * (1 / 25)
        # # df_t = pd.DataFrame()
        # # df_t['Time'] = x_axis
        # # df_t['Sensor_Signal'] = -df
        # plt.title(self.patient_title)
        # plt.xlabel('Time')
        # plt.ylabel('Angle (Degrees$^\circ$ / Sec)')
        # # plt.plot(x_axis, -df, '--', label='Sensor signal')
        #
        # # plt.plot(x_axis, -df, label='Sensor signal')
        #
        # plt.plot(self.cali_list, self.signal, label='Sensor signal')
        # plt.legend()
        # plt.show()
        self.plot_signal()

    def dt_cost_calculation_video_sensor(self):
        names_array = str(
            'self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD, self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD, self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD, self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD, self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD, self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD, self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD, self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD, self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD, self.FER, self.FEN, self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS')
        self.var_names = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]
        if self.DT_OUTPUT_FILE is not None and self.ST_OUTPUT_FILE is not None:
            self.file_name = self.DT_OUTPUT_FILE

            # self.run_command(plot=True)
            self.sensor_data_location = self.file_name
            self.video_sensor_like()

            dt_AngV_Mean = self.AngV_Mean
            self.results_array_DT = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                     self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                     self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                     self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                     self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                     self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                     self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                     self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                     self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                     self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                     self.FER, self.FEN,
                                     self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                     ]
            self.task_type = 'DT'
            self.var_names_DT = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]

            self.file_name = self.ST_OUTPUT_FILE
            # self.sensor_data_location = self.file_name
            # self.run_command(plot=True)
            self.video_sensor_like()

            st_AngV_Mean = self.AngV_Mean
            self.results_array_ST = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                     self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                     self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                     self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                     self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                     self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                     self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                     self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                     self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                     self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                     self.FER, self.FEN,
                                     self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                     ]
            self.task_type = 'ST'
            self.var_names_ST = ['{}_'.format(self.task_type) + i.split('.')[1] for i in names_array.split(',')]
            self.var_names = self.var_names_ST + self.var_names_DT + ['DT_Cost']

            self.DT_Cost = (1 - (dt_AngV_Mean / st_AngV_Mean)) * 100

            self.results_array = self.results_array_ST + self.results_array_DT + [self.DT_Cost]
            # print(dt_AngV_Mean)
            # print(st_AngV_Mean)
            # print(self.DT_Cost)
        else:
            # self.run_command(plot=True)
            # self.file_name = self.sensor_data_location
            self.video_sensor_like()

            self.results_array = [self.FI, self.AngR_Mean, self.PowR_PD, self.FlexT_Mean, self.FlexT_CV, self.ExT_CV,
                                  self.AngV_Mean, self.AngV_SD, self.AngV_CV, self.AngV_PD,
                                  self.AngR_Mean, self.AngR_SD, self.AngR_CV, self.AngR_PD,
                                  self.PowR_Mean, self.PowR_SD, self.PowR_CV, self.PowR_PD,
                                  self.RiseT_Mean, self.RiseT_SD, self.RiseT_CV, self.RiseT_PD,
                                  self.FallT_Mean, self.FallT_SD, self.FallT_CV, self.FallT_PD,
                                  self.RFT_Mean, self.RFT_SD, self.RFT_CV, self.RFT_PD,
                                  self.FlexT_Mean, self.FlexT_SD, self.FlexT_CV, self.FlexT_PD,
                                  self.ExT_Mean, self.ExT_SD, self.ExT_CV, self.ExT_PD,
                                  self.FET_Mean, self.FET_SD, self.FET_CV, self.FET_PD,
                                  self.FER, self.FEN,
                                  self.CROR, self.CRLRSR, self.CRLS, self.CPOR, self.CPLRSR, self.CPLS
                                  ]
        print(self.results_array)

    def run_and_save_results_video_sensor(self):

        self.video_sensor_like()

        self.dt_cost_calculation_video_sensor()

        ''' Save the results '''

        # if self.DT_Cost is not None:
        #
        # if self.DT_Cost is not None:
        #     results_array.append(self.DT_Cost)
        #     var_names.append('DT_Cost')
        # print(var_names)
        results_dict = {}
        results_list = []
        for i in range(len(self.var_names)):
            results_dict[self.var_names[i]] = self.results_array[i]
            results_list.append([self.var_names[i], self.results_array[i]])
        # print(results_dict)

        results_list = np.array(results_list).T
        # print(results_list)
        df = pd.DataFrame(results_list[1].reshape(1, -1), columns=list(results_list).pop(0))
        # df.index = self.patient_title
        # print(df)
        # print('Yes' * 100)
        # print(self.file_name)
        self.patient_title = self.file_name.split('\\')[-1].split('.')[0]

        if self.DT_OUTPUT_FILE is not None and self.ST_OUTPUT_FILE is not None:
            df.to_excel('{}//videosensor_results_all_{}.xlsx'.format(self.results_location, self.patient_title))
        else:
            df.to_excel('{}//viodeosensor_results_{}.xlsx'.format(self.results_location, self.patient_title))


def frailty_postprocess(excel_file, frailty_index_file, task_type, video_file_location, angular_acc):
    signal_class = Signal_Analysis_Mohammad()
    signal_class.file_name = excel_file
    signal_class.results_location = frailty_index_file
    signal_class.task_type = task_type
    signal_class.video_file_location = video_file_location
    signal_class.angular_acc_folder_path = angular_acc
    signal_class.run_and_save_results()

if __name__ == "__main__":
    sub_id = 'HML0493_SingleTask'
    ST_DT = 'ST'
    excel_file = rf"C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\Angle\{sub_id}.xlsx"
    frailty_index_file = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\frailty_index'
    video_file_location = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Data'
    angular_acc = r'C:\Users\mrouzi\Desktop\C2SHIP_AI\Results\angular_acc'

    frailty_postprocess(excel_file=excel_file,
                        frailty_index_file=frailty_index_file,
                        task_type=ST_DT,
                        video_file_location=video_file_location,
                        angular_acc=angular_acc)
