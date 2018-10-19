#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:37:45 2018

@author: jeremiasknoblauch

Description: Generate AR process with CPs and outliers to test robust BOCPDMS
"""

import numpy as np
from detector import Detector
from BVAR_NIG import BVARNIG
from cp_probability_model import CpModel
import matplotlib.pyplot as plt
from Evaluation_tool import EvaluationTool


"""parameters for AR-process"""
T = 5000
S1 = S2 = 1
lag_list = [1,2,3]
CP_locations = [250, 750, 2000, 2500, 4000, 4250]
num_seg = len(CP_locations) + 1
coef_list = [np.array([2.5, 0.5, -0.1, 0.3]),
             np.array([-1, 0.25, -0.7, 0.0]),
             np.array([1, 0.75, 0, 0]),
             np.array([3, -0.9, 0.05, -0.05]),
             np.array([0, 0.6, -0.2, -0.1]),
             np.array([-2, 0.5, -0.1, 0.3]),
             np.array([0, -0.7, -0.1, 0.15])]
var_list = [2, 1.5, 2.5, 0.5, 1.2, 2, 0.7]
burn_in = 100

"""Generate AR (with CPs but without noise)"""

data = np.zeros(T + burn_in)
for seg_id in range(0, num_seg):
    
    
    """STEP 1: Get segment length"""
    if seg_id == 0:
        """need to add burn-in for the first segment"""
        seg_len = CP_locations[seg_id]-1 + burn_in
        data_indices = np.linspace(0, CP_locations[seg_id]-2, seg_len, 
                                       dtype=int)
    elif seg_id == num_seg-1:
        seg_len = T - CP_locations[seg_id-1]
        data_indices = np.linspace(CP_locations[seg_id-1]-1, T-1, seg_len, 
                                       dtype=int)
    else:
        seg_len = CP_locations[seg_id] - CP_locations[seg_id-1] - 1
        data_indices = np.linspace(CP_locations[seg_id-1]-2, 
                    CP_locations[seg_id]-1, seg_len, dtype=int)
    
    """STEP 2: Generate  data"""
    raw_seg = np.random.normal(0, var_list[seg_id], size = seg_len)
    lag_len = len(coef_list[seg_id]) - 1 #-1 for the constant term!
    
    if seg_id == 0:
        seg_len = seg_len - lag_len
    elif seg_id != 0:
        """if not first segment, use the data of the previous segment"""
        prev = data[(data_indices[0] - lag_len):(data_indices[0] - 1)]
        raw_seg = np.insert(raw_seg, 0, prev)
    
    for j in range(lag_len, seg_len):
        raw_seg[j] = (np.sum(raw_seg[(j-lag_len):j] * coef_list[seg_id][1:]) 
            + coef_list[seg_id][0] + raw_seg[j])
    
    """put into data"""
    if seg_id == 0:
        data[data_indices] = raw_seg
    else:
        data[data_indices] = raw_seg[lag_len-1:]

data = data[burn_in:]        

"""Inject Outliers"""
outlier_locations = [100, 400, 1000, 1500, 2700, 3500, 4300, 4800]
#added on top of the value it has at the moment
outlier_magnitude = [1000, -1000, 150, -150, 50, -50, 250, -250]
for i in range(0, len(outlier_locations)):
    data[outlier_locations[i]] = data[outlier_locations[i]] + outlier_magnitude[i]

"""normalize"""
data = (data - np.mean(data))/np.sqrt(np.var(data))

plt.plot(data)
plt.show()


cp_model = CpModel(200)
a,b = 1.5, 1.5
AR_models = []
for lag in lag_list:
    AR_models += [BVARNIG(
                    prior_a = a,prior_b = b,
                    S1 = S1,S2 = S2,
                    prior_mean_beta = np.zeros(lag + 1),
                    prior_var_beta = 0.0075*np.identity(1+lag),
                    #prior_mean_scale = prior_mean_scale,
                    #prior_var_scale = prior_var_scale,
                    intercept_grouping = None,
                    nbh_sequence = [0]*lag,
                    restriction_sequence = [0]*lag,
                    hyperparameter_optimization = "online")]

model_universe = np.array(AR_models)
model_prior = np.array([1/len(model_universe)]*len(model_universe))
        
"""run detectors, potentially plot stuff"""
"""Build and run detector"""
detector = Detector(data=data.reshape(T, S1, S2), 
        model_universe=model_universe, 
        model_prior = model_prior,
        cp_model = cp_model, 
        S1 = S1, S2 = S2, T = T, 
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold = 200,
        notifications = 100,
        save_performance_indicators = True,
        generalized_bayes = "power_divergence",
        alpha = 5, #0.0005, 0.25, 0.5, 1.0// 0.05, 0.1 upwards
        generalized_bayes_hyperparameter_learning = True)
detector.run()

"""Store results + real CPs into EvaluationTool obj"""
EvT = EvaluationTool()
#EvT.add_true_CPs(true_CP_location=true_CP_location, 
#                 true_CP_model_index=true_CP_location, 
#             true_CP_model_label = -1)
EvT.build_EvaluationTool_via_run_detector(detector)
        

minus = 3
EvT.plot_run_length_distr(buffer=0, show_MAP_CPs = True, 
                   mark_median = False, 
        mark_max = True, upper_limit = T-2-minus, print_colorbar = True, 
        colorbar_location= 'bottom',log_format = True, aspect_ratio = 'auto', 
        #C1=1,C2=0, 
        time_range = np.linspace(1,
                                 T-2-minus, 
                                 T-2-minus, dtype=int), 
        #start = 622 + 2, stop = 1284-minus, #start=start, stop = stop, 
        all_dates = None, #all_dates,
        event_time_list=outlier_locations + CP_locations,#datetime.date(715,1,1)], 
        label_list=["o"]*len(outlier_locations) + ["cp"]*len(CP_locations), 
        space_to_colorbar = 0.52,
        custom_colors = ["blue", "blue"]*100, #["blue"]*len(event_time_list),
        custom_linestyles = ["solid"]*3,
        custom_linewidth = 3,
        arrow_colors= ["black"]*len(outlier_locations) + ["orange"]*len(CP_locations),
        number_fontsize = 14,
        arrow_length = 135,
        arrow_thickness = 3.0,
        xlab_fontsize =14,
        ylab_fontsize = 14, 
        arrows_setleft_indices = [0],
        arrows_setleft_by = [50],
        zero_distance = 0.0,
        ax = None, figure = None,
        no_transform = True,
        date_instructions_formatter = None, #yearsFmt,
        date_instructions_locator = None,
        #ylabel_coords = ylabel_coords,
        xlab = "Year",
        arrow_distance = 25)
print("CPs are ", detector.CPs[-2])
print("MSE is", np.mean(detector.MSE))
print("NLL is", np.mean(detector.negative_log_likelihood))
plt.figure()
plt.plot(detector.alpha_list)
plt.show()






