#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:12:57 2018

@author: jeremiasknoblauch

Description: Run robust Air pollution analysis
"""


import numpy as np
import scipy
import os
import csv
import matplotlib.pyplot as plt

from BVAR_NIG import BVARNIG
from BVAR_NIG_DPD import BVARNIGDPD
from detector import Detector
from Evaluation_tool import EvaluationTool
from cp_probability_model import CpModel



"""STILL NEES SETTING UP!"""



run_detectors = False    #whether we want to run detector or just read data
normalize = True        #normalize station-wise
daily_avg = True        #use daily averages (vs 2h-averages)
deseasonalize_2h = False    #only useful for 2h-averages. Will deseasonalize
                            #for each weekly 2h-interval ( = 12 * 7 intervals)
if daily_avg:
    deseasonalize_2h = False    #if daily_avg is True, 2h-deseasonalizing makes
                                #no sense
    deseasonalize_day = True #only one of the two deseasonalizations should be 
                        #chosen, and this one means that we only take weekday 
                        #averages
shortened, shortened_to = False, 500 #wheter to process only the first 
                                     #shortened_to observations and stop then

"""folder containing dates and data (with result folders being created at 
run-time if necessary)"""
baseline_working_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS//Code//SpatialBOCD//Data//AirPollutionData")
results_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS/Code//SpatialBOCD//PaperNIPS")

"""subset of the Airpollution data analyzed"""
cp_type = "CongestionChargeData" #only option available, originally wanted to
                                    #look at another time frame but didn't
                                    
"""Distance matrices computed using symmetrized road distances (i.e., taking
d(i,j) = 0.5*[d_road(i,j) + d_road(j,i)] and euclidean distances"""
dist_file_road = (baseline_working_directory + "//" + cp_type + "//" + 
                  "RoadDistanceMatrix_")
dist_file_euclid = (baseline_working_directory + "//" + cp_type + "//" + 
                   "EuclideanDistanceMatrix_")

"""File prototype for 2h-averaged station data from 08/17/2002 - 08/17/2003"""
prototype = "_081702-081703_2h.txt"

"""Decide if you want to take the bigger or smaller set of stations for the
analysis"""
mode = "bigger" #bigger, smaller (bigger contains more filled-in values)


"""These indices are used for reading in the station data of each station"""
if mode == "bigger":
    stationIDs = ["BT1", "BX1", "BX2", "CR2", "CR4", 
                  "EA1", "EA2", "EN1", "GR4", "GR5", 
                  "HG1", "HG2", "HI0", "HI1", "HR1", 
                  "HS2", "HV1", "HV3", "KC1", "KC2",
                  "LH2", "MY1", "RB3", "RB4", "TD0", 
                  "TH1", "TH2", "WA2", "WL1"]
elif mode == "smaller":
    stationIDs = ["BT1", "BX2", "CR2", "EA2", "EN1", "GR4",
                  "GR5", "HG1", "HG2", "HI0", "HR1", "HV1",
                  "HV3", "KC1", "LH2", "RB3", "TD0", "WA2"]
    
num_stations = len(stationIDs)

"""STEP 1: Read in distances"""

"""STEP 1.1: Read in road distances (as strings)"""
pw_distances_road = []
station_IDs = []
count = 0 
with open(dist_file_road + mode + ".csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        pw_distances_road += row


"""STEP 1.2: Read in euclidean distances (as strings)"""
pw_distances_euclid = []
station_IDs = []
count = 0 
with open(dist_file_euclid + mode + ".csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        pw_distances_euclid += row

"""STEP 1.3: Convert both distance lists to floats and matrices"""
pw_d_r, pw_d_e = [], []
for r,e in zip(pw_distances_road, pw_distances_euclid):
    pw_d_r.append(float(r))
    pw_d_e.append(float(e))
pw_distances_road = np.array(pw_d_r).reshape(num_stations, num_stations)
pw_distances_euclid = np.array(pw_d_e).reshape(num_stations, num_stations)


"""STEP 2: Convert distance matrices to nbhs. Cutoffs define the concentric
rings around the stations in the road-distance or euclidean space"""
cutoffs = [0.0, 10.0, 20.0, 30.0, 40.0, 100.0] 
num_nbhs = len(cutoffs) - 1

"""STEP 2.1: road distances"""
road_nbhs = []
for location in range(0, num_stations):
    location_nbh = []
    for i in range(0, num_nbhs):
        larger_than, smaller_than = cutoffs[i], cutoffs[i+1]
        indices = np.intersect1d( 
            np.where(pw_distances_road[location,:] > larger_than),
            np.where(pw_distances_road[location,:] < smaller_than)).tolist()
        location_nbh.append(indices.copy())
    road_nbhs.append(location_nbh.copy())
        
"""STEP 2.2: euclidean distances"""
euclid_nbhs =[]
for location in range(0, num_stations):
    location_nbh = []
    for i in range(0, num_nbhs):
        larger_than, smaller_than = cutoffs[i], cutoffs[i+1]
        indices = np.intersect1d( 
            np.where(pw_distances_euclid[location,:] > larger_than),
            np.where(pw_distances_euclid[location,:] < smaller_than)).tolist()
        location_nbh.append(indices.copy()) 
    euclid_nbhs.append(location_nbh.copy())


"""STEP 3: Read in station data for each station"""
station_data = []
for id_ in stationIDs:
    file_name = (baseline_working_directory + "//" + cp_type + "//" + 
                 id_ + prototype)
    
    """STEP 3.1: Read in raw data"""
    #NOTE: Skip the header 
    data_raw = []
    count = 0 
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if count > 0:
                data_raw += row
            count += 1    
    
    """STEP 3.2: Convert to floats"""
    #NOTE: We have row names, so skip every second
    dat = []
    for entry in data_raw:
        dat += [float(entry)]
    
    
    """STEP 3.3: Append to station_data list"""
    station_data.append(dat.copy())


"""STEP 4: Format the station data into a matrix"""
T, S1, S2 = len(station_data[0]), num_stations, 1
data = np.zeros((T, num_stations))
for i in range(0, num_stations):
    data[:,i] = np.array(station_data[i])
intercept_priors = np.mean(data,axis=0)
hyperpar_opt = "caron"




"""STEP 5: Transformation if necessary"""
if shortened:
    T = shortened_to
    data = data[:T,:]
    

if daily_avg:
    """average 12 consecutive values until all have been processed"""
    new_data = np.zeros((int(T/12), num_stations))
    for station in range(0, num_stations):
        new_data[:, station] = np.mean(data[:,station].
                reshape(int(T/12), 12),axis=1)
    data= new_data
    T = data.shape[0]


"""Deseasonalize based on week-day averages if we have 2h-frequency"""
if deseasonalize_day:
    if deseasonalize_2h:
        print("CAREFUL! You want to deseasonalize twice, so deseasonalizing " +
              "was aborted!")
    elif not daily_avg:
        mean_day = np.zeros((7, num_stations))
        #deseasonalize
        for station in range(0, num_stations):
            """get the daily average. Note that we have 12 obs/day for a year"""
            for day in range(0, 7):
                selection_week = [False]*day + [True]*12 + [False]*(6-day)
                selection = (selection_week * int(T/(7*12)) + 
                             selection_week[:(T-int(T/(7*12))*7*12)])
                mean_day[day, station] = np.mean(data[selection,station])
                data[selection,station] = (data[selection,station] - 
                    mean_day[day, station])

"""Deseasonalize based on week-day averages if we have 24h-frequency"""                
if deseasonalize_day and daily_avg:
    mean_day = np.zeros((7, num_stations))
    #deseasonalize
    for station in range(0, num_stations):
        """get the daily average. Note that we have 12 obs/day for a year"""
        #Also note that T will already have changed to the #days
        for day in range(0, 7):
            selection_week = [False]*day + [True] + [False]*(6-day)
            selection = (selection_week * int(T/7) + 
                         selection_week[:(T-int(T/7)*7)])
            mean_day[day, station] = np.mean(data[selection,station])
            data[selection,station] = (data[selection,station] - 
                mean_day[day, station])
              

"""Deseasonalize based on 2h averages if we have 24h-frequency"""                    
if deseasonalize_2h:
    if deseasonalize_day:
        print("CAREFUL! You want to deseasonalize twice, so deseasonalizing " +
              "was aborted!")
    else:
        mean_2h = np.zeros((12*7, num_stations))
        for station in range(0, num_stations):
            """get the average for each 2h-interval for each weekday"""
            for _2h in range(0, 12*7):
                selection_2h = [False]*_2h + [True] + [False]*(12*7-1-_2h)
                selection = (selection_2h * int(T/(7*12)) + 
                             selection_2h[:(T-int(T/(7*12))*7*12)])
                mean_2h[_2h, station] = np.mean(data[selection,station])
                data[selection,station] = (data[selection,station] - 
                    mean_2h[_2h, station])

"""normalize the data"""    
if normalize:
    data = (data - np.mean(data, axis=0))/np.sqrt(np.var(data,axis=0))
    intercept_priors = np.mean(data,axis=0)
    
"""Plot the data before you run analysis"""
num_row = int(num_stations/7) #num_stations
num_col = 7 #1
height_ratio = [1]*num_row
fig, ax_array = plt.subplots(num_row, num_col, sharex = True, sharey = True,
                                 figsize = (10,4), 
                             gridspec_kw = {'height_ratios':height_ratio})
for i in range(0, num_row):
    if num_row == num_stations:
        ax_array[i].plot(data[:,i])
    else:
        for j in range(0, num_col):
            ax_array[i,j].plot(data[:,(i*num_col) + j + 1])


    
    
"""STEP 6: Select the priors"""
res_seq_list = [
        [[0]],
        [[0]]*2,
        [[0,1]]*2,
        [[0]]*3,
        [[0,1]]*3,
        [[0]]*4,
        [[0,1]]*4,
        [[0]]*5,
        [[0,1]]*5
        ]


res_tight_week = [[0]]*1


VB_window_size = 300
full_opt_thinning = 25
full_opt_thinning_schedule = None #np.array([50,100,150,300])
first_full_opt = 5
SGD_batch_size = 10
anchor_batch_size_SCSG = 30
anchor_batch_size_SVRG = 20

a,b = 1,20 #1, 30
alpha_param = 0.005 #0.005
alpha_rld = 0.1 #0.025
intensity = 1000

prior_mean_scale = 0
prior_var_beta = shrinkage = 20 #5 or 20

rld = "power_divergence" #power_divergence kullback_leibler
rld_learning = True
param_learning = "individual" #"individual" #"individual" #"individual"
np.random.seed(999)



"""STEP 7: Intercept grouping"""
grouping = np.zeros((S1*S2, S1*S2))
for i in range(0, S1*S2):
    grouping[i,i]=1
grouping = grouping.reshape((S1*S2, S1,S2))

lags = [1] #[1,2,3]
road_model, euclid_model = True, True
model_universe = []

for lag in lags:
    
    res_tight_week = [[0]]*lag
    
   
    #creates AR model    
    model_universe = model_universe + [
                BVARNIGDPD(
                     prior_a=a, 
                     prior_b=b, #b, 
                     S1=S1, 
                     S2=S2, 
                     alpha_param = alpha_param,
                     prior_mean_beta=None, 
                     prior_var_beta=None,
                     prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                     prior_var_scale=shrinkage,
                     general_nbh_sequence=[[[]]]*S1*S2,
                     general_nbh_restriction_sequence = res_tight_week,
                     general_nbh_coupling = "weak coupling", 
                     hyperparameter_optimization = "online", #"online", #"online", #"online",
                     VB_window_size = VB_window_size,
                     full_opt_thinning = full_opt_thinning,
                     SGD_batch_size = SGD_batch_size,
                     anchor_batch_size_SCSG = anchor_batch_size_SCSG,
                     anchor_batch_size_SVRG = anchor_batch_size_SVRG,
                     first_full_opt = first_full_opt,
                     full_opt_thinning_schedule = full_opt_thinning_schedule,
                     intercept_grouping = grouping
                )
            ]
                
    #creates road nbh model 
    if road_model:           
        model_universe = model_universe +[
                    BVARNIGDPD(
                         prior_a=a, 
                         prior_b=b, #b, 
                         S1=S1, 
                         S2=S2, 
                         alpha_param = alpha_param,
                         prior_mean_beta=None, 
                         prior_var_beta=None,
                         prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                         prior_var_scale=shrinkage,
                         general_nbh_sequence=road_nbhs,
                         general_nbh_restriction_sequence = res_tight_week,
                         general_nbh_coupling = "weak coupling", 
                         hyperparameter_optimization = "online", #"online", #"online", #"online",
                         VB_window_size = VB_window_size,
                         full_opt_thinning = full_opt_thinning,
                         SGD_batch_size = SGD_batch_size,
                         anchor_batch_size_SCSG = anchor_batch_size_SCSG,
                         anchor_batch_size_SVRG = anchor_batch_size_SVRG,
                         first_full_opt = first_full_opt,
                         full_opt_thinning_schedule = full_opt_thinning_schedule,
                         intercept_grouping = grouping
                    )
                ]
    
    #creates euclid nbh model
    if euclid_model:
        model_universe = model_universe + [
                    BVARNIGDPD(
                         prior_a=a, 
                         prior_b=b, #b, 
                         S1=S1, 
                         S2=S2, 
                         alpha_param = alpha_param,
                         prior_mean_beta=None, 
                         prior_var_beta=None,
                         prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                         prior_var_scale=shrinkage,
                         general_nbh_sequence=euclid_nbhs,
                         general_nbh_restriction_sequence = res_tight_week,
                         general_nbh_coupling = "weak coupling", 
                         hyperparameter_optimization = "online", #"online", #"online", #"online",
                         VB_window_size = VB_window_size,
                         full_opt_thinning = full_opt_thinning,
                         SGD_batch_size = SGD_batch_size,
                         anchor_batch_size_SCSG = anchor_batch_size_SCSG,
                         anchor_batch_size_SVRG = anchor_batch_size_SVRG,
                         first_full_opt = first_full_opt,
                         full_opt_thinning_schedule = full_opt_thinning_schedule,
                         intercept_grouping = grouping
                    )
                ]     
        
model_universe = np.array(model_universe)           
model_prior = np.array([1/len(model_universe)]*len(model_universe))
cp_model = CpModel(intensity)

"""Build and run detector"""
detector = Detector(data=data, model_universe=model_universe, 
        model_prior = model_prior,
        cp_model = cp_model, S1 = S1, S2 = S2, T = T, 
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold = 75,
        notifications = 5,
        save_performance_indicators = True,
        training_period = 50,
        generalized_bayes_rld = rld, 
        alpha_param_learning =  param_learning,
        alpha_param  = alpha_param, 
        alpha_param_opt_t = 100, 
        alpha_rld = alpha_rld, 
        alpha_rld_learning = rld_learning, 
        loss_der_rld_learning="absolute_loss")
detector.run()

"""Store results + real CPs into EvaluationTool obj"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_run_detector(detector)

"""STEP 10: Plot the raw data + rld"""
height_ratio =[10,14]
custom_colors = ["blue", "purple"] 
fig, ax_array = plt.subplots(2, figsize=(8,5), sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio})
plt.subplots_adjust(hspace = .35, left = None, bottom = None,
                    right = None, top = None)
ylabel_coords = [-0.065, 0.5]

#Plot of raw Time Series
EvT.plot_raw_TS(data.reshape(T,S1*S2), indices = [0], xlab = None, 
        show_MAP_CPs = True, 
        time_range = np.linspace(1,T, T, dtype=int), 
        print_plt = False,
        ylab = "value", ax = ax_array[0],
        custom_colors_series = ["black"],
        custom_colors_CPs = ["blue", "blue"]* 100,
        custom_linestyles = ["solid"]*100,
        ylab_fontsize = 14,
        ylabel_coords = ylabel_coords)
                           
#Run length distribution plot
EvT.plot_run_length_distr(buffer=0, show_MAP_CPs = True, 
                                   mark_median = False, 
    mark_max = True, upper_limit = 1000, print_colorbar = True, 
    colorbar_location= 'bottom',log_format = True, aspect_ratio = 'auto', 
    C1=0,C2=700, 
    time_range = np.linspace(1,
                             T-2, 
                             T-2, dtype=int), 
    start = 1, stop = T, 
    all_dates = None, 
    custom_colors = ["blue", "blue"] * 30, 
    custom_linestyles = ["solid"]*30,
    custom_linewidth = 3,
    xlab_fontsize =14,
    ylab_fontsize = 14, 
    ax = ax_array[1], figure = fig,
    no_transform = True,
    date_instructions_formatter = None, 
    date_instructions_locator = None,
    ylabel_coords = ylabel_coords,
    xlab = "observation number",
    arrow_distance = 25)
    
EvT.plot_model_posterior(log_format = False, indices=[0,1,2])
    
"""STEP 11: Plot some performance metrics"""
print("CPs are ", detector.CPs[-2])
print("MSE is", np.mean(detector.MSE))
print("MAE is", np.mean(detector.MAE))
print("NLL is", np.mean(detector.negative_log_likelihood), 1.96*scipy.stats.sem(detector.negative_log_likelihood))
print("rld model is",detector.generalized_bayes_rld )
print("RLD alpha is", detector.alpha_rld)
print("NOTE: The results for the standard method are taken from" +
      "Knoblauch & Damoulas (2018) and can be reproduced here: " + 
      "https://github.com/alan-turing-institute/bocpdms/")







