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

if False:
    fig.savefig("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS/Code//SpatialBOCD" + "//Paper//Presentation//APNetwork//AllAPs.pdf")

    
    
"""STEP 6: Select the priors"""
#prior_mean_scale = 0.0
#intensity = 180
#var_scale_list = 0.0005
#a_prior_list =  100
#b_prior_list = 25
##AR_selections = [1,2,3,4,5]
#res_seq_list = [
#        [[0]],
#        [[0]]*2,
#        [[0,1]]*2,
#        [[0]]*3,
#        [[0,1]]*3,
#        [[0]]*4,
#        [[0,1]]*4,
#        [[0]]*5,
#        [[0,1]]*5
#        ]

#################
#REPORTED IN ICML SUBMISSION:
# a = 100, b = 25, bvp = 0.0005, intensity = 180, models: AR(1-5), 
# [[0]]*1-5, [[0,1]]*1-5, daily data, deseasonalized
#################



################
#NIPS: Use ARs##
###############

#Notice: For larger lag lengths, we need to jack up b and shrinkage for numerical reasons

#       a       b               int     ap      arld    shrink      lagl
#1:     5       5*pow(10,3)     100     0.2      -      0.25        6       FAILED inversion in precompute
#2:     5       pow(10,4)       100     0.2      -      0.25        6       
#3:     5       pow(10,5)       100     0.2      -      0.25        6       Ran, basically every obs a CP & param est at 0
#4:     5       pow(10,6)       100     0.2      -      0.25        6       Ran, basically every obs a CP & param estimates at 0
#L:     5       pow(10,4)       100     0.1      -      0.25        6       Ran, no CPs after 109 (warm-up)
#C:     5       pow(10,4)       100     0.3      -      0.25        6       FAILED inversion in precompute
#R:     5       pow(10,4)       100     0.5      -      0.25        6       Ran, after warm-up finds CPs at  192, 576, 908, 1245, 2917 
#GL:    5       pow(10,4)       100     0.2      -      0.1         6       FAILED (inv)
#GC:    5       pow(10,4)       100     0.2      -      0.05        6       lots of CPs at beginning, then none
#GR:    5       pow(10,4)       100     0.2      -      0.005       6       lots of CPs at beginning, then none
#YL:    1       pow(10,4)       100     0.05      -     0.1         6       FAILED  (inversion in precompute?)
#YC:    1       pow(10,4)       100     0.025      -    0.1         6       FAILED  (inversion in precompute?)
#YR:    1       pow(10,4)       100     0.01      -     0.1         6       FAILED  (inversion in precompute?)

#C:     1       pow(10,3)       100     0.2      -      0.25        3       FAILED inversion in precompute
#YL:    1       pow(10,3)       100     0.15     -      0.1         3       Ran, CPs 105, 327, 959, 2391
#YC:    1       pow(10,3)       100     0.125    -      0.1         3       FAILED inversion in precompute
#YR:    1       pow(10,3)       100     0.1      -      0.1         3       FAILED inversion in precompute
#AL:    1       pow(10,3)       100     0.25     -      0.25        1       Ran, finds CP at 2669
#AC:    1       pow(10,3)       100     0.2      -      0.25        1       Ran, finds CP at 1239
#AR:    1       pow(10,3)       100     0.15      -     0.25        1       Ran, finds CP at 1872
#JL:    1       pow(10,2)       100     0.25     -      0.25        1       Ran, finds no CP
#JC:    1       pow(10,2)       100     0.2      -      0.25        1       Ran, finds CP at 1703
#JR:    1       pow(10,2)       100     0.15      -     0.25        1       Ran, finds CPs at 1595, 2579, 3590 + very decent MSE/MAE values  

#conclusions:   - From JR vs YL: pow(10,2) enough, perhaps we should go even lower
#               - Maybe should also set ap much lower, i.e. <= 0.15, though JR was alright with 0.15!
#               - defo need correction of the XX_t inversion in bvarnigdpd
    
#YC:    5       pow(10,4)       100     0.02     -      0.1         1       
#YR:    5       pow(10,4)       100     0.01     -      0.1         1  
#JL:    1       pow(10,3)       100     0.02     -     1           1     [70, 0],[626, 0],[1293, 0],1735, 0][2647, 0][2752, 0][2950, 0][3885, 0][4276, 0], MSE = 0.9, MAE = 0.77
#JC:    1       pow(10,3)       100     0.02     -     10          1    [623, 0], [1210, 0], [2645, 0], [2947, 0], MSE = 0.9, MAE = 0.77
#AL:    1       pow(10,3)       100     0.01     -     0.25        1    CPs at [531, 0],[867, 0],[1286, 0],[1660, 0] [2647, 0],[2886, 0],[3271, 0],[4254, 0]] MSE = 0.89, MAE = 0.76 
#AC:    1       pow(10,3)       100     0.005    -     0.25        1    
#AR:    1       pow(10,3)       100     0.001    -     0.25        1    CPs at  [531, 0], [879, 0], [1291, 0],[1660, 0],[2188, 0],[2534, 0],[2884, 0],[3264, 0],[3883, 0],[4254, 0]]   MSE = 0.89, MAE = 0.76

#YL:    1       pow(10,2)       100     0.125      -     0.25        1  MSE = 0.933, MAE = 0.775, 2 CP clusters [1620], [3260]  
#GL:    1       pow(10,2)       100     0.1     -       0.25         1  MSE = 0.93, mAE = 0.78, 5 CPs at   [538, 0], [1238, 0], [1994, 0], [2534, 0], [3255, 0], [4381, 0] 
#L:     1       pow(10,2)       100     0.075     -       0.25       1 
#C:     1       pow(10,2)       100     0.05     -       0.25        1 too many CPs, MAE = 0.74, MSE = 0.85
#R:     1       pow(10,2)       100     0.025     -       0.25       1 too many CPs! But MAE = 0.738, MSE = 0.847
#1:     1       pow(10,2)       100     0.01     -       0.25        1 too many CPs! But MAE = 0.73, MSE = 0.84
#2:     3       5               100     0.2      -       0.25        1 MSE = 0.96, MAE = 0.79, 50% failed opt, CPs: 89, 0],[132, 0],[151, 0],[374, 0],[539, 0],[709, 0],[1389, 0],[1665, 0],[1865, 0],[1913, 0],[3054, 0],[3132, 0],[3274, 0],[3292, 0],[3330, 0],[3342, 0],[3384, 0],[3498, 0],[3706, 0],[4204, 0],[4267, 0],[4379, 0]
#DECENT:
#3:     3       25              100     0.2      -       0.25        1 CPs: [131, 0], [269, 0], [940, 0], [4201, 0]], decent MSE/MAE val but larger than for Jacks
#4:     1       pow(10,2)       100     0.01     -       0.25        1 MAE = 0.735, MSE = 0.84, but way too many CPs


#3:     2       25              100     0.05     -       1          1   nan MSE and MAE, way too many CPs.

#AL:    1       15              500     0.05     -       1          1
#DECENT:
#AC:    1       15              500     0.05     -       5          1    pretty good! Have two explainable CPs and MSE = 0.91, MAE = 0.77, CP cluster at 725, one at 1760, 2147 (!), 3430, cluster at 3480
#AR:    LEAVE EMPTY SINCE IT SEEMS IT CRASHES OFTEN WITH 3 PROCESSES RUNNING
#JL:    1       15              500     0.05     -      25          1    way way too many CPs, MSE = 0.98, MAE = 0.8
#JC:    1       15              500     0.075     -       1          1   crashed
#JR:    1       15              500     0.075     -       5          1   way too many CPs

#JL:    1       15              500     0.05     -       1            1   way too many CPs, MSE = 0.97, MAE = 0.81
#JC:    1       15              500     0.05     -       2.5          1   way too many CPS, MSE & MAE nan
#C:     1       15              500     0.05     -       1.5          1   way too many CPS, MSE & MAE nan
#DECENT:
#R:     1       15              500     0.075     -       2.5          1  too many CPs, but far less than for shr = 10 (13), MAE = 0.77, MSE = 0.91   
#AL:    1       15              2500    0.05     -        2.5         1   way too many CPs, MSE and MAE nan
#AC:    1       15              500     0.075    -        10          1   way too many CPs, MSE = 0.98, MAE = 0.81
#1:     1       15              500     0.025     -       0.5          1  way too many CPs, but MAE = 0.69, MSE = 0.7695
#2:     1       15              500     0.025     -       1           1
#4:     1       15              500     0.025     -       2.5          1

#JL:    1       15              500     0.025       -      0.25         1 MSE= 0.76, MAE = 0.69 but way too may CPs
#JC:    1       15              500     0.025       -      0.1          1 way too many CPs, but MAE = 0.7, MSE = 0.776
#JR:    1       15              500     0.025       -      0.01         1 MSE =0.82, MAE = 0.72, but way too many CPs (but defo less than the other two settings, which makes sense)
#AL:    1       15              500     0.05        -      0.25         1 MSE = 0.81, MAE = 0.70 but way too many CPs
#AC:    1       15              500     0.05        -      0.1         1 MSE = 0.82, MAE = 0.726
#AR:    NONE
#GL:    1       15              500     0.075       -      0.25         1  MSE = 0.845, MAE = 0.735, too many CPs
#GC:    1       15              500     0.075       -      0.1         1  12 CPs (not outragously too many), MSE = 0.84, MAE = 0.735
#GR:    1       15              500     0.075       -      0.01         1  56 CPs (too many), MSE = 0.78 , MAE =0.69
#L:     1       15              500     0.075       -       5           12  did not finish execution, but barely changed values (alpha too large)
#C:     1       15              500     0.05       -       2.5           12  did not finish execution, but barely changed values (alpha too large)

#L:     1       15              500     0.005       -       50           6  does not work, params don't move
#C:     1       15              500     0.001       -       100          6  seems to work, params do move


#R:     1       15              500     0.05        -      0.01         1   too many CPs, (20), but true CP defo in there. MSE = 0.86, MAE = 0.744

#2:     1       25              500     0.1         -       0.5         1   many failures during exec
#3:     1       25              500     0.1         -       0.25        1   decent, 10 CPs with MSE = 0.88, MAE = 0.756
#4:     1       25              500     0.1         -       0.1        1    seems to work really well, but too many CPs
#1:     NONE 

#DAILY AVG:
#JL:    1       25              100     0.05        -       1          1   seems about right, 2 CPs around the cong charge, though MSE > 1 still, and opts fail too often
#JC:    1       25              100     0.1         -       1            1 too robust, no cps and MSE > 1, MAE = 0.8
#JR:    1       25              100     0.15        -       1            1 too robust, one CP and MSE > 1, MAE = 0.8
#JC:    1       25              100     0.025        -       1          1 too robust, one CP close to target MSE > 1
#JR:    1       25              100     0.01        -       1          1  still a bit too robust, but EXCELLENT CP location, MSE = 0.98
#JL:    1       25              100     0.1        -       1    1 too robust, one CP close to target MSE >0
#JL:    1       25              100     0.01        -       10   JACKPOT, we get the CP where it should be, MSE > 0
#JC:    1       25              100     0.005        -       10  JACKPOT, we get the CP where it should be, MSE > 0
#JR:    1       25              100     0.001        -       10  we get the CP where it should be + some more, MSE < 0
#With model universe + lag = 7
#JL:    1       25              100     0.01        -       0.01          7+++  many failures in exec
#JC:    1       25              100     0.075        -       0.1          7+++
#JR:    1       25              100     0.075        -       0.5          7+++
#AL:    1       25              100     0.075               0.5             7+++ many failures in exec, nan MSE & MAE and one CP
#AC:    1       25              100     0.075               1             7+++ only CPs everywhere, MSE = 0.98, MAE = 0.82
#YL:    1       25              100     0.01                0.25            7+++   99%failed opts many failures in exec
#L:     1       15              100     0.1        -      10             7+++     no failed opts, params fairly close to 0, cluster of CPs at beginning, then none
#C:     1       15              100     0.1        -      2.5           7+++      no failed opts, params fairly close to 0, 3 CPs
#2:     1       15              100     0.05                1           7+++      2 CPs (114, 363), MSE = 0.98, MAE = 0.81, 50% failed opt

#2h AVG:
#AL:    1       15              500     0.1        -      5         1  robust, but too many CPs (params don't move)
#AC:    1       15              500     0.1        -      10        1  robust, but too many CPs (params don't move)
#L:     1       15              100     0.1        -      pow(10,3)         1  did not finish, but seemed alright
#C:     1       15              100     0.1        -      pow(10,4)         1
#YR:    1       100             500     0.075       -       0.25    1       36% failed opts, robust, finds 10 CPs and MAE = 0.76, MSE = 0.89

#GC:    1       15              500     0.075       -      0.5              3+++ opt takes way too long
#GR:    1       15              500     0.075       -      1                3+++ opt takes way too long


#Q: Have we tried combo of small b, small shrinkage? Seems like it worked well but we don't have a large sample!
#Q: Have we tried alpha-param 0.1-0.2 with small b? Yes, and it was pretty okay apparently!!! TRY MORE

#With 4, try 12-lagged using less expensive opt. (thres = 25, window = 300, opts at 10,25,75,150,300)
#: 1       15              500     0.05     -       5          12 TRY AGAIN LATER
#: 1       25              500     0.1         -       0.01        1


#3 models, lag length = 3, 2h avg
#4:     1       15              4000    0.1               0.01              3


#YL:     1       25              100     0.02        -       pow(10,1)          7+++
#YR:     1       25              100     0.02        -      pow(10,2)          7+++
#GL:     1       25              100     0.05        -      pow(10,-2)          7+++ lots of exc failures, aborted
#JL:     1       25              100     0.05        -      pow(10,-2) 
#L:      1       25              100     0.075        -      pow(10,2)
#C:
#R:     1       25              100    0.035        -       pow(10,2)          7+++ no failed opts, MSE = 0.99, MAE = 0.82, 8 CPs with one at correct place
#2:     1       25             100     0.02        -       pow(10,-4)          7+++   MAE = 0.92, MAE = 2.1 => actually worse than mean-pred
#2:     3       5              100     0.075        -       1                   3+++ MAE = 0.82, MSE = 1.02, too robust & only one CP at start 
#AL:    3       5              100     0.075        -       5                   7+++ MSE = 0.975, MAE = 0.79, one CP cluster + one in the middle
#AC:    3       5              100     0.075        -       10                  7+++ too many CPs (aoround 12), MSE & MAE not great, always AR chosen
#JL:    3       5              100     0.075        -       pow(10,3)           7+++ too many CPs (around >100) MSE & MAE not great
#R:     3       5               100     0.075       -       1                   1+++  seems to be too robust, but only 2 CPs, high MSE/MAE
#YL:    3       5               100     0.1         -       1                   1+++  only one CP, too robust (high MSE, MAE)
#C:     3       5               100     0.1                 pow(10,2)           1+++  CPs at 78, 140, 210, 360. MAE, MSE not great. Always AR-model chosen.
#L:     3       5               100     0.001                 pow(10,1)           1+++ GOOD TRY TO IMPROVE A few too many CPs, but very nice MSE, MAE
#2:     3       5               100     0.025                 pow(10,1)           1+++ GOOD RESULT: One CP at 174, MSE = 0.98, MAE = 0.801, switches from model 2 (euclid) to 1 (road)
#C:     3       5               100     0.01                5                     1+++ GOOD TRY TO IMPROVE A few too many CPs, but very nice MAE, MSE (fewer CPs than for 0.001)
#AC:    3       5               100     0.01                pow(10,-3)            1+++  4 CPs, but params don't move much

#TRYING TO IMPROVE THE ABOVE: (note: If this doesn't work, try with different b-values)
#L:     0.5       1               1000     0.01                 5                1+++ 17 CPs
#C:     0.5       1               1000    0.01                2.5               1+++  19CPs
#R:     0.5       1               1000    0.01                1               1+++    20 CPs
#2:     0.5       1               1000    0.01                0.5              1+++ MSE, MAE good but 27 CPs (not robust enough)
#L:     1        0.5             1000     0.01               2.5*pow(10,1)    1+++  way too many CPs
#C:     1        0.5              1000    0.01                5*pow(10,1)      1+++  GOOD RESULT: One CP at 174, two more at the end, MSE and MAE were nans but the fit was probably great!
#R:     1        0.5              1000    0.01                pow(10,2)        1+++  GOOD 5 CPs, MSE = 0.95, MAE = 0.8
#2:     1       0.5              1000     0.01                5*pow(10,2)       1+++ not robust enough, way too many CPs
#AC:    1       0.5             1000      0.01                pow(10,3)         1+++ not robust enough, way too many CPs
#2:     1       0.5             1000      0.01                7.5              1+++ not robust enough, 26 CPs
#AC:     1       0.5             1000      0.01                15              1+++  not robust enough, 15 CPs
#L:     3        5            1000     0.01               2.5*pow(10,1)         too many CPs and nans for MSE, MAE

#2:     3       5               100     0.05                5                   1+++  not robust enough, too many CPs
#JL:     3       5               100     0.06                5                   1+++ too robust, has only 1 CP
#L:     3       5               100     0.04                5                   1+++  GOOD 5 CPs, MSE = 1.019, MAE = 0.82
#C:     3        5             1000    0.025                5                   1+++  GOOD 6 CPs, MSE = 0.95, MAE = 0.79
#2:     1        0.5              1000    0.05                5*pow(10,1)       1+++  waaaay too many CPS (basically every obs a CP)
#AC:    1        0.5              1000    0.075                5*pow(10,1)      1+++  waaaay too many CPS (basically every obs a CP)

#2:     1        0.5              1000    0.01                5*pow(10,1)       1+++ not great, does not get good pred + too many CPs

#AC:    3       5               1000     0.025                0.25               1+++  13 CPs

#L:     3       5               1000     0.0001               0.25               1+++  25 CPs
#C:     3       5               1000     0.025                0.25               1+++  13 CPs
#R:     3       5               1000     0.05                0.25               1+++   7 CPs

#JL:    1        0.5              1000    0.01                5*pow(10,1)       1+++  4 CPs, okay MSE, MAE(should be good acc. to prev analysis)
#JR:    1        0.5              1000    0.01                7.5*pow(10,1)     1+++  1 CP middle + cluster at start, okay MSE, MAE

#GL:    1        0.5              1000    0.025                5*pow(10,1)      1+++  way too many CPs (every ob a CP)
#GC:    1        0.5              1000    0.025               7.5*pow(10,1)     1+++  way too many CPs (every ob a CP)
#GR:    1        0.5              1000    0.025               10*pow(10,1)      1+++  way too many CPs (every ob a CP)

#YL:    3       5                1000    0.075           -       5              1+++  6 CPs, not great MSE/MAE
#YR:     3       5               1000     0.075       -       2                 1+++  TRY TO REFINE 0 CPs, but very close to finding one in the middle! Not great for MSE/MAE

#2:    3       5               1000     0.025                 pow(10,1)          1+++ (should be 1 CP again), but finds 6


#2:    1       2500              1000     0.01        -       10                   1+++  does not find a single CP
#3:     1       2500               1000     0.025                5                 1+++  does not find a single CP, but param estimates move as we would want them to
#4:     1       2500               1000     0.025                1                 1+++  does not find a single CP

#L:    3        5              1000       0.0725                2                     1+++ 2 CPs, but one slightly too early & the other right at the end
#C:    3        5              1000       0.07                2                     1+++  4 CPs, not very brilliant MSE, MAE
#R:    1        2500              1000       0.07                2                   1+++ very decent MSE, MAE, too robust (cluster at start & no CP elsewhere)

#AL:   3       5               1000     0.025                 pow(10,1)         1+++, 2+++, 3+++ not great, 1 CP too early, large MSE & MAE
#AC:   3       5               1000     0.01                  0.05                  1+++, 2+++, 3+++ 

#vary RLD with a_rld = 0.05, 0.1, 0.25 L to R (0.25 failed latest)
#GL:   3       5              1000     0.01        -       10                   1+++  FAILED
#GC:   3       5              1000     0.01        -       10                   1+++   FAILED
#GR:   3       5              1000     0.01        -       10                   1+++   FAILED

#vary RLD with a_rld = 0.3, 0.5, 0.75 L to R (0.25 failed latest)
#GL:   3       5              1000     0.01        -       10                   1+++  FAILED
#GC:   3       5              1000     0.01        -       10                   1+++   FAILED
#GR:   3       5              1000     0.01        -       10                   1+++   FAILED
#GR:   3       5              1000     0.01        -       10                   1+++   rld=1, GOOD results but no CPs

#GL:   3       5               100     0.005                 pow(10,1)          1++ rld =1 too robust (no CP)
#GC:   3       5               100     0.001                 pow(10,1)          1++ rld =1 too robust (no CP)
#GR:   3       5               100     0.001                 pow(10,1)          1++ rld =1.5 too robust (no CP)



#JL:   3       5               1000     0.075       -       2                 1+++, 2+++, 3+++  CPs at start and end
#JC:   3       5               1000     0.07        -       2                 1+++, 2+++, 3+++  CPs at start and end
#JR:   3       5               1000     0.06       -       2                 1+++, 2+++, 3+++


#R:    1        0.5              1000       0.07                2                   1+++   too robust
#L:    3        50              1000       0.0725                2                     1+++

#2:     1       500               1000     0.025                5                 1+++  too robust (i.e. no CPs)
#4:     1       500               1000     0.025                5                 1+++  too robust (i.e. no CPs)
#4:     1       250               1000     0.025                5                 1+++  too robust (i.e. no CPs)
#3:     1       100               1000     0.025                5                 1+++
#2:     1       50               1000     0.025                5                 1+++   SAVED PERFECT. Works. MAE = 0.817, MSE=1.003
#4:     1       25               1000     0.025                5                 1+++   SAVED Also works. (slightly better ito MAE = 0.81, MSE= 1.01)

#C:     3       5               100     0.005                 pow(10,1)          1++ rld =2   too robust (2 cps at start)
#R:     3       5               100     0.005                 pow(10,1)          1++ rld =2.5 4 cps at start, last one at true val

#C:     3       50               100     0.005                 pow(10,1)          1++ rld =2   too robust (2 cps at start)
#R:     3       50               100     0.005                 pow(10,1)          1++ rld =2.5 4 cps at start, last one at true val


#AL:    1        0.5              1000    0.01                5*pow(10,1)         1+++ rld = 1.0 FAILED
#AL:    1        0.5              1000    0.01                5*pow(10,1)         1+++ rld = 2.0 FAILED
#AL:    1        0.5              1000    0.01                pow(10,1)         1+++ rld = 2.0 FAILED

#GL:    1       25               1000     0.025                5                 1+++  rld = 1.0 too robust (i.e. no CPs)

#Next round: Hopefully with multiple models!
#AFter that: Hopefully with alpha_rld

#Giulio: Better for smaller alpha_param, working for rld = 1.0.
#GL:    3       5               100     0.0005                 pow(10,1)          1++    rld = 1.0 too robust still
#GC:    3       5               100     0.0001                 pow(10,1)          1++    rld = 1.0 too robust still, best of the 3
#GR:    3       5               100     0.00005                 pow(10,1)          1++    rld = 1.0 too robust still

#My machine: try getting there via b
#C:     3       25               100     0.005                 pow(10,1)           1++   rld = 0.8 pretty good! Nearly finds the CP!
#R:     3       25               100     0.005                 pow(10,1)           1++   rld = 0.9 similar to C, but not as good

#L:     1       0.5             100     0.005                 pow(10,1)           1++   rld = 1.0  too robust (only 1 CP)
#C:     3       5               100     0.005                 pow(10,1)           1++   rld = 0.6   too robust (only 1 CP)
#R:     3       5               100     0.005                 pow(10,1)           1++   rld = 0.7   too robust (only 1 CP)


#Modify the 'working' result from my machine on Arne's
#AL:   1       50               1000     0.025                5                 3+++  exec fails, aborted
#AC:   1       50               1000     0.025                5                 7+++  exec fails, aborted

#Try the ones finding only one CP with smaller alphas and bigger b.
#2:     1       100               1000     0.01                5                 1+++  gets correct CP and finds it relatively early (improve on this!)
#4:     1       250               1000     0.005                5                 1+++ gets the correct CP, but finds it rather late

#2:     1       75               1000     0.01                5                 1+++   SAVED gets correct CP, finds it early 
#4:     1       100               1000     0.005                5                 1+++ 

#2:     1       25               1000     0.01                5                 1+++ EXCELLENT! Saved.

#AL:   3       25               100     0.005                 pow(10,1)           3+++  fails
#AC:   3       25               100     0.005                 pow(10,1)           7+++  fails
#AL:   3       25               100     0.005                 pow(10,3)           3+++  
#AC:   3       25               100     0.005                 pow(10,3)           7+++  

#2:     1       10               1000     0.01                5                 1+++ 4CPs (need slightly larger b)
#3:     1       5               1000     0.01                5                 1+++  6CPs (need slightly larger b)
#4:     1       1               1000     0.01                5                 1+++

#JL:   1       25               1000     0.01       -        5                 1+++, 2+++, 3+++  
#JC:   1       25               1000     0.01        -       5                 1+++:7+++


#GL:    1       5               100     0.0001                 pow(10,1)          1++    rld = 0.75   too robust (no CP)
#GC:    1       5               100     0.0001                 pow(10,1)          1++    rld = 0.5    too robust (no CP)
#GR:    1       5               100     0.0001                 pow(10,1)          1++    rld = 0.25   too robust (no CP)


#2:     1       12.5               1000     0.005               5               1+++  4 CPs
#3:     1       15               1000     0.005               5                 1+++  Good, but we have a second CP at 250, so slightly too liberal
#4:     1       20               1000     0.005                 5                 1+++ SAVED, 1 CP at 176, period of model uncertainty before that



#GL:    1       1               100     0.0001                 pow(10,1)          1++    rld = 1     gets CP at 150, but way too robust
#GC:    1       0.1               100     0.0001                 pow(10,1)          1++    rld = 2      too robust, gets no CPs
#GR:    1       0.01               100     0.0001                 pow(10,1)          1++    rld = 2  gets 2 CPs, one at correct pos & one at end (i.e., not rob enough)


#2:     1       17.5              1000     0.005                 5                1+++:7+++  some issues during exec 
#3:     1       20               1000     0.005                 5                 1+++:7+++  many issues during exec
#4:     1       50               1000     0.005                5                 1+++:7+++ , rld = 1 FAILED

#L:     1       20                100     0.01                 5           1++:7+++   rld = 0.25 
#C:     1       20               100     0.01                 5           1++:7+++   rld = 0.1   (failed for 0.05, 0.1, 0.25)
#R:     1       20               100     0.01                 5           1++:7+++   rld = 0.01  


#AL:    1       20               1000     0.005                 5                 1+++:3+++ SAVED on Arne's machine Nice results, but only 1+++ part used anyways + 1 CP too many at end
#       MSE = 0.820 [total: 13.85 (3.11)] ()  MAE = 0.748 [12.64 (2.3)] ()

#AC:     1       50               1000     0.005                5                 1+++:3+++ Very clear change around CG with single CP, MSE/MAE values old due to false T.P.


#KL-comparison: My machine
#4:      1       20               1000     0.005                 5                  1+++ Every obs a CP
#4:      1       20               1000     0.005                 pow(10,-3)         1+++ 
#4:      100       25               1000     0.005                 0.0005           1+++ 20+ CPs, only one model


#Make it work with rld, My office machine
#L:     1       20               1000     0.005                 5           1++  rld = 0.1     RL not robust enough 
#C:     1       20               1000     0.005                 5           1++   rld = 0.05   RL not robust enough 
#R:     1       20               1000     0.005                 5           1++   rld = 0.01   RL not robust enough 

#Make it work with rld, Guilio
#GL:    1       20               1000     0.005                 5           1++    rld = 0.005     not robust enough
#GC:    1       20               1000     0.005                 5           1++    rld = 0.25       too robust  
#GR:    1       20               1000     0.005                 5           1++    rld = 0.5        too robust  

#AL:    1       20               1000     0.005                 5           1++    rld = 0.001  RL not robust enough 
#AC:    1       20               1000     0.005                 5           1++    rld = 0.15   RL robust enough but MSE/MAE not great  

#some of the same values as before to check that it is not inferential accuracy
#AL:    1       20               1000     0.005                 5           1++    rld = 0.01  RL not robust enough 
#AC:    1       20               1000     0.005                 5           1++    rld = 0.05   RL robust enough but MSE/MAE not great  


#AL:    1       30               1000     0.005                 5           1++    rld = 0.05    SAVED Perfect! Works
#AC:    1       50               1000     0.005                 5           1++    rld = 0.05    1 CP too many at end

#GL:    1       20               1000     0.005                 5           1++    rld = 0.01     not robust enough
#GC:    1       20               1000     0.005                 5           1++    rld = 0.05      not robust enough  
#GR:    1       20               1000     0.005                 5           1++    rld = 0.075     not robust enough

#KL-comparison:
#2:
#3:   

#TOO ROBUST for rld = 0.2, 0.25, 0.3
#L:     1       20               1000     0.005                 5           1++   rld = 0.125  works, but we find CP rather late
#C:     1       20               1000     0.005                 5           1++   rld = 0.15  too robust
#R:     1       20               1000     0.005                 5           1++   rld = 0.175 too robust


#GL:    1       25               1000     0.005                 5           1++    rld = 0.05     not robust enough
#GC:    1       30               1000     0.005                 5           1++    rld = 0.05      not robust enough  
#GR:    1       40               1000     0.005                 5           1++    rld = 0.05     not robust enough


#GL:    1       25               1000     0.005                 5           1++    rld = 0.05     not robust enough
#GC:    1       30               1000     0.005                 5           1++    rld = 0.05      not robust enough  
#GR:    1       40               1000     0.005                 5           1++    rld = 0.05     not robust enough


#GL:    1       20               1000     0.005                 5           1++    rld = 0.05     
#GC:    1       15               1000     0.005                 5           1++    rld = 0.05      
#GR:    1       10               1000     0.005                 5           1++    rld = 0.05     


#AL:    1       20               1000     0.005                 5           1++    rld = 0.1      SAVED. USED IN PLOT
#AC:    1       20               1000     0.005                 5           1++    rld = 0.08     not robust enough

#AL:    1       20               1000     0.005                 5           1++    rld = 0.09
#AC:    1       20               1000     0.005                 5           1++    rld = 0.1  + alpha learning enabled 


#4:     1       100               1000     0.005                 5           1++    rld = 0.1 init; LEARNIGN + SAVED CP correct + nice model change!

#L,C,R: Trying to see what happens when I learn alpha

#NOTE: b seems to go down when we do hyperpar opt 
#NOTE: Seems that we should not have shrinkage at values < 1 (?)
#NOTE: Seems that too large b-values don't help us 
#NOTE: alpha_param should be <0.2 (too many failed opts for 0.2), but probably also larger than 0.01 (we get a lot of CPs else)
#Note: Got better res with
#       1       100             1000        0.01                20              1++  rld = 0.1, + LEARNING
#same setting on AL with ap= 0.025 and AC ap=0.075 both too robust

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


DPD = True
KL = False

lags = [1] #[1,2,3]
road_model, euclid_model = True, True
model_universe = []

for lag in lags:
    
    res_tight_week = [[0]]*lag
    
    if DPD:
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
    if KL:
        model_universe = model_universe + [
                    BVARNIG(
                         prior_a=a, 
                         prior_b=b, #b, 
                         S1=S1, 
                         S2=S2, 
                         #alpha_param = alpha_param,
                         prior_mean_beta=None, 
                         prior_var_beta=None,
                         prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                         prior_var_scale=shrinkage,
                         general_nbh_sequence=[[[]]]*S1*S2,
                         general_nbh_restriction_sequence = res_tight_week,
                         general_nbh_coupling = "weak coupling", 
                         hyperparameter_optimization = "online", #"online", #"online", #"online",
                         intercept_grouping = grouping
                    )
                ]
                    
        #creates road nbh model 
        if road_model:           
            model_universe = model_universe +[
                        BVARNIG(
                             prior_a=a, 
                             prior_b=b, #b, 
                             S1=S1, 
                             S2=S2, 
                             #alpha_param = alpha_param,
                             prior_mean_beta=None, 
                             prior_var_beta=None,
                             prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                             prior_var_scale=shrinkage,
                             general_nbh_sequence=road_nbhs,
                             general_nbh_restriction_sequence = res_tight_week,
                             general_nbh_coupling = "weak coupling", 
                             hyperparameter_optimization = "online", #"online", #"online", #"online",
                             intercept_grouping = grouping
                        )
                    ]
        
        #creates euclid nbh model
        if euclid_model:
            model_universe = model_universe + [
                        BVARNIG(
                             prior_a=a, 
                             prior_b=b, #b, 
                             S1=S1, 
                             S2=S2, 
                             #alpha_param = alpha_param,
                             prior_mean_beta=None, 
                             prior_var_beta=None,
                             prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                             prior_var_scale=shrinkage,
                             general_nbh_sequence=euclid_nbhs,
                             general_nbh_restriction_sequence = res_tight_week,
                             general_nbh_coupling = "weak coupling", 
                             hyperparameter_optimization = "online", #"online", #"online", #"online",
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
        generalized_bayes_rld = rld, #"power_divergence", #"kullback_leibler", #"power_divergence" , #"power_divergence", #"kullback_leibler",
        alpha_param_learning =  param_learning,#"together", #"individual", #"individual", #"individual", #"individual", #"together",
        alpha_param  = alpha_param, 
        alpha_param_opt_t = 100, #100, (100 used for paper!)
        #alpha_rld_opt_t=0 #default 
        alpha_rld = alpha_rld, #pow(10, -5), #0.25,
        alpha_rld_learning = rld_learning, #"power_divergence",
        #alpha_rld = 0.25, #0.00000005,pow(10,-12)
        #alpha_rld_learning=True,
        loss_der_rld_learning="absolute_loss")
detector.run()

"""Store results + real CPs into EvaluationTool obj"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_run_detector(detector)

      
#"""store that EvT object onto hard drive"""
#prior_spec_str = ("//" + cp_type + "//a=" + str(a) + 
#    "//b=" + str(b) )
# 1       20               1000     0.01                5                 1+++ 
#detector_path = (results_directory  + "//AirPollution") #+ 
##            #prior_spec_str + "//daily_good_recreated")
##if not os.path.exists(detector_path):
##    os.makedirs(detector_path)
##
#results_path = detector_path + "//LEARNING_results_a=1_b=100_int=1000_ap=001_shr=20_mod=1++.txt" 
#EvT.store_results_to_HD(results_path)



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
        #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
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
    #event_time_list=[715 ],
    #label_list=["nilometer"], space_to_colorbar = 0.52,
    custom_colors = ["blue", "blue"] * 30, 
    custom_linestyles = ["solid"]*30,
    custom_linewidth = 3,
    #arrow_colors= ["black"],
    #number_fontsize = 14,
    #arrow_length = 135,
    #arrow_thickness = 3.0,
    xlab_fontsize =14,
    ylab_fontsize = 14, 
    #arrows_setleft_indices = [0],
    #arrows_setleft_by = [50],
    #zero_distance = 0.0,
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
print("MSE is", np.mean(detector.MSE), 1.96*scipy.stats.sem(detector.MSE))
print("MAE is", np.mean(detector.MAE), 1.96*scipy.stats.sem(detector.MAE))
print("NLL is", np.mean(detector.negative_log_likelihood), 1.96*scipy.stats.sem(detector.negative_log_likelihood))
print("rld model is",detector.generalized_bayes_rld )
print("RLD alpha is", detector.alpha_rld)
#print("param inf uses", mode)
print("param alpha is", detector.model_universe[0].alpha_param)
print("intensity is", intensity)
print("shrinkage is", shrinkage)
print("alpha param learning:", detector.alpha_param_learning)
print("failed optimizations:", detector.model_universe[0].failed_opt_count/
          detector.model_universe[0].opt_count)


#fig = EvT.plot_predictions(
#        indices = [0], print_plt = True, 
#        legend = False, 
#        legend_labels = None, 
#        legend_position = None, 
#        time_range = None,
#        show_var = False, 
#        show_CPs = True)
#plt.close(fig)
#fig = EvT.plot_run_length_distr(
#    print_plt = True, 
#    time_range = None,
#    show_MAP_CPs = True, 
#    show_real_CPs = False,
#    mark_median = False, 
#    log_format = True,
#    CP_legend = False, 
#    buffer = 50)
#plt.close(fig)
#
#print("MSE", np.mean(detector.MSE))
#print("NLL", np.mean(detector.negative_log_likelihood))
#print("a", a)
#print("b", b)
#print("intensity", intensity)
#print("beta var prior", var_scale )





