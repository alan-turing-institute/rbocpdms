#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:17:34 2018

@author: jeremiasknoblauch

Description: Create AR(1) processes moving independently with CPs, and one s
             contaminated series which is AR(1) + massive errors.
"""

import numpy as np
import scipy
from BVAR_NIG_DPD import BVARNIGDPD
from BVAR_NIG import BVARNIG
from detector import Detector
from cp_probability_model import CpModel
from Evaluation_tool import EvaluationTool
import matplotlib.pyplot as plt


"""STEP 1: Set up the simulation"""
normalize = True
mode = "DPD" #KL, both
K = 50 #number of series
k = 1   #number of contaminated series
T = 600
burn_in = 100
data = np.zeros((T,K,1))
AR_coefs = [np.ones(K) * (-0.5),
            np.ones(K) * 0.75, #,
            np.ones(K) * -0.7]
levels = [np.ones(K) * 0.3, 
          np.ones(K) * (-0.25), #,
          np.ones(K) * 0.3]
CP_loc = [200,400] #, 400]
contamination_df = 4
contamination_scale = np.sqrt(5)

#T=600 setting: Only 1 CP too many!
#rld model is power_divergence
#RLD alpha is 0.1
#param inf uses DPD
#param alpha is 0.25
#intensity is 100
#shrinkage is 0.05
#alpha param learning: None

"""STEP 2: Run the simulation with contamination for i<k"""
for cp in range(0, len(CP_loc) + 1):
        #Retrieve the correct number of obs in segment
    if cp == 0:
        T_ = CP_loc[0] + burn_in
        start = 0
        fin = CP_loc[0]
    elif cp==len(CP_loc):
        T_ = T - CP_loc[cp-1]
        start = CP_loc[cp-1]
        fin = T #DEBUG: probably wrong.
    else:
        T_ = CP_loc[cp] - CP_loc[cp-1]
        start = CP_loc[cp-1]
        fin = CP_loc[cp]
        
    #Generate AR(1)  
    for i in range(0,K):
        np.random.seed(i)
        next_AR1 = np.random.normal(0,1,size=T_) 
        for j in range(1, T_):
            next_AR1[j] = next_AR1[j-1]*AR_coefs[cp][i] + next_AR1[j]  + levels[cp][i]
            
        #if i < k, do contamination
        if i<k:
            np.random.seed(i*20)
            contam = contamination_scale*scipy.stats.t.rvs(contamination_df, size=T_)
            contam[np.where(contam <3)] = 0
            next_AR1 = (next_AR1 + contam)
        
        #if first segment, cut off the burn-in
        if cp == 0:
            next_AR1 = next_AR1[burn_in:]
        
        #add the next AR 1 stream into 'data'
        data[start:fin,i,0] = next_AR1

"""STEP 3: Set up analysis parameters"""
S1, S2 = K,1 #S1, S2 give you spatial dimensions
if normalize:
    data = (data - np.mean(data))/np.sqrt(np.var(data))


"""STEP 3: Set up the optimization parameters"""
VB_window_size = 200
full_opt_thinning = 20
SGD_approx_goodness = 10
anchor_approx_goodness_SCSG = 25
anchor_approx_goodness_SVRG = 25
alpha_param_opt_t = 0 #don't wait with training
first_full_opt = 10


"""STEP 4: Set up the priors for the model universe's elements"""
#got good performance for T=200, K=2, a_p = 0.4, a_rld = 0.05
#got performance for T = 200, K = 2, a_p = 0.25, a_rld = 0.08, shrink = 0.1, int = 600
#For T=600, K=2 pretty good performance for
#rld model is kullback_leibler
#RLD alpha is 0.25
#param inf uses DPD
#param alpha is 0.1
#intensity is 50
#shrinkage is 0.1
#alpha param learning: None
############
#also good:
#rld model is power_divergence
#RLD alpha is 0.1
#param inf uses DPD
#param alpha is 0.3
#intensity is 100
#shrinkage is 0.1
#alpha param learning: None
############
#also good: (2 cps, but one too early)
#rld model is power_divergence
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.2
#intensity is 100
#shrinkage is 0.1
#alpha param learning: None
###########
#porentially also good, though CPs were at wrong places:
#rld model is power_divergence
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.3
#intensity is 100
#shrinkage is 0.25
#alpha param learning: None

###########
#PERFECT: T = 600, K = 5
#rld model is power_divergence
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.3
#intensity is 100
#shrinkage is 0.1
#alpha param learning: None

###########
#NEAR PERFECT: T = 1000, K = 5
#rld model is power_divergence
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.3
#intensity is 100
#shrinkage is 0.5
#alpha param learning: None

###########
#NEAR PERFECT: T = 600, K = 5
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.2
#intensity is 100
#shrinkage is 100
#alpha param learning: None

###########
#SAVED VERSION:
#rld model is power_divergence
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.2
#intensity is 100
#shrinkage is 100
#alpha param learning: None
#window = 200, thinning = 20, SGD approx = 10, anchors = 25, first full opt =10

###########
#FOR K = 50: Really good except we get 4 too many CPs in first segment
#rld model is power_divergence
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.1
#intensity is 100
#shrinkage is 100
#alpha param learning: None
#similar for a_p = 0.25 
#With a_p = 0.3 or 0.35 we get really good result! (if using KL)

a, b = 3,5
alpha_param = 0.9
alpha_rld = 0.35
rld = "kullback_leibler" #power_divergence kullback_leibler
rld_learning = False
param_learning = None #"individual" #"individual" #"individual"

prior_mean_scale, prior_var_scale = np.mean(data), 100 #np.sqrt(np.var(data))
cp_intensity = 100

"""STEP 5: Create models"""
model_universe = []
if mode == "DPD" or mode == "both":
    model_universe = model_universe + [BVARNIGDPD(
                 prior_a=a, 
                 prior_b=b, #b, 
                 S1=S1, 
                 S2=S2, 
                 alpha_param = alpha_param,
                 prior_mean_beta=None, 
                 prior_var_beta=None,
                 prior_mean_scale=prior_mean_scale, #prior_mean_scale, 
                 prior_var_scale=prior_var_scale,
                 general_nbh_sequence=[[[]]]*S1*S2,
                 general_nbh_restriction_sequence = [[0]],
                 general_nbh_coupling = "weak coupling", 
                 hyperparameter_optimization = "online", #"online", #"online", #"online",
                 VB_window_size = VB_window_size,
                 full_opt_thinning = full_opt_thinning,
                 SGD_batch_size = SGD_approx_goodness,
                 anchor_batch_size_SCSG = anchor_approx_goodness_SCSG,
                 anchor_batch_size_SVRG = anchor_approx_goodness_SVRG,
                 first_full_opt = first_full_opt
            )]
if mode == "KL" or mode == "both":
    model_universe = model_universe + [BVARNIG(
                    prior_a = a,
                    prior_b = b,
                    S1 = S1,
                    S2 = S2,
                    prior_mean_scale = prior_mean_scale,
                    prior_var_scale = prior_var_scale,
                    general_nbh_sequence = [[[]]]*S1*S2,
                    general_nbh_restriction_sequence = [[0]],
                    hyperparameter_optimization = "online" #"online"
            )]

"""STEP 6: Set up the detector from this"""
model_universe = np.array(model_universe)
model_prior = np.array([1.0/len(model_universe)]*len(model_universe))
cp_model = CpModel(cp_intensity)
detector = Detector(
        data=data, 
        model_universe=model_universe, 
        model_prior = model_prior,
        cp_model = cp_model, 
        S1 = S1, 
        S2 = S2, 
        T = T, 
        store_rl=True, 
        store_mrl=True,
        trim_type="keep_K", 
        threshold = 200,
        notifications = 25,
        save_performance_indicators = True,
        generalized_bayes_rld = rld, #"power_divergence", #"kullback_leibler", #"power_divergence" , #"power_divergence", #"kullback_leibler",
        alpha_param_learning =  param_learning,#"together", #"individual", #"individual", #"individual", #"individual", #"together",
        alpha_param  = alpha_param, 
        alpha_param_opt_t = 100, #, #) #,
        alpha_rld = alpha_rld, #pow(10, -5), #0.25,
        alpha_rld_learning = rld_learning, #"power_divergence",
        #alpha_rld = 0.25, #0.00000005,pow(10,-12)
        #alpha_rld_learning=True,
        loss_der_rld_learning="absolute_loss")
detector.run()




"""STEP 7: Make graphing tool"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_run_detector(detector)
        

"""STEP 8: Inspect convergence of the hyperparameters"""
for lag in range(0, len(model_universe)):
    plt.plot(np.linspace(1,len(detector.model_universe[lag].a_list), 
                         len(detector.model_universe[lag].a_list)), 
             np.array(detector.model_universe[lag].a_list))
    plt.plot(np.linspace(1,len(detector.model_universe[lag].b_list),
                         len(detector.model_universe[lag].b_list)), 
             np.array(detector.model_universe[lag].b_list))

"""STEP 9+: Inspect convergence of alpha-rld"""
if detector.generalized_bayes_rld == "power_divergence" and mode == "DPD":
    plt.plot(detector.alpha_list)
    for lag in range(0, len(model_universe)):
        plt.plot(detector.model_universe[lag].alpha_param_list)


"""STEP 10: Plot the raw data + rld"""
height_ratio =[10,14]
custom_colors = ["blue", "purple"] 
fig, ax_array = plt.subplots(2, figsize=(8,5), sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio})
plt.subplots_adjust(hspace = .35, left = None, bottom = None,
                    right = None, top = None)
ylabel_coords = [-0.065, 0.5]

#Plot of raw Time Series
EvT.plot_raw_TS(data.reshape(T,S1,S2), indices = [0,1], xlab = None, 
        show_MAP_CPs = True, 
        time_range = np.linspace(1,T, T, dtype=int), 
        print_plt = False,
        ylab = "value", ax = ax_array[0],
        #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
        custom_colors_series = ["black"]*4,
        custom_colors_CPs = ["blue", "blue"]* 100,
        custom_linestyles = ["solid"]*100,
        ylab_fontsize = 14,
        ylabel_coords = ylabel_coords)
                           
#Run length distribution plot
EvT.plot_run_length_distr(buffer=0, show_MAP_CPs = False, 
                                   mark_median = False, 
    mark_max = True, upper_limit = 1000, print_colorbar = True, 
    colorbar_location= 'bottom',log_format = True, aspect_ratio = 'auto', 
    C1=0,C2=1, 
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
    
"""STEP 11: Plot some performance metrics"""
print("CPs are ", detector.CPs[-2])
print("MSE is", np.mean(detector.MSE), 1.96*scipy.stats.sem(detector.MSE))
print("MAE is", np.mean(detector.MAE), 1.96*scipy.stats.sem(detector.MAE))
print("NLL is", np.mean(detector.negative_log_likelihood),1.96*scipy.stats.sem(detector.MSE))
print("rld model is",detector.generalized_bayes_rld )
print("RLD alpha is", detector.alpha_rld)
print("param inf uses", mode)
print("param alpha is", detector.model_universe[0].alpha_param)
print("intensity is", cp_intensity)
print("shrinkage is", prior_var_scale)
print("alpha param learning:", detector.alpha_param_learning)

#baseline_working_directory = "//Users//jeremiasknoblauch//Documents//OxWaSP"+
#    "//BOCPDMS/Code//SpatialBOCD//PaperNIPS"
#results_path = baseline_working_directory + "//KL_K=5_k=1_T=600_2CPs_results_arld=015_ap=02_nolearning_rld_dpd_int=100_shrink=100.txt" 
#EvT.store_results_to_HD(results_path)
#fig.savefig(baseline_working_directory + "//well_log" + mode + ".pdf",
#            format = "pdf", dpi = 800)
#    
    
#    


#NOTE: We need a way to use BVARNIG(DPD) for only fitting a constant!
