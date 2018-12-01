#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:04:27 2018

@author: jeremiasknoblauch

Description: Well-log data processing
"""

"""System packages/modules"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv
import datetime
import matplotlib

"""Modules of the BOCPDMS algorithm"""
from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from BVAR_NIG_DPD import BVARNIGDPD
from detector import Detector
from Evaluation_tool import EvaluationTool


baseline_working_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS/Code//SpatialBOCD//Data//well log") 
well_file = baseline_working_directory + "//well.txt"


mode = "DPD" #Two modes available: 
             #"DPD" -> Density Power Divergence, which is
             #         the same as the beta-divergence.
             #"KL"  -> Kullback Leibler Divergence, i.e. standard bayes

"""STEP 1: Read in the nile data from well.txt"""
raw_data = []
count = 0 
with open(well_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        raw_data += row

raw_data_float = []
for entry in raw_data:
    raw_data_float.append(float(entry))
raw_data = raw_data_float

"""STEP 2: Format the data so that it can be processed with a Detector
object and instantiations of ProbabilityModel subclasses"""
T = int(len(raw_data))
S1, S2 = 1,1 #S1, S2 give you spatial dimensions
data = np.array(raw_data).reshape(T,1,1)


"""STEP 3: Set up the optimization parameters. These only apply if mode = "DPD"
           since the standard Bayesian inference is exact"""
VB_window_size = 360 #W in pseudo-code of paper
full_opt_thinning = 20 #i.e., every when the run-length is divisible by 20, 
                       #we perform a full optimization step
SGD_approx_goodness = 10 #In the pure SGD steps, how big is your batch
anchor_approx_goodness_SVRG = 50 #in the SVRG steps, how big is your batch (NOT USED)
anchor_approx_goodness_SCSG = 25 #in the SCSG steps, how big is your batch
first_opt = 10 
alpha_param_opt_t = 0 #Indicates how many time period you wait before learning 
                      #about beta (alpha in DPD notation of Basu et al. ('98))


"""STEP 4: Set up the priors for the model universe's elements"""                                     
              
a, b = 1, pow(10,7)
alpha_param = 0.05 #Initialization for the parameter-beta (alpha)
alpha_rld = 0.0001 #Initialization for the run-length-beta (alpha)
rld_DPD = "power_divergence" #The run-length robustification can be set inde-
                            #pendently from the parameter robustness.

a_KL, b_KL = 1, pow(10,4)
rld_KL = "kullback_leibler" #The run-length robustification can be set inde-
                            #pendently from the parameter robustness.
    
rld_learning = True #Whether or not we learn about the beta (alpha) 
                    #robustifying the run-length (if rld = "power_divergence")
param_learning = "individual" #Irrelevant for well-log because we only run
                              #a single model.

#Set Prior mean and variance scales
prior_mean_scale, prior_var_scale = np.mean(data), 0.25 
cp_intensity = 100 # cp_intensity = k => prior prob P(CP at time t) = 1/k
np.random.seed(999) #To exactly reproduce paper results/pictures

"""STEP 5: Create models"""
model_universe_DPD = []
model_universe_KL = []
model_universe_DPD = model_universe_DPD + [BVARNIGDPD(
             prior_a=a, 
             prior_b=b, #b, 
             S1=S1, 
             S2=S2, 
             alpha_param = alpha_param,
             prior_mean_beta=None, 
             prior_var_beta=None,
             prior_mean_scale=prior_mean_scale, 
             prior_var_scale=prior_var_scale,
             nbh_sequence=None,
             restriction_sequence = None,
             hyperparameter_optimization = "online", 
             VB_window_size = VB_window_size,
             full_opt_thinning = full_opt_thinning,
             SGD_batch_size = SGD_approx_goodness,
             anchor_batch_size_SCSG = anchor_approx_goodness_SCSG,
             anchor_batch_size_SVRG = None, 
             first_full_opt = first_opt
        )]
model_universe_KL = model_universe_KL + [BVARNIG(
                prior_a = a_KL,
                prior_b = b_KL,
                S1 = S1,
                S2 = S2,
                prior_mean_scale = prior_mean_scale,
                prior_var_scale = prior_var_scale,
                nbh_sequence = None,
                restriction_sequence = None,
                hyperparameter_optimization = "online" 
        )]

    
"""STEP 6: Set up the detectors from this"""
model_universe_DPD = np.array(model_universe_DPD)
model_universe_KL = np.array(model_universe_KL)
model_prior = np.array([1.0/len(model_universe_DPD)]*len(model_universe_DPD))
cp_model = CpModel(cp_intensity)


detector_DPD = Detector(
        data=data, 
        model_universe=model_universe_DPD, 
        model_prior = model_prior,
        cp_model = cp_model, 
        S1 = S1, 
        S2 = S2, 
        T = T, 
        store_rl=True, 
        store_mrl=True,
        trim_type="keep_K", 
        threshold = 50,
        notifications = 100,
        save_performance_indicators = True,
        generalized_bayes_rld = rld_DPD, 
        alpha_param_learning =  param_learning, 
        alpha_param  = alpha_param, 
        alpha_param_opt_t = 100, 
        alpha_rld = alpha_rld, 
        alpha_rld_learning = rld_learning, 
        loss_der_rld_learning="absolute_loss"
        )
detector_DPD.run()

detector_KL = Detector(
        data=data, 
        model_universe=model_universe_KL, 
        model_prior = model_prior,
        cp_model = cp_model, 
        S1 = S1, 
        S2 = S2, 
        T = T, 
        store_rl=True, 
        store_mrl=True,
        trim_type="keep_K", 
        threshold = 50,
        notifications = 100,
        save_performance_indicators = True,
        generalized_bayes_rld = rld_KL, 
        alpha_param_learning =  param_learning, 
        alpha_param  = alpha_param, 
        alpha_param_opt_t = 100, 
        alpha_rld = alpha_rld, 
        alpha_rld_learning = rld_learning, 
        loss_der_rld_learning="absolute_loss"
        )
detector_KL.run()


"""STEP 7: Make graphing tool"""
EvTDPD = EvaluationTool()
EvTDPD.build_EvaluationTool_via_run_detector(detector_DPD)
EvTKL = EvaluationTool()
EvTKL.build_EvaluationTool_via_run_detector(detector_KL)
        
    
"""STEP 8: Plotting Pictures in paper"""
matplotlib.rcParams.update({'figure.autolayout': False})


"""Get the different CPs"""
CPsDPD = np.array([e[0] for e in EvTDPD.results[EvTDPD.names.index("MAP CPs")][-2]])
CPsKL = np.array([e[0] for e in EvTKL.results[EvTKL.names.index("MAP CPs")][-2]])

k  = 25
additional_CPs = []
for cp_kl in CPsKL:
    lower = CPsDPD - k < cp_kl
    upper = CPsDPD + k > cp_kl
    if (not np.any(lower == upper)):
        additional_CPs.append([cp_kl,0])

height_ratio =[10,4,8]

KL_CP_color = "crimson"
DPD_CP_color = "darkblue"
max_color_KL = "red"
max_color_DPD = "blue"
max_width = 1 
CP_linewidth_DPD = 2 
CP_linewidth_KL = 1 
CP_style_KL = (0,(1,2.25))
CP_style_DPD = "solid" 
CP_transparence_KL = 0.75
CP_transparence_DPD = 0.5
show_CPs_in_rld = False

xlabsize, ylabsize, ticksize = 15,15,12 

fig, ax_array = plt.subplots(3,  
                             figsize=(18,10), 
                             sharex = True,  
                             gridspec_kw = {'height_ratios':height_ratio})
fig.subplots_adjust(hspace = .05,
                    left = None, bottom = None,
                    right = None, top = None)
ylabel_coords = [0.0, 0.25] 


EvTDPD.plot_raw_TS(data.reshape(T,S1*S2), indices = [0], xlab = None, 
        show_MAP_CPs = True, 
        time_range = np.linspace(1,T, T, dtype=int), 
        print_plt = False,
        ylab = "Response", 
        ax = ax_array[0], 
        custom_colors_series = ["black"]*5,
        custom_colors_CPs = [DPD_CP_color]* 100,
        custom_linestyles = [CP_style_DPD]*100,
        custom_linewidth = CP_linewidth_DPD,
        custom_transparency = CP_transparence_DPD,
        ylab_fontsize = ylabsize,
        yticks_fontsize = ticksize,
        ylabel_coords = [-0.06,0.5], 
        additional_CPs = additional_CPs,
        custom_colors_additional_CPs = [KL_CP_color] * 100,
        custom_linewidth_additional_CPs = CP_linewidth_KL,
        custom_linestyles_additional_CPs = [CP_style_KL] * 10,
        custom_transparency_additional_CPs = CP_transparence_KL)


EvTDPD.plot_run_length_distr(buffer=0, show_MAP_CPs = show_CPs_in_rld, 
    mark_median = False, 
    mark_max = True, 
    upper_limit = 1300, 
    print_colorbar = False, 
    colorbar_location= None,
    xlab = "",
    ylab = "", 
    log_format = False, aspect_ratio = 'auto', 
    time_range = np.linspace(1,
                             T-2, 
                             T-2, dtype=int), 
    start = 1, stop = T, 
    all_dates = None, 
    custom_colors = [DPD_CP_color] * 30, 
    custom_linestyles = [CP_style_DPD]*30,
    custom_linewidth = CP_linewidth_DPD,
    xlab_fontsize = xlabsize,
    ylab_fontsize = ylabsize, 
    xticks_fontsize = ticksize,
    yticks_fontsize = ticksize,
    ax = ax_array[1], figure = fig,
    no_transform = True,
    date_instructions_formatter = None, 
    date_instructions_locator = None,
    arrow_distance = 25,
    mark_max_linewidth = max_width,
    mark_max_color = max_color_DPD)


EvTKL.plot_run_length_distr(buffer=0, show_MAP_CPs = show_CPs_in_rld, 
                                   mark_median = False, 
    mark_max = True, upper_limit = 1200, 
    print_colorbar =  True, 
    colorbar_location= 'bottom',
    space_to_colorbar = 0.6, 
    log_format = False, aspect_ratio = 'auto', 
    C1=0,C2=700, 
    time_range = np.linspace(1,
                             T-2, 
                             T-2, dtype=int), 
    start = 1, stop = T, 
    all_dates = None, 
    custom_colors = [KL_CP_color] * 30, 
    custom_linestyles = [CP_style_KL]*30,
    custom_linewidth = CP_linewidth_KL,
    xlab_fontsize =xlabsize,
    ylab_fontsize = ylabsize, 
    xticks_fontsize = ticksize,
    yticks_fontsize = ticksize,
    ylabel_coords = [-0.06, 1.25], 
    ax = ax_array[2], figure = fig,
    no_transform = True,
    date_instructions_formatter = None, 
    date_instructions_locator = None,
    xlab = "Time",
    ylab = "run length", 
    arrow_distance = 25,
    mark_max_linewidth = max_width,
    mark_max_color = max_color_KL)

   
fig.savefig(baseline_working_directory + "//well.pdf",
            format = "pdf", dpi = 800)  
fig.savefig(baseline_working_directory + "//well.jpg",
        format = "jpg", dpi = 800)  


    
"""STEP 9: Plot some performance metrics"""
def abs_loss_lim(x, lim):
    x[np.where(x >= lim)] = lim
    return np.abs(x)
sd =  np.sqrt(np.var(data))

print("CPs are ", detector_DPD.CPs[-2])
train = 0
until = -2
resids = (data[1:] - EvTKL.results[10].reshape(T,1)[:-1])[train:until]
print("summary MSE KL:", 
      np.mean(np.power((data[1:] - 
                        EvTKL.results[10].reshape(T,1)[:-1])[train:until],2)))
print("summary MAE KL:", 
      np.mean(np.abs((data[1:] - 
                      EvTKL.results[10].reshape(T,1)[:-1])[train:until])))

resids = (data - EvTDPD.results[10].reshape(T,1)[:-1])[train:until]
print("summary MSE DPD:", 
      np.mean(np.power(((data - 
                    EvTDPD.results[10].reshape(T,1)[:-1]))[train:until],2)))
print("summary MAE DPD:", 
      np.mean(np.abs(((data - 
                    EvTDPD.results[10].reshape(T,1)[:-1]))[train:until])))


