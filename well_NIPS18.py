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
import os
"""Modules of the BOCPDMS algorithm"""
from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from BVAR_NIG_DPD import BVARNIGDPD
from detector import Detector
from Evaluation_tool import EvaluationTool

baseline_working_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS/Code//SpatialBOCD//Data//well log") 
well_file = baseline_working_directory + "//well.txt"

BOCPDMS_accepted = False #if BOCPDMS work accepted, we run the analysis with
                         #multiple models (perhaps)
mode = "DPD" #"KL", "both"
normalize = False #simply for numerical stability 
shortened = False
shortened_to = 1500


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
if shortened:
    raw_data = raw_data[:shortened_to]
T = int(len(raw_data))
S1, S2 = 1,1 #S1, S2 give you spatial dimensions
data = np.array(raw_data).reshape(T,1,1)
if normalize:
    data = (data - np.mean(data))/np.sqrt(np.var(data))


"""STEP 3: Set up the optimization parameters"""
VB_window_size = 360
full_opt_thinning = 20
SGD_approx_goodness = 10
anchor_approx_goodness_SVRG = 50
anchor_approx_goodness_SCSG = 25
first_opt = 10 
alpha_param_opt_t = 0 #don't wait with training
#alpha_param = 1 #conservative initialization
#alpha_rld = 0.1 #conservative initialization, note that we don't use it yet
#loss_der_rld_learning = (Detector.
#loss_param_learning


"""STEP 4: Set up the priors for the model universe's elements"""

 #0.5 pretty good, but not robust enough yet, 0.75 too high
                   #good performance for a_p = 0.2, KL RLD
                   #Note: We get good results for ap = 0.5, arld =  0.175 with rld and ap opt.
                   #        what we observe is that ap stays at around 0.5, and aprld goes to 0 quick.
                   #good results with MSE = 0.32 and only 2 missed CPs for arld = 0.15, RLD-powdiv active
                   #    alpha_param learning active (final value 0.67, started close to 0.2), alpha_rld learning disabled,
                   #    shrinkiage = 0.1, VB window = 500, thinning = 50, SGD approx =5, anchor = 50
                   #good results (=all CPs) for alpha_rld= 0.15 + pow div without learning, param alpha = 0.35 without learning,
                   #    int = 1000, shrinkage = 0.1
                   #good results with MSE = 0.25 for alpha_rld= 0.15 + pow div without learning, param alpha = 0.425 without learning,
                   #    int = 1000, shrinkage = 0.1 bust still 2 CPs that shouldn't be there
                   #BEST SO FAR:
                   #good results with MSE = 0.267 for alpha_rld= 0.15 + pow div without learning, param alpha = 0.55 without learning,
                   #    int = 1000, shrinkage = 0.1 bust still 2 CPs that shouldn't be there
                   #okay results with MSE = 0.35 for alpha_rld= 0.15 + pow div without learning, param alpha = 0.75 without learning,
                   #    int = 1000, shrinkage = 0.1 bust still 2 CPs that shouldn't be there, but missing a CPat 1500
                   #too robust results with MSE = 0.35 for alpha_rld= 0.15 + pow div without learning, param alpha = 0.7 without learning,
                   #    int = 1000, shrinkage = 0.05 missin 4 + one outlier
                   #BEST SO FAR:
                   #good results with MSE = 0.27 for alpha_rld= 0.15 + pow div without learning, param alpha = 0.6 without learning,
                   #    int = 1000, shrinkage = 0.1 bust still 1 CP that shouldn't be there
                   #NOTE: For shrinkage = 0.5, we get good results if alpha_param = 1.25, but still could be better!
                   #NOTE: We get a good result for shrink = 0.5, DPD with alpha param = 0.8 and rld = 0.1 (cp int = 100). Try
                   #        with KL rld distro, one CP missing
                   #coming very close with everything as before and alpha param 0.75, 0.725 (still 1-2 cps missing though)
                   #one too many for 0.65; one missing with 0.6
                   #DEBUG: Even with shrinkage = 0.1, we get false alarms if we put b=100, a =1! (cp intensity = 10, but still 
                   #        get them for 100, too). Also with 50.
                   #intensity seems to impact KL much more than ROBUST
                   #for int = 25, shrink = 0.1, alpha_param = 0.6, alpha_rld = 0.08, we get something which only misses one CP! 
                   #    It also does nog fall for any outlier...
                   #slightly too many CPs for ap= 0.65, arld = 0.1, CP int = 25
                   #best so far also works with shrinkg = 0.15 and cp ind = 50. Actually works better than KL (ito MsE)
                   #REALLY GOOD: alpha_param = 0.55, alpha_rld = 0.15, int = 50, shr = 0.25
                   #WE GET ALL BUT ONE WITH
                   #rld model is power_divergence
#                    RLD alpha is 0.15
#                    param inf uses DPD
#                    param alpha is 0.55
#                    intensity is 50
#                    shrinkage is 0.35 (similar with 0.5!)
#                    alpha param learning: False
#                  #rld model is kullback_leibler
#                    RLD alpha is 0.15
#                    param inf uses DPD
#                    param alpha is 0.4
#                    intensity is 500
#                    shrinkage is 1
#                    alpha param learning: False  
                   #with alpha rld = 0.25 gets worse, but does not get that weird off last one
                   
#FROM OFFICE MACHINE:
#really good, with two outlier-segments as CP + one CP missed
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.5 or 0.45
#intensity is 500
#shrinkage is 1
#alpha param learning: None
  
#Pretty good, but we get two outliers we don't want + one outlier missed!                 
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1.5
#intensity is 500
#shrinkage is 0.5
#alpha param learning: None
#failed optimizations: 0.0
                   
#Surprisingly good! Not missing any CP but one,no outliers                   
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1.5
#intensity is 100
#shrinkage is 0.05
#alpha param learning: None

#lots of outliers here (shrinkage is difference)                 
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1.5
#intensity is 250
#shrinkage is 0.5
#alpha param learning: None
#failed optimizations: 0.0
                   
#Gets all CPs, gets additional outliers too                   
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1.5
#intensity is 500
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.0
                   
#STILL MISSING ONE CP
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1.5
#intensity is 500
#shrinkage is 0.575
#alpha param learning: None
#failed optimizations: 0.0      
     
#No missing CPs, but too many outliers!              
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1.5
#intensity is 300
#shrinkage is 0.6
#alpha param learning: None
#failed optimizations: 0.0      

#################
# NON NORMALIZED           
################
#For KL version, works adequately for shrinkage = 0.1, horribly for shrink = 1, overdetects >4 outliers with 0.25

#works, but we miss 4 CPs and have one outlier. a=1, b=10000; use prior mean for beta prior
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.025
#intensity is 100
#shrinkage is 0.1
                   
#Only missed 4 CPs again, have an outlier
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.0001
#intensity is 100
#shrinkage is 0.1
                   
#missed the usual CPs again (around 2500)
#MSE is nan
#NLL is 9.99333329916
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.005
#intensity is 100
#shrinkage is 0.1

#Discovers a few at beginning and then goes haywire, b ->7124
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 5e-06
#intensity is 100
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.7001174398120963

#Gets a few in the middle, lets b -> 34320
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.025
#intensity is 100
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.13050612141865456

#Gets too many until the last real CP, b->4063     [likely numerical issue at end]          
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 1e-05
#intensity is 100
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.7049918166939444

#Gets way too many CPs, b->4355             
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 5e-05
#intensity is 100
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.6780385043010123
  
#Again, way too many CPs except for that one place where we should have them. 
                   #thinly spreak, too.
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.0001
#intensity is 100
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.6461920529801325

#This is close to perfect, misses 3 CP around 2500, no outliers               
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.01
#intensity is 100
#shrinkage is 1
#alpha param learning: None
#failed optimizations: 0.275942706810874

#Again, close to perfect, misses 2 `CP around 2500, no outliers 
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.005
#intensity is 100
#shrinkage is 1, 0.15, 0.25
#alpha param learning: None
#failed optimizations: 0.38465936160076225

#only missing one CP
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.005
#intensity is 100
#shrinkage is 0.25
#alpha param learning: None
#failed optimizations: 0.35795089343103226
                   
#misses loads of CPsrld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.003, 0.005 => shrinkage seems to have an effect!
#intensity is 100
#shrinkage is 10
#alpha param learning: None
#failed optimizations: 0.46395063283168564    

#misses 2 CPs, finds one outlier as CP
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.005
#intensity is 100
#shrinkage is 2
#alpha param learning: None
#failed optimizations: 0.39001400409350423

#same old: 2 CPs missing
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.003
#intensity is 100
#shrinkage is 0.25
#alpha param learning: None
#failed optimizations: 0.38986285875865967
                   
#same old: 2 CPS MISSING:
#rld model is kullback_leibler
#RLD alpha is 0.15
#param inf uses DPD
#param alpha is 0.005
#intensity is 100
#shrinkage is 0.35
#alpha param learning: None
#failed optimizations: 0.3728862973760933

#next things to test: alpha_param around 0.001-0.005, try to get that one/two more (shr = 0.25: only one missing)
#home machine:
#all for cp_int = 100
#1: replicate 0.005, 0.25 missing 2
#2: try       0.003, 0.25 missing 2, add 2
#3: try       0.005, 0.2 missing 2
#4: try       0.003, 0.2 missing 2
#office machine
#L: try       0.005, 0.15 missing 2
#C: try       0.003, 0.15 missing 2
#R: try       0.001, 0.25 missing 2, add 2-3   
                   
#Try next: Increas CP prob with alpha-param around 0.001-0.005, shr=0.25
#1: 0.005, 0.25, 25, missing 2
#2: 0.005, 0.25, 50, missing 2
#3: 0.005, 0.25, 75, missing 2
#4: 0.003, 0.2, 25, missing 2, add 1
#L: 0.003, 0.2, 50, missing 2
#C: 0.003, 0.2, 75, missing 2, add 2
#R: 0.001, 0.2, 100, missing 2-3, add 2-3
                   
#CP intensity does not have an impact apparently (only makes RLD MORE FLIPPANT!)
#so set to 100 again.
#1: 0.005, 0.1, missing 2, one off a little
#2: 0.005, 1, missing 2
#3: 0.005, 2, missing 2
#4: 0.003, 1  missing 4
#L: 0.003, 2, missing 4, add 2-3
#C: 0.01, 10, missing 5
#R: 0.01, 5, missing 2
                   
#Try to shrink even less...
#1: 0.005, 5, missing 2 CPs
#2: 0.005, 10 missing a lot (only finds 3)
#3: 0.005, 25, missing: a lot (only find 2)
#4: 0.005, 50, missing: a lot (NO CPS!)
#L: 0.005, 0.25, threshold = 25, missing: 2
#C: 0.005, 0.25, threshold = 50
#R: 0.01, 2, threshold = 75 (normal)   
                   
#shrinkage alone does not pull the wagon, apparently/maybe I need to shrink more, not less!
                   
#R: 0.005, 100  also misses 
#C: 0.005, 0.75  also misses 2 gets 2 add         
#L: 0.01, 1 also misses 2                   
                   
#shrinkage around    
#1: 0.005, 0.1 missing 2 cps
#2: 0.005, 0.05 missing 2-3 cps
#3: 0.005, 0.01, missing 3 cps
#4: 0.005, 0.005, missing 6 cps
#L: 0.005, 0.001, missing 4-5 cps try 0.005, 0.085
#C: 0.005, 0.0005 misses CPs try 0.005, 0.075
#R: 0.005, 0.0001, strange: RLD seems right, but no CPs declared! Now try 0.025
#LG: 0.001, 0.1, not enough CPs, try 0.005 + 0.25 + param learning overdetects
#CG 0.001, 0.05 missing 4-5, try 0.005 + 0.1 + param learning overdetects
#RG 0.001, 0.01   not enough CPs try 0.005 + 0.01 + param learning overdetects
                   
#LG: 0.004, 0.1, still missing >2 cps
#CG 0.004, 0.25 still missing 2 cps
#RG 0.004, 0.5   still missing 2 cps 
#also not working: 0.005, 0.075 & 0.005

#1: 0.0075, 1, missing 4
#2: 0.0075, 0.5 missing 2
#3: 0.0075, 0.25 missing 3
#4: 0.0075, 0.25 missing 3
#L: 0,00075, 0.05 missing 4, 4 add
#C: 0.00075, 0.25 missing 3, many add
#R: 0.01, 0.05 missing all but 2
#GL: 0.1, 1 missing 5
#GC: 0.002, 0.1 missing 4
#GR: 0.00075, 0.1 missing 5 many adds      

#WITH SGD NOT SVRG              
#1: 0.005, b=pow(10,2), 0.25 only finds 2
#2: 0.005, b=pow(10,3), 0.25 ?
#3: 0.005, b=pow(10,4), 0.35 missing 3
#4: 0.005, b=pow(10,4), 0.25 ?
#GL: 0.005, 0.1 misses 2
#GC: 0.005, 0.5 misses 4
#GR: 0.005, 1 misses 2
#L: 0.005, 2 misses 4
#C: 0.005, 5 misses 8
#R: 0.005, 0.05  misses 4                 
                   
#GL: 0.005, 0.1, b=pow(10,3) missing 5
#GC: 0.005, 0.1, b=pow(10,2) missing 7
#GR: 0.005, 1, b=pow(10,3) missing 5
#1: 0.005, b=2*pow(10,4)    missing 3               
#2: 0.005, b=0.5*pow(10,4) missing 4     
#3: 0.005, b=3*pow(10,4) missing 2
#4: 0.005, b=4*pow(10,4) missing 2, 2 add                
                   
#L: 0.001, 0.25 missing only 2
#C: 0.001, 0.1 missing 4+
#R: 0.001, 0.01 missing 4+                 
#GL: 0.005, 0.1, b=2*pow(10,4) 3 missing
#GC: 0.005, 0.1, b=3*pow(10,4) 2 missing
#GR: 0.005, 0.1, b=5*pow(10,4) 2-3 missing 
#1: 0.005, 0.25, b=7*pow(10,4) 2 missing
#2: 0.005, 0.25, b=pow(10,5)   2 missing + 1 
#3: 0.005, 0.1, b=7*pow(10,4)  3 missing
#4: 0.005, 0.1, b=pow(10,5)    2 missing
#GL: 0.005, 0.5, b=2*pow(10,4) 2 missing
#GC: 0.005, 0.5, b=3*pow(10,4) 3 missing
#GR: 0.005, 0.5, b=5*pow(10,4) 2 missing   
#L: 0.001, 0.25 b=pow(10,4)*2, a=100 numerical issue, finds lots at start & then none
#C: 0.001, 0.25 b=pow(10,4)*3, a=100 similar to L, maybe probs are too small?!
#R: 0.001, 0.25 b=pow(10,4)*5, a=100 similar again   
                   
#1: 0.005, 0.25, b=10*pow(10,4) a=10 (very few CPs, only 3)
#2: 0.005, 0.25, b=10*pow(10,4) a=100  (not a single CP)
#3: 0.005, 0.25, b=10*pow(10,4) a-1000  numerically instable/overdetects after first CP
#4: 0.005, 0.25, b=10*pow(10,4) a=10000   (lots of numerical issues during exec)
#GL: 0.005, 1, b=2*pow(10,4) 3 missing
#GC: 0.005, 1, b=3*pow(10,4) 2 missing
#GR: 0.005, 1, b=5*pow(10,4) 3 missing    
#AL: 0.005, 1, b=0.5*pow(10,4) 4 missing
#AC: 0.0075, 1, b=0.5*pow(10,4) 4 missing 
#AR: 0.01, 1, b=0.5*pow(10,4) 3 missing
#L: 0.005, 0.25, b=pow(10,4), a=2
#C: 0.005, 0.25, b=pow(10,4), a=5
#R: 0.005, 0.25, b=pow(10,4), a=7.5  
#JL: 0.005, 0.25, b=pow(10,5), a=2 2 missing
#JC: 0.005, 0.25, b=pow(10,5), a=5 4 missing
#JR:  0.005, 0.25, b=pow(10,5), a=7.5 2 missing     
#YL: 0.005, 0.01, b=pow(10,4), a=1 4 missing
#YC: 0.005, 0.005, b=pow(10,4), a=1 4missing
#YR: 0.005, 0.001, b=pow(10,4), a=1 4 missing
                   
                   
#1: 0.005, 0.25, b=2, a=pow(10,4) not working at all, way too many CPs
#2: 0.005, 0.25, b=pow(10,4), a = 5, int = 25 lots missing 
#3: 0.005, 0.25, b=pow(10,4), a = 5, int = 50 5 missing
#4: 0.005, 0.25, int = 50 4 missing
#GL: 0.005, 0.01, b=pow(10,4), a=5 misses 3-4
#GC: 0.005, 0.005, b=pow(10,4), a=5 misses 4+
#GR: 0.005, 0.001, b=pow(10,4), a=5  misses 4+                  
#AL: 0.01, 10, b=0.25*pow(10,4) misses 5+
#AC: 0.01, 1, b=0.25*pow(10,4) misses 5
#AR: 0.01, 1, b=pow(10,3) misses 5+
#JL: 0.001, 0.2, b=pow(10,4), a=2  miss 3 + 1
#JC: 0.001, 0.2, b=5*pow(10,5), a=2 miss 2 add 2
#JR:  0.001, 0.2, b=10*pow(10,5), a=2 miss 3 add 2
#YL: 0.15, 1, b=pow(10,4), a=1 3CPs found [dominated by opt. failure]
#YC: 0.15, 10, b=pow(10,4), a=1 2CPs found
#YR: 0.15, 100, b=pow(10,4), a=1 1CP found

#1: 0.1, 0.25 missing 5cps
#2: 0.005, 0.25, b=0.1*pow(10,4), a=2 missing 5 cps   
#3: 0.1, 0.5   missing 5 cps no failed opt
#4: 0.005, 0.25, b=0.05*pow(10,4), a=2  missing 3 cps
#L: 0.1, 1 only finds 2 cps
#C: 0.1, 10 only finds 3 cps
#R: 0.1, 0.25, b=pow(10,5)   misses only 4 cps, b -> larger!
#GL: 0.005, 0.25, b=0.75*pow(10,4), a=2 number CPs: 5
#GC: 0.005, 0.25, b=0.5*pow(10,4), a=2 number CPs: 5
#GR: 0.005, 0.25, b=0.25*pow(10,4), a=2  number CPs: 6
#AL: 0.05, 0.25, b=1*pow(10,4) 
#AC: 0.05, 0.25, b=5*pow(10,4)
#AR: 0.05, 0.25, b=10*pow(10,4)
#YL: 0.05, 0.1, b=pow(10,4), 
#YC: 0.05, 0.1, b=5*pow(10,4), 
#YR: 0.05, 0.1, b=10*pow(10,4),   
 
                   
#1: 0.1, 0.5 b=10*pow(10,4), misses 5
#2: 
#3: 0.1  0.5 b=100*pow(10,4)  only finds 3
#4:
#L: 0.05, 0.25, b=100*pow(10,4) misses a few
#C: 0.05, 0.25, b=500*pow(10,4) PERFECT FIT saved to my office machine
#R: 0.05, 0.25, b=1000*pow(10,4)  PERFECT FIT saved to my office machine
#GL: 0.05, 0.25, b=pow(10,4)*pow(10,4), EXCELLENTE! Catches all CPs + 2 outliers SAVE to guilio machine
#GC: 0.05, 0.25, b=5*pow(10,4)*pow(10,4), misses 2
#GR: 0.05, 0.25, b=10*pow(10,4)*pow(10,4),    misses 3           
#JL: --
#JC: 0.05, 0.5, b=pow(10,10) 2 CPs no failed opts
#JR: 0.05, 0.5, b=5*pow(10,10) no CP no failed opts
                   
                   
#1: 0.05, 0.5 b=2*pow(10,4),a=5
#2: 0.05, 1, b=2*pow(10,4), a = 5
#3: 0.05 10 b=2*pow(10,4)  a=5
#4: 0.05 100 b=2*pow(10,4)  a=5           

#Idea: Maybe change SVRG to SGD again for r<W and run some again to see if same! 
#Idea: larger b, smaller shrinkage, more robust?      
#Idea/Obs: For smaller alpha, we are less stable in opt! Might be numerical tradeoff
#maybe with a = 5?
 
    
                   
#KL:  setting: a=1, =pow(10,4) [pow(10,3)*5?], shrink = 0.25, cp_int = 1200, prior_mean = mean
#KL:   MSE is 27540236.505 [[ 3212873.03822018]]
#      MAE is 3249.52302253 [[ 130.20212777]]
#      NLL is 9.76019667165 [[ 3212873.03822018]]                   
                   
#DPD: a=1, b=pow(10,7), shrik=0.25, cp_int = 100, prior-mean = mean, rld = 0.0001
#MSE is 32803959.0593 [[ 5619695.43520509]]
#MAE is 3189.88222572 [[ 150.30294243]]
#NLL is 9.76801174751 [[ 5619695.43520509]]     
                   
                   
a, b = 1, pow(10,7)
alpha_param = 0.05 #pow(10,-5)
alpha_rld = 0.0001 #0.08 slightly too small
rld = "power_divergence" #power_divergence kullback_leibler
rld_learning = True
param_learning = "individual" #"individual" #"individual" #"individual" #"individual"

prior_mean_scale, prior_var_scale = np.mean(data), 0.25 #np.sqrt(np.var(data))
cp_intensity = 100
np.random.seed(999)

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
                 nbh_sequence=None,
                 restriction_sequence = None,
                 hyperparameter_optimization = "online", #"online", #"online", #"online",
                 VB_window_size = VB_window_size,
                 full_opt_thinning = full_opt_thinning,
                 SGD_batch_size = SGD_approx_goodness,
                 anchor_batch_size_SCSG = anchor_approx_goodness_SCSG,
                 anchor_batch_size_SVRG = anchor_approx_goodness_SVRG,
                 first_full_opt = first_opt
            )]
if mode == "KL" or mode == "both":
    model_universe = model_universe + [BVARNIG(
                    prior_a = a,
                    prior_b = b,
                    S1 = S1,
                    S2 = S2,
                    prior_mean_scale = prior_mean_scale,
                    prior_var_scale = prior_var_scale,
                    nbh_sequence = None,
                    restriction_sequence = None,
                    hyperparameter_optimization = "online"#"online"
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
        threshold = 50,
        notifications = 100,
        save_performance_indicators = True,
        generalized_bayes_rld = rld, #"power_divergence", #"kullback_leibler", #"power_divergence" , #"power_divergence", #"kullback_leibler",
        alpha_param_learning =  param_learning,#"together", #"individual", #"individual", #"individual", #"individual", #"together",
        alpha_param  = alpha_param, 
        alpha_param_opt_t = 100, #, #) #,
        alpha_rld = alpha_rld, #pow(10, -5), #0.25,
        alpha_rld_learning = rld_learning, #"power_divergence",
        #alpha_rld = 0.25, #0.00000005,pow(10,-12)
        #alpha_rld_learning=True,
        loss_der_rld_learning="absolute_loss"
        )
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
if detector.generalized_bayes_rld == "power_divergence":
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
EvT.plot_raw_TS(data.reshape(T,1), indices = [0], xlab = None, 
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
print("failed optimizations:", detector.model_universe[0].failed_opt_count/
          detector.model_universe[0].opt_count)

#results_path = baseline_working_directory + "//results_KL_int=100_b=pow(10,4)_a=1_shrink=025.txt" 
#EvT.store_results_to_HD(results_path)
#fig.savefig(baseline_working_directory + "//well_log_KL_.pdf",
#            format = "pdf", dpi = 800)
#    
#    
    

height_ratio =[10,10]
custom_colors = ["blue", "purple"] 
fig, ax_array = plt.subplots(2, figsize=(8,5), sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio})
plt.subplots_adjust(hspace = .1, left = None, bottom = None,
                    right = None, top = None)
ylabel_coords = [-0.065, 0.5]

ax_array[1].plot(np.linspace(100,T,T-101,dtype=int), 
        detector.model_universe[0].alpha_param_list,
        linewidth=2)
ax_array[1].set_ylabel(r'$\beta_{p}$', fontsize = 20)
ax_array[0].plot(np.linspace(1,83,83,dtype=int)*int(T/83), 
        np.array(detector.alpha_list),
        linewidth=2)
ax_array[0].set_ylabel(r'$\beta_{rld}$', fontsize = 20)

fig.savefig(baseline_working_directory + "//beta_trajectory.pdf",
            format = "pdf", dpi = 800)

#results_path = baseline_working_directory + "//results_DPD_fullOpt.txt" 
#EvT.store_results_to_HD(results_path)

#NOTE: We need a way to use BVARNIG(DPD) for only fitting a constant!