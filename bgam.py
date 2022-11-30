#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:45:46 2022

@author: isabelgaron
"""

import numpy as np
import sys

sys.path.append('/Users/isabelgaron/anaconda3/envs/pgam/GAM_library')
sys.path.append('/Users/isabelgaron/anaconda3/envs/firefly/proj_files')
#from GAM_library.GAM_library import *#general_additive_model as heresthegam

import pandas as pd
import scipy.stats as sts
from post_processing import postprocess_results
import gam_data_handlers as gdh

from GAM_library import *
import matplotlib.pylab as plt
import statsmodels.api as sm
from bgam_data_handler import *

def mvmt_graph(trial, trial_num):

    plt.figure()
    x_rew_circle = 60 * np.sin(np.pi*2*np.linspace(0, 1, 100))
    y_rew_circle = 60 * np.cos(np.pi*2*np.linspace(0, 1, 100))

    plt.title(trial_profile(trial_num))

    sel_time = (timept[trial_num] >= t_flyON[trial_num]) & (
        timept[trial_num] <= t_stop[trial_num])
    plt.ylim(-40, 400)
    plt.xlim(-300, 300)
    plt.plot(x_monk[trial_num][sel_time], y_monk[trial_num][sel_time])
    '''plt.scatter(np.sin(((ang_path[trial_num]/2)*np.pi)/180)*(rad_path[trial_num]/2),
                np.cos(((ang_path[trial_num]/2)*np.pi)/180)*(rad_path[trial_num]/2))'''
    plt.scatter(x_fly[trial_num], y_fly[trial_num], color='r')
    plt.plot(x_rew_circle + x_fly[trial_num],
             y_rew_circle + y_fly[trial_num], 'r')
    plt.plot(path_x[trial_num], path_y[trial_num])
    plt.figure()
    plt.subplot(211)
    plt.title('Radial dist ' + trial_outcome)

    plt.plot(timept[trial_num], rad_target[trial_num])

    plt.axvspan(t_flyON[trial_num], t_flyOFF[trial_num],
                color="yellow", zorder=0)

    plt.vlines([t_move[trial_num], t_stop[trial_num]], min(
        rad_target[trial_num]), max(rad_target[trial_num]), 'r')
    plt.xlabel('time [sec]')
    plt.subplot(212)
    plt.title('Angular dist')
    plt.plot(timept[trial_num], ang_target[trial_num])

    plt.axvspan(t_flyON[trial_num], t_flyOFF[trial_num],
                color="yellow", zorder=0)

    plt.vlines([t_move[trial_num], t_stop[trial_num]], min(
        ang_target[trial_num]), max(ang_target[trial_num]), 'r')

    plt.xlabel('time [sec]')
    plt.tight_layout()

def actual_data(data, var_list, trial_ids):

    nobs = len(data[0])
    
    size_knots = 5
    order = 4
    y=data[0,:]
    #y=(~y.astype(bool)).astype(int) # inverting bool successfully inverts lines
    
    lis_funcs = []
    x_ranges = []
    xxs = []
    sm_handler = smooths_handler()
    for i in range(1, len(data)):

        xi_range = [np.min(data[i,:]),np.max(data[i,:])]
        x_ranges.append(xi_range)
        xx1 = np.linspace(xi_range[0], xi_range[1],100)
        xxs.append(xx1)
        knots1 = np.linspace(xi_range[0],xi_range[1],size_knots)

         #CAREFUL with - first name is for data?
        sm_handler.add_smooth(var_list[i-1], [data[i,:]], ord=order, knots=[knots1], knots_num=size_knots, perc_out_range=0.0,
                           trial_idx=trial_ids,is_cyclic=[False], lam=None,penalty_type='der',der=2)
    
    kernel_h_length = 20
    x_ranges.append([0,kernel_h_length])
    xx1 = np.linspace(0, kernel_h_length,20)
    xxs.append(xx1)
    sm_handler.add_smooth(var_list[9], [data[10,:]], is_temporal_kernel=True, ord=order, knots_num=5,
                          trial_idx=trial_ids,kernel_length=kernel_h_length,kernel_direction=1)
    '''kernel_h_length = 10
    x_ranges.append([0,kernel_h_length])
    xx1 = np.linspace(0, kernel_h_length,100)
    xxs.append(xx1)
    sm_handler.add_smooth(var_list[7], [data[8,:]], is_temporal_kernel=True, ord=order, knots_num=5,
                          trial_idx=trial_ids,kernel_length=kernel_h_length,kernel_direction=1)
    '''
    link = sm.genmod.families.links.logit()
    bernouFam = sm.genmod.families.family.Binomial(link=link)
    gam_model = general_additive_model(sm_handler,var_list,y,
                                           bernouFam,fisher_scoring=False)

    full,reduced = gam_model.fit_full_and_reduced(var_list,th_pval=0.001, max_iter=10 ** 4,#, tol=10 ** (-8),
                                  conv_criteria='gcv', 
                                  method='L-BFGS-B', #smooth_pen=[1] * 4, 
                                  gcv_sel_tol=10 ** (-13), use_dgcv=True, fit_initial_beta=True,pseudoR2_per_variable=True)
    print('===============================================')
    print("Num vars: "+str(len(reduced.variable_expl_variance)))
    for i in range(0,len(reduced.variable_expl_variance)):
        print("VAR " + str(i))
        
        print("Variable: " + str(reduced.variable_expl_variance[i]['variable']))
        print("Pseudo Rsq: " + str(reduced.variable_expl_variance[i]['pseudo-R2'])) 
        print("Variance expl: " +str(reduced.variable_expl_variance[i]['var_expl']))
        
    
    plt.close('all')
    plt.figure()
    fig, axs = plt.subplots(1,10, sharey=True)
    
    funcs = []
    
    for j in range(0, len(var_list)):
        f1y, f1y_min, f1y_max, la1 = full.smooth_compute([xxs[j]], var_list[j])
        #plt.subplot(1, len(var_list), j+1)
        axs[j].set_title(var_list[j])
        axs[j].plot(xxs[j], f1y, 'k-', label='model')
        axs[j].fill_between(xxs[j], f1y_min,f1y_max)
        funcs.append(f1y)
        
        
    return funcs, x_ranges

def asses_data(functions, test_data, bounds):# add what to include
    
    val = np.zeros((len(test_data), len(test_data[0])))
    for i in [1, 2, 3, 4,5,6,7,8,9]:
        xx1 = np.linspace(bounds[i-1][0], bounds[i-1][1], 100)
        compare = np.tile(xx1, (len(test_data[i]),1)).T
        inds = np.argmin(np.abs(compare - test_data[i]), axis=0)
        func_eval = functions[i-1][inds]
        val[i,:] = func_eval
    if test_data[10,0] == 0:
        print(functions[9])
        func_eval = functions[9][:len(test_data)]
        print(func_eval)
        val[10,:] += func_eval
    '''if len(test_data[9]) > 17:
        val[9,0:17] = functions[8][0:17]
    else:
        val[9,:] = functions[8][0:len(test_data[9])]'''
    #val[10,:] = func_eval
    return val

def check_data(trial, funcs, bounds):
    conf_mat = np.zeros((2, 2))

    plt.figure()

    for i in range(0, 44):

        downsample_s = construct_bgam_downsample_backup(m53s100, m53s100.successes[300 + i], .8, 17)
        res1 = asses_data(funcs, downsample_s, bounds)
        # time warp so compoletely aligned
        downsample_f = construct_bgam_downsample_backup(m53s100, m53s100.failures[300 + i], .8, 17)
        res2 = asses_data(funcs, downsample_f, bounds)

        x1 = np.arange(0, len(res1[1])) * 20 / len(res1[1])
        x2 = np.arange(0, len(res2[1])) * 20 / len(res2[1])

        sum1 = np.sum(res1, axis=0)
        sum2 = np.sum(res2, axis=0)

        if sum1[-1] > 0:
            conf_mat[0, 0] += 1
        else:
            conf_mat[0, 1] += 1
        if sum2[-1] < 0:
            conf_mat[1, 1] += 1
        else:
            conf_mat[1, 0] += 1
        '''sum1 =  np.sum(res1, axis=0)
        sum2 = np.sum(res2, axis=0)
        step_s = len(sum1-1)/20
        print(step_s)
        for j in range(0, 20):
            print(int(np.around(j*step_s)))
            print(sum1[int(np.around(j*step_s))])
            if sum1[int(np.around(j*step_s))] >0:
                conf_mat[0,0,j] +=1
            else:
                conf_mat[0,1,j] +=1
        for k in range(0,20):        
            if sum2[int(np.around(j*step_s))] <0:
                conf_mat[1,0,k] +=1
            else:
                conf_mat[1,1,k] +=1'''

        plt.plot(x1, np.sum(res1, axis=0), color="blue")
        plt.plot(x2, np.sum(res2, axis=0), color="grey")
        plt.axhline(0)
        ax = plt.gca()
        ax.set_xticklabels(np.around(np.linspace(0, 100, 10)))
        plt.title("Initial " + str(proportion_considered) + "% Considered")
        plt.xlabel("Proportion of trial")

        '''plt.subplots(1,3)
        plt.subplot(131)
        plt.plot(np.sum(res1, axis=0), color="blue")
        plt.plot(np.sum(res2, axis=0), color="grey")
        plt.subplot(132)
        plt.plot(downsample_f[1], color="lightgrey", label = "t_since_fly")
        plt.plot(downsample_s[1], color="lightblue", label = "st_since_fly")
        plt.plot(downsample_f[2], color="grey", label = "t_since_move")
        plt.plot(downsample_s[2], color="blue", label = "st_since_move")
        plt.plot(downsample_f[3], color="grey", label = "s_density")
        plt.plot(downsample_s[3], color="blue", label = "f_density")
        plt.plot(downsample_f[4], color="darkgrey", label = "distance")
        plt.plot(downsample_s[4], color="darkblue", label = "sdistance")
        plt.plot(downsample_f[5], color="darkgrey", label = "linvel")
        plt.plot(downsample_s[5], color="darkblue", label = "slinvel")
        plt.plot(downsample_f[6], color="darkgrey", label = "angvel")
        plt.plot(downsample_s[6], color="darkblue", label = "sangvel")
        plt.legend()
        plt.subplot(133)
        plt.plot(res2[1], color="lightgrey", label = "t_since_fly")
        plt.plot(res1[1], color="lightblue", label = "st_since_fly")
        plt.plot(res2[2], color="grey", label = "t_since_move")
        plt.plot(res1[2], color="blue", label = "st_since_move")
        plt.plot(res2[3], color="grey", label = "s_density")
        plt.plot(res1[3], color="blue", label = "f_density")
        plt.plot(res2[4], color="darkgrey", label = "distance")
        plt.plot(res1[4], color="darkblue", label = "sdistance")
        plt.plot(res2[5], color="darkgrey", label = "linvel")
        plt.plot(res1[5], color="darkblue", label = "slinvel")
        plt.plot(res2[6], color="darkgrey", label = "angvel")
        plt.plot(res1[6], color="darkblue", label = "sangvel")'''
        # plt.legend()
    print("Confiusion matrix")
    print(conf_mat)
    '''plt.figure()
    plt.plot(conf_mat[0,0,:], label = "Successes correct")
    plt.plot(conf_mat[1,0,:], label = "Failures correct")
    plt.plot(conf_mat[0,1,:], label = "Successes wrong")
    plt.plot(conf_mat[1,1,:], label = "Failures wrong")
    plt.xlabel("% of trial length")
    plt.ylabel("# trials")
    plt.legend()'''
                       
if __name__ == '__main__':
    samples = np.random.gamma(2,1, 1000)
    plt.figure()
    plt.hist(samples, bins = 50)
    plt.show()
    print("========is gammmmmmm========")
    plt.figure()
    m53s100 = Trial('/Users/isabelgaron/anaconda3/envs/firefly/proj_files/monkey/m53s114.mat')
    np.random.seed(5)
    proportion_considered = .75
    smooth_over = 33 #ms #TODO finish this
    
    n_train = 300
    n_test = 30
    # check that this total is possible
    
    final = []
    
    num_nobs = 0
    num_s = 0
    num_f=0
    indices = []
    # TODO - leave one out check on each variable, labeling axis, at different input windows
    # also confusion matrices, also chekc when reward occurs versus when motion dtops
    # 200, .75, 2786 trials
    #### randomize
    successes = np.random.shuffle(m53s100.successes)
    failures = np.random.shuffle(m53s100.successes)
    
    successes_train = successes[:n_train]
    successes_test = successes[n_train:n_test]
    
    failures_train = successes[:n_train]
    failures_test = successes[n_train:n_test]

    for i in range(1, n_train):#.75, .9
    #cheating 1, .9, works with .75, 1, and 1, .75
    #starts working qt 60% of total trial, as low as .3, dcsles evenly, dont need to rerun
    # question is how early separation emerges
        #print("index: " + str(i))
        samp_s = construct_bgam_downsample_backup(m53s100, m53s100.successes[i], proportion_considered, smooth_over)
        samp_f = construct_bgam_downsample_backup(m53s100, m53s100.failures[i], proportion_considered, smooth_over)
        if samp_f is not None:
            if samp_s is not None:
                final.append(samp_s)
                final.append(samp_f)
                num_nobs+=len(samp_s[0])
                num_nobs+=len(samp_f[0])
                #print(len(samp_s[0]))
                #print(len(samp_f[0]))
                num_s+=len(samp_s[0])
                num_f+=len(samp_f[0])
                indices= np.concatenate((indices, np.repeat(m53s100.successes[i], len(samp_s[0]))))
                indices= np.concatenate((indices, np.repeat(m53s100.failures[i], len(samp_f[0]))))
    indices = np.array(indices)
    print("Failures: " + str(num_f))
    print("Successes: " + str(num_s))
    data = np.hstack(final)
    
    funcs,bounds = actual_data(data, ['t_since_firefly', 't_since_move', 'density', 'distance','lin_vel','ang_vel','lin_acc','ang_acc', 'prev_dist','prev_perf'], indices)
    #,'prev_perf','prev_dist']


    conf_mat = np.zeros((2, 2))
    plt.figure()
    # discriminability metric = this averaged over 100 timesteps
    #    compare = np.tile(xx1, (len(test_data[i]),1)).T
    #         inds = np.argmin(np.abs(compare - test_data[i]), axis=0)
    #         func_eval = functions[i-1][inds]
    num_test = 20
    s_mat = np.zeros((num_test, 100))
    f_mat = np.zeros((num_test, 100))

    plt.figure()
    fig, axs = plt.subplots(3,1, sharey=True, sharex=True)
    
    for i in range(0, num_test):
        downsample_s = construct_bgam_downsample_backup(m53s100, m53s100.successes[n_train + i], .9, smooth_over)
        res1 = asses_data(funcs, downsample_s, bounds)
        # time warp so compoletely aligned
        downsample_f = construct_bgam_downsample_backup(m53s100, m53s100.failures[n_train + i], .9, smooth_over)
        res2 = asses_data(funcs, downsample_f, bounds)

        x1 = np.arange(0, len(res1[1])) * 100 / len(res1[1])
        x2 = np.arange(0, len(res2[1])) * 100 / len(res2[1])

        sum1 = np.sum(res1, axis=0)
        sum2 = np.sum(res2, axis=0)
        

        for j in range(0, 100):
            s_mat[i, j] = sum1[np.argmin(abs(x1 - j))]
            f_mat[i,j] = sum2[np.argmin(abs(x2-j))]


        if sum1[-1] > 0:
            conf_mat[0, 0] += 1
        else:
            conf_mat[0, 1] += 1
        if sum2[-1] < 0:
            conf_mat[1, 1] += 1
        else:
            conf_mat[1, 0] += 1
        ##### nah or just normal link???? logit
        link = sm.genmod.families.links.logit()
        bernouFam = sm.genmod.families.family.Binomial(link=link)
        axs[0].plot(x1,bernouFam.link.inverse(np.sum(res1, axis=0)), color="blue")
        axs[1].plot(x2,bernouFam.link.inverse(np.sum(res2, axis=0)), color="grey")
        
    # try with 39 and 51

    #39
    #giveups = [1230, 1150, 1046, 880, 778, 730, 703, 675, 543, 140, 134, 88]
    #51 - 
    #giveups = [1681,1161,1675, 1304, 1299, 1226,  1111, 1100, 753, 743,  641, 610, 568, 558, 219,722, 46 ]
    #105
    #giveups = [714, 692, 691, 675, 658, 646, 634, 535, 496, 464, 428, 427, 417, 394, 339, 276, 224, 212, 190, 189, 120, 118, 113]
    #114 - 675, 591,513, 509, 498,280, 278 train then giveup test
    giveups = [741,712, 693, 690, 686, 675, 599, 591, 585, 584, 582, 513, 509, 498, 480, 280, 278, 258, 242] # 19 tested, 7 trained on, 15->failure predicted from outset, 
    #100
    #giveups = [430,420, 385, 317,299, 296, 293, 280, 213, 180, 171, 165, 158,135,122, 82, 78, 76, 70, 48, 16]
    # not good, and seperatbility regulrly around 85
    #123
    #giveups_123 = [758,756,710, 691, 599, 598, 556, 536, 413, 400, 336, 333, 293, 220, 159, 54, 8, 6]
    #giveups_124 = [656,555,554,549, 504, 503, 420, 414, 412, 404, 397, 335, 328, 305, 280, 258, 249, 231, 135, 131, 126,2]

    for i in giveups:
        downsample_s = construct_bgam_downsample_backup(m53s100, i, .8, smooth_over)
        res1 = asses_data(funcs, downsample_s, bounds)

        x1 = np.arange(0, len(res1[1])) * 100 / len(res1[1])

        sum1 = np.sum(res1, axis=0)

        ##### nah or just normal link???? logit
        link = sm.genmod.families.links.logit()
        bernouFam = sm.genmod.families.family.Binomial(link=link)
        axs[2].plot(x1,bernouFam.link.inverse(np.sum(res1, axis=0)), color = 'orange')
   
    print(np.std(s_mat, axis=0))
    print(np.std(f_mat, axis=0))
    # actually d'
    delta = abs(bernouFam.link.inverse(np.average(s_mat, axis=0))-bernouFam.link.inverse(np.average(f_mat, axis=0)))\
                  /bernouFam.link.inverse(np.average(np.std(f_mat, axis=0)))
    
    
    axs[0].set_title("Successes 114 -  Initial " + str(proportion_considered) + "% Considered, d' = " + str(np.average(delta)))
    axs[1].set_ylabel("Probability of Success")
    axs[1].set_title("Failures")
    axs[2].set_title("Give ups")
    #axs[2].set_xticklabels(np.around(np.linspace(0, 80)))
    axs[2].set_xlabel('Proportion of Trial')
    plt.legend()
    
    print(s_mat)
    print("================================================")
    print("d'") # replace with bayes discriminability
    print(np.average(delta))
    #print(np.sum(bernouFam.link.inverse(np.average(s_mat, axis=0)-np.std(s_mat, axis=0))-bernouFam.link.inverse(np.average(f_mat, axis=0)+np.std(f_mat, axis=0))))
    
    axs[0].plot(delta, color="red")
    axs[1].plot(delta, color="red")
    '''plt.plot(bernouFam.link.inverse(np.median(s_mat, axis=0)), color="coral")
    plt.plot(bernouFam.link.inverse(np.median(f_mat, axis=0)), color="coral")
    plt.plot(bernouFam.link.inverse(np.average(s_mat, axis=0)-np.std(s_mat, axis=0)), color="pink")
    plt.plot(bernouFam.link.inverse(np.average(f_mat, axis=0)+np.std(f_mat, axis=0)), color="pink")
    '''
    print("Confiusion matrix")
    print(conf_mat)
    plt.show()
    
    
    
