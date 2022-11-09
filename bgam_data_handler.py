#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:32:19 2022

@author: isabelgaron
"""
import sys
path_to_ff_lib = '/Users/isabelgaron/anaconda3/envs/firefly/proj_files/firefly_utils'
sys.path.append(path_to_ff_lib)
import math
import csv
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from firefly_utils.data_handler import *
from scipy import stats
import pandas as pd

import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix

# TODO
# Define trial class
# filter to target trials
# set up subsampler
# construct delta ts
# return df that can be used in bgam 
#
class Trial(object):   
    def __init__(self, base_file):
  
        behav_stat_key = 'behv_stats'
        spike_key = 'units'
        behav_dat_key = 'trials_behv'
        lfp_key = None
        
        # duration of the pre-trial interval in sec
        pre_trial_dur = 0.2
        post_trial_dur = 0.2
        
        
        # list of sesssion for which the eye tracking was on left eye
        use_left_eye = ['53s48']
        
        dat = loadmat(base_file)
        exp_data = data_handler(dat, behav_dat_key, spike_key, None, behav_stat_key, pre_trial_dur=pre_trial_dur,
                                post_trial_dur=post_trial_dur,
                                extract_lfp_phase=False,
                                use_eye=use_left_eye, extract_fly_and_monkey_xy=True, skip_not_ok=False)
        
        
        #behavioral vars : dictionaries, with keys the trial number
        behav = exp_data.behav
        
        # sampling time in seconds (0sec = target onset)
        
        self.timept = behav.time_stamps
        self.lin_vel = behav.continuous.rad_vel  # cm/sec, linear velocity
        self.ang_vel = behav.continuous.ang_vel  # deg/sec, "angular" velocity
        self.lin_acc = behav.continuous.rad_acc  # cm /sec^2, linear acceleration
        self.ang_acc = behav.continuous.ang_acc  # deg /sec^2, anggular acceleration
        # cm, rad dist to target (nan before target on)
        self.rad_target = behav.continuous.rad_target
        # deg, ang dist to target (nan before target on)
        self.ang_target = behav.continuous.ang_target
        
        # at the beginning of a trial the monkey is "teleported" in  the origin (0,0)
        # *_path will have the distance traveled from to origin (nan before target on)
        self.rad_path = behav.continuous.rad_path  # cm, lin dist from origin
        self.ang_path = behav.continuous.ang_path  # deg, ang dist from origin
        
        
        self.eye_hori = behav.continuous.eye_hori  # rad, horizontal eye displacement
        self.eye_vert = behav.continuous.eye_vert  # rad, vertical eye displacement
        
        # x and y positioin of the monkey (usee time points after target onset to avoid
        # artefacts due to the monkry being teleported to the origin)
        self.x_monk = behav.continuous.x_monk
        self.y_monk = behav.continuous.y_monk
        
        # x/y position of the firefly
        self.x_fly = behav.continuous.x_fly
        self.y_fly = behav.continuous.y_fly
        
        
        # EVENT VARS
        self.t_flyON = behav.events.t_targ  # tarrget onset (0sec)
        self.t_flyOFF = behav.events.t_flyOFF  # ff offset time in sec (0.3 sec is the std)
        self.t_move = behav.events.t_move  # movement onset time in sec
        self.t_stop = behav.events.t_stop  # movement offset time in sec
        self.t_reward = behav.events.t_reward  # rerward time in sec
        
        
        # Basic Vals
        self.num_trials = len(self.t_stop)
        self.dist_to_fly = np.asarray([np.linalg.norm(np.array([0, -32.5]) - np.array([self.x_fly[i], self.y_fly[i]]))
                                  for i in range(0, self.num_trials)])
        self.index_tstop = [np.where(self.timept[i] >= self.t_stop[i])[0][0]
                       for i in range(0, self.num_trials)]
        self.index_tstart = [np.where(self.timept[i] >= self.t_move[i])[0][0]
                       for i in range(0, self.num_trials)]
        self.dist_stop_to_fly = np.asarray([np.linalg.norm(np.array([self.x_monk[i][self.index_tstop[i]], self.y_monk[i][self.index_tstop[i]]]) - np.array([self.x_fly[i], self.y_fly[i]]))
                                       for i in range(0, self.num_trials)])
        self.euc_distmoved = np.asarray([np.linalg.norm(np.array([0, 0])-np.array([self.x_monk[i][self.index_tstop[i]], self.y_monk[i][self.index_tstop[i]]]))
                                    for i in range(0, self.num_trials)])
        self.rad_target_init = [self.rad_target[i][np.where(~np.isnan(self.rad_target[i]))[0][0]] for i in range(0, self.num_trials)]
        self.ang_target_init = [self.ang_target[i][np.where(~np.isnan(self.ang_target[i]))[0][0]] for i in range(0, self.num_trials)]
        
        def calc_dists(ind):
            xpoints = np.sin(self.ang_path[ind]*(np.pi/180))*self.rad_path[ind]
            ypoints = np.cos(self.ang_path[ind]*(np.pi/180))*self.rad_path[ind]
        
            euc = np.sqrt(np.ediff1d(xpoints)**2+np.ediff1d(ypoints)**2)
        
            tot_dist = np.sum(euc)
            return tot_dist
        
        
        self.path_len = np.asarray([calc_dists(i) for i in range(0, self.num_trials)])

        self.prop_dist = np.divide(self.dist_stop_to_fly, self.dist_to_fly)
        
        
        #  trial information: numpy structured array with one row per trial
        
        self.trial_type = exp_data.info.trial_type
        
        print('trial types:', self.trial_type.dtype.names)


    
        self.trial_pool = np.where((self.trial_type['ptb'] == 0)
                              &(self.trial_type['microstim'] == 0)
                              &(self.trial_type['replay'] == 0)
                              &(self.trial_type['all']))[0]
        s = []
        f = []
        [s.append(i) if self.trial_type['reward'][i]==1 else f.append(i) for i in self.trial_pool]
        self.successes = s
        self.failures = f
    
def construct_bgam_sample(trial, index, window):
    # num targ vars = 7
    trial_len = int(np.around(len(trial.timept[index])*window))
    print(trial_len)
    num_targ_vars = 7
    data_mat = np.zeros((num_targ_vars, trial_len))
    # set outcome
    data_mat[0,:] = trial.trial_type['reward'][index]
    # time since firefly
    data_mat[1,:] = trial.timept[index][0:trial_len]
    # time since move
    data_mat[2,:] = np.arange((trial.index_tstart[index])*-1, trial_len- trial.index_tstart[index])*.006
    # set density
    data_mat[3,:] = trial.trial_type['density'][index]
    # continuous distance to fly
    continuous_dist = np.asarray([np.linalg.norm(np.array([trial.x_monk[index][i], trial.y_monk[index][i]-32.5]) 
                                                 - np.array([trial.x_fly[index], trial.y_fly[index]]))
                                  for i in range(0, trial_len)])
    continuous_dist[np.isnan(continuous_dist)] = trial.dist_to_fly[index]
    data_mat[4,:] = continuous_dist[0:trial_len]
    data_mat[5,:] = trial.lin_vel[index][0:trial_len]
    data_mat[6,:] = trial.ang_vel[index][0:trial_len]
    
    return data_mat

def construct_bgam_downsample(trial, index, window, avg_over):
    # TODO- rename avg_over
    # num targ vars = 7
    ind_stop = np.where(trial.timept[index] >= trial.t_stop[index])[0][0]
    wind_len = int(np.around(len(trial.timept[index][0:ind_stop])*window))
    trial_len = int(np.floor(wind_len/avg_over))
    # make targ_vars an input variable
    num_targ_vars = 7
    data_mat = np.zeros((num_targ_vars, trial_len))
    # set outcome
    data_mat[0,:] = trial.trial_type['reward'][index]
    # time since firefly
    data_mat[1,:] = trial.timept[index][0:wind_len-avg_over+1:avg_over]
    # time since move
    t_since_move = np.zeros(trial_len)
    t_range = np.arange(trial.index_tstart[index], wind_len, avg_over)*.006
    startes = trial_len-len(t_range)
    t_since_move[startes:]=t_range
    data_mat[2,:] = np.arange(0, wind_len-avg_over+1, avg_over)*.006 - trial.index_tstart[index]*.006

    #np.arange((trial.index_tstart[index])*-1, trial_len- trial.index_tstart[index], avg_over)*.006
    #print(data_mat[2,:])
    # set density
    data_mat[3,:] = trial.trial_type['density'][index]
    # continuous distance to fly
    # this is non continuous
    continuous_dist = np.asarray([np.linalg.norm(np.array([trial.x_monk[index][i], trial.y_monk[index][i]-32.5]) 
                                                 - np.array([trial.x_fly[index], trial.y_fly[index]]))
                                  for i in range(0, wind_len)])#figure out how to do nicely
    continuous_dist[np.isnan(continuous_dist)] = trial.dist_to_fly[index]
    continuous_dist = continuous_dist[:trial_len*avg_over]
    #should i average distnace of step? is technically continuous
    data_mat[4,:] = np.average(np.reshape(continuous_dist, (trial_len, avg_over)), axis = 1)

    data_mat[5,:] = trial.lin_vel[index][0:wind_len-avg_over+1:avg_over]
    data_mat[6,:] = trial.ang_vel[index][0:wind_len-avg_over+1:avg_over]
    
    #print(np.where(trial.timept[index] >= trial.t_reward[index])[0]/len(trial.timept[index]))
    #print(np.where(trial.timept[index] >= trial.t_stop[index])[0][0]/len(trial.timept[index]))
    
          
    return data_mat

def construct_bgam_downsample_backup(trial, index, window, avg_over):
    # TODO- rename avg_over
    # num targ vars = 7
    wind_len = int(np.around(len(trial.timept[index])*window))
    trial_len = int(np.floor(wind_len/avg_over))
    # make targ_vars an input variable
    num_targ_vars = 7
    data_mat = np.zeros((num_targ_vars, trial_len))
    # set outcome
    data_mat[0,:] = trial.trial_type['reward'][index]
    # time since firefly
    data_mat[1,:] = trial.timept[index][0:wind_len-avg_over+1:avg_over]
    # time since move
    t_since_move = np.zeros(trial_len)
    t_range = np.arange(trial.index_tstart[index], wind_len, avg_over)*.006
    startes = trial_len-len(t_range)
    t_since_move[startes:]=t_range
    data_mat[2,:] = np.arange(0, wind_len-avg_over+1, avg_over)*.006 - trial.index_tstart[index]*.006

    #np.arange((trial.index_tstart[index])*-1, trial_len- trial.index_tstart[index], avg_over)*.006
    #print(data_mat[2,:])
    # set density
    data_mat[3,:] = trial.trial_type['density'][index]
    # continuous distance to fly
    # this is non continuous
    continuous_dist = np.asarray([np.linalg.norm(np.array([trial.x_monk[index][i], trial.y_monk[index][i]-32.5]) 
                                                 - np.array([trial.x_fly[index], trial.y_fly[index]]))
                                  for i in range(0, wind_len)])#figure out how to do nicely
    continuous_dist[np.isnan(continuous_dist)] = trial.dist_to_fly[index]
    continuous_dist = continuous_dist[:trial_len*avg_over]
    #should i average distnace of step? is technically continuous
    data_mat[4,:] = np.average(np.reshape(continuous_dist, (trial_len, avg_over)), axis = 1)

    data_mat[5,:] = trial.lin_vel[index][0:wind_len-avg_over+1:avg_over]
    data_mat[6,:] = trial.ang_vel[index][0:wind_len-avg_over+1:avg_over]
    
    #print(np.where(trial.timept[index] >= trial.t_reward[index])[0]/len(trial.timept[index]))
    #print(np.where(trial.timept[index] >= trial.t_stop[index])[0][0]/len(trial.timept[index]))
    
          
    return data_mat
    
def read_multiple_SF(df, base_file):
    timept, lin_vel,ang_vel, lin_acc, ang_acc, rad_target,\
        ang_target,rad_path,ang_path,x_monk,y_monk,\
            x_fly, y_fly, t_flyON, t_flyOFF, t_move, \
                t_stop, t_reward, num_trials, dist_to_fly,\
                    index_tstop, index_tstart, dist_stop_to_fly, \
                        euc_distmoved,rad_target_init, ang_target_init, \
                            path_len, prop_dist, trial_type = boot_it_up(base_file)
    #binlinreg() #TODO-perform with balanced population sizes
    
    trial_pool = np.where((trial_type['ptb'] == 0)
                          &(trial_type['microstim'] == 0)
                          &(trial_type['replay'] == 0)
                          &(trial_type['all']))[0]
    print(trial_type['ptb'].tolist())
    print(trial_type['microstim'].tolist())
    print(trial_type['replay'].tolist())
    print(trial_type['controlgain'].tolist())
    print(trial_type['all'].tolist())
    print(trial_type['reward'].tolist())
    
    s = []
    f = []
    [s.append(i) if trial_type['reward'][i]==1 else f.append(i) for i in trial_pool]
    
    print(len(s))
    print(len(f))
    if len(s) > len(f):
        subsample_s = np.random.random_integers(len(s)-1, size=len(f))
        s = [s[x] for x in subsample_s]
    elif len(f) > len(s): 
        subsample_f = np.random.random_integers(len(f)-1, size=len(s))
        f = [f[x] for x in subsample_f]
    s_bin = np.ones(len(s))
    f_bin = np.zeros(len(f))
    trial_pool = np.concatenate((s,f))
    print(s)
    print(f)
    print(len(s))

    print(len(f))
    bin_trial_suc = np.concatenate((s_bin,f_bin))

    #bin_trial_suc = trial_type['reward'][trial_pool]#np.concatenate((k_bin,g_bin))

    # Dependant Variable
    final_dists = prop_dist[trial_pool]
    # Independent variables

    initial_dists = [dist_to_fly[x] for x in trial_pool]
    ang_dists = [ang_target_init[x] for x in trial_pool]
    rad_dists = [rad_target_init[x] for x in trial_pool]
    
    t_moves = [t_move[x][0] for x in trial_pool]
    start_early = [1 if i < 0 else 0 for i in t_moves]
    reaction_time = [t_move[x][0] if t_move[x][0] > 0  else 0 for x in trial_pool]
    density = trial_type['density'][trial_pool]
    landmarks = trial_type['landmark'][trial_pool]
    fireflystays = trial_type['firefly_fullON'][trial_pool]
    initial_vel1 = [np.average(lin_vel[x][index_tstart[x]:index_tstart[x]+50]) for x in trial_pool]
    initial_vel2 = [np.average(lin_vel[x][index_tstart[x]:index_tstart[x]+100]) for x in trial_pool]
    initial_vel3 = [np.average(lin_vel[x][index_tstart[x]:index_tstart[x]+150]) for x in trial_pool]
    #[print(timept[x][index_tstart[x]+100]-timept[x][index_tstart[x]]) for x in trial_pool]

    d = {'successes': bin_trial_suc, 'index':trial_pool,'initial_dists': initial_dists, 'ang_dists':ang_dists,
         'rad_dists':rad_dists, 't_moves':t_moves, 'start_early':start_early,
         'density':density, 'landmarks':landmarks, 'fireflystays':fireflystays,
         'initial_vel1':initial_vel1, 'initial_vel2':initial_vel2, 
         'reaction_time':reaction_time,'initial_vel3':initial_vel3}
    df2 = pd.DataFrame(data=d)
    df2['start_early']=df2['start_early'].astype('category')
    df2['density']=df2['density'].astype('category')
    df2['landmarks']=df2['landmarks'].astype('category')
    df2['fireflystays']=df2['fireflystays'].astype('category')
    df = df.append(df2)
    return df

if __name__ == '__main__':
    
    plt.figure()
    m53s100 = Trial('/Users/isabelgaron/anaconda3/envs/firefly/proj_files/monkey/m53s100.mat')
    final = []
    for i in range(1, 10):
        samp_s = construct_bgam_downsample(m53s100, m53s100.successes[i], 1, 17)
        samp_f = construct_bgam_downsample(m53s100, m53s100.failures[i], 1, 17)
        final.append(samp_s)
        final.append(samp_f)
    final = np.hstack(final)
    print(final.shape)
    plt.figure()
    plt.imshow(final, aspect='auto')
    plt.show()
    
