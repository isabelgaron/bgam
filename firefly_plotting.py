#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:58:53 2022

@author: isabelgaron + EDOARDO
"""

import sys
path_to_ff_lib = 'path/to/firefly_utils/'
sys.path.append(path_to_ff_lib)
import matplotlib.pylab as plt
from scipy.io import loadmat
from firefly_utils.data_handler import *

class Session(object):
    def __init__(self, base_file):

        behav_stat_key = 'behv_stats'
        spike_key = 'units'
        behav_dat_key = 'trials_behv'
        lfp_key = 'lfps'

        # duration of the pre-trial interval in sec
        pre_trial_dur = 0.2
        post_trial_dur = 0.2

        # list of sesssion for which the eye tracking was on left eye
        use_left_eye = ['53s48']

        dat = loadmat(base_file)
        exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur,
                                post_trial_dur=post_trial_dur, lfp_alpha=None, lfp_beta=None, lfp_theta=None,
                                extract_lfp_phase=True,
                                use_eye=use_left_eye, extract_fly_and_monkey_xy=True, skip_not_ok=False)
        spk = spike_counts(dat, spike_key)
        self.brain_area = spk.brain_area
        self.spike_times = spk.spike_times
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

        self.r_belief = behav.continuous.rad_belief
        self.theta_belief = behav.continuous.ang_belief

        self.x_leye = behav.continuous.xleye
        self.y_leye = behav.continuous.yleye

        self.x_reye = behav.continuous.xreye
        self.y_reye = behav.continuous.yreye

        # EVENT VARS
        self.t_flyON = behav.events.t_targ  # tarrget onset (0sec)
        self.t_flyOFF = behav.events.t_flyOFF  # ff offset time in sec (0.3 sec is the std)
        self.t_move = behav.events.t_move  # movement onset time in sec
        self.t_stop = behav.events.t_stop  # movement offset time in sec
        self.t_reward = behav.events.t_reward  # rerward time in sec

        # Basic Vals
        self.num_trials = len(self.t_stop)

        #  trial information: numpy structured array with one row per trial

        self.trial_type = exp_data.info.trial_type

        # Basic Vals
        self.num_trials = len(self.t_stop)
        self.dist_to_fly = np.asarray([np.linalg.norm(np.array([0, -32.5]) - np.array([self.x_fly[i], self.y_fly[i]]))
                                       for i in range(0, self.num_trials)])
        self.index_tstop = [np.where(self.timept[i] >= self.t_stop[i])[0][0]
                            for i in range(0, self.num_trials)]
        self.index_tstart = [np.where(self.timept[i] >= self.t_move[i])[0][0]
                             for i in range(0, self.num_trials)]
        self.dist_stop_to_fly = np.asarray([np.linalg.norm(
            np.array([self.x_monk[i][self.index_tstop[i]], self.y_monk[i][self.index_tstop[i]]]) - np.array(
                [self.x_fly[i], self.y_fly[i]]))
                                            for i in range(0, self.num_trials)])
        self.euc_distmoved = np.asarray([np.linalg.norm(
            np.array([0, 0]) - np.array([self.x_monk[i][self.index_tstop[i]], self.y_monk[i][self.index_tstop[i]]]))
                                         for i in range(0, self.num_trials)])

        def calc_dists(ind):
            xpoints = np.sin(self.ang_path[ind] * (np.pi / 180)) * self.rad_path[ind]
            ypoints = np.cos(self.ang_path[ind] * (np.pi / 180)) * self.rad_path[ind]

            euc = np.sqrt(np.ediff1d(xpoints) ** 2 + np.ediff1d(ypoints) ** 2)

            tot_dist = np.sum(euc)
            return tot_dist

        self.path_len = np.asarray([calc_dists(i) for i in range(0, self.num_trials)])

        self.prop_dist = np.divide(self.dist_stop_to_fly, self.dist_to_fly)

        #  trial information: numpy structured array with one row per trial

        print('trial types:', self.trial_type.dtype.names)

        self.trial_pool = np.where((self.trial_type['ptb'] == 0)
                                   & (self.trial_type['replay'] == 0)
                                   & (self.trial_type['all']))[0]

        self.trial_pool = [x for x in self.trial_pool if np.all(np.isnan(self.x_reye[x])) == False]
        s = []
        f = []
        [s.append(i) if self.trial_type['reward'][i] == 1 else f.append(i) for i in self.trial_pool]
        self.successes = s
        self.failures = f

        self.ppc_chans = np.where(exp_data.brain_area == 'PPC')[0]
        self.pfc_chans = np.where(exp_data.brain_area == 'PFC')[0]
        self.mst_chans = np.where(exp_data.brain_area == 'MST')[0]
        self.vip_chans = np.where(exp_data.brain_area == 'VIP')[0]
        arr_areas = []
        if len(self.ppc_chans):
            arr_areas.append("PPC")
        if len(self.pfc_chans):
            arr_areas.append("PFC")
        if len(self.mst_chans):
            arr_areas.append("MST")
        if len(self.vip_chans):
            arr_areas.append("VIP")
        self.incl_brain_areas = arr_areas

def mvmt_graph(session, trial_num):

    plt.figure()
    x_rew_circle = 60 * np.sin(np.pi * 2 * np.linspace(0, 1, 100))
    y_rew_circle = 60 * np.cos(np.pi * 2 * np.linspace(0, 1, 100))

    plt.title(trial_num)

    sel_time = (session.timept[trial_num] >= session.t_flyON[trial_num]) & (
            session.timept[trial_num] <= session.t_stop[trial_num])
    plt.ylim(-40, 400)
    plt.xlim(-300, 300)
    plt.plot(session.x_monk[trial_num][sel_time], session.y_monk[trial_num][sel_time])
    plt.scatter(session.x_fly[trial_num], session.y_fly[trial_num], color='r')
    plt.plot(x_rew_circle + session.x_fly[trial_num],
             y_rew_circle + session.y_fly[trial_num], 'r')
    plt.plot(session.x_monk[trial_num], session.y_monk[trial_num])
    plt.show()

def vel_graph(session, trial_num):
    plt.figure()
    plt.subplot(121)

    plt.plot(session.timept[trial_num], session.lin_vel[trial_num])

    plt.axvspan(session.t_flyON[trial_num], session.t_flyOFF[trial_num],
                color="yellow", zorder=0)

    plt.xlabel('time [sec]')
    plt.title("Linear Velocity")
    plt.subplot(122)

    plt.plot(session.timept[trial_num], session.ang_vel[trial_num])

    plt.axvspan(session.t_flyON[trial_num], session.t_flyOFF[trial_num],
                color="yellow", zorder=0)

    plt.xlabel('time [sec]')

    plt.tight_layout()
    plt.title("Angular Velocity")
    plt.show()

def print_session_info(session):
    num_trial = session.num_trials
    num_succ = np.sum([1 if (session.trial_type['reward'][i] == 1) &(session.trial_type['all'][i]) else 0 for i in range(0, session.num_trials)])
    num_fail = np.sum([1 if session.trial_type['reward'][i] == 0 &(session.trial_type['all'][i]) else 0 for i in range(0, session.num_trials)])
    s_f_ratio = num_fail/num_succ
    num_bad = np.sum([1 if session.trial_type['all'][i] == 0 else 0 for i in range(0, session.num_trials)])
    num_nomove = np.sum([1 if (session.t_stop[i][0] > 6.5) else 0 for i in range(0, session.num_trials)])

    high_dense = len(np.where((session.trial_type['density'] == .005))[0])
    low_dense = len(np.where((session.trial_type['density'] == .0001))[0])
    ptbs = len(np.where((session.trial_type['ptb'] == 1))[0])
    micros = len(np.where((session.trial_type['microstim'] == 1))[0])
    lands = len(np.where((session.trial_type['landmark'] == 1))[0])
    repls = len(np.where((session.trial_type['replay'] == 1))[0])
    controls = len(np.where((session.trial_type['controlgain'] != 1))[0])
    full_on = len(np.where((session.trial_type['firefly_fullON'] == 1))[0])

    num_channels = len(session.brain_area)
    ppc = len(session.ppc_chans)
    pfc = len(session.pfc_chans)
    mst = len(session.mst_chans)
    vip = len(session.vip_chans)
    arr_areas = []
    if ppc:
        arr_areas.append("PPC")
    if pfc:
        arr_areas.append("PFC")
    if mst:
        arr_areas.append("MST")
    if vip:
        arr_areas.append("VIP")
    print("# trials: " +str(num_trial))
    print("Brain areas: " +str(arr_areas))
    print("# channels: " +str(num_channels))
    print("# trials rewarded: " +str(num_succ))
    print("# trials unrewarded: " +str(num_fail))
    print("# trials excluded: " +str(num_bad))
    print("# trials no movement: " +str(num_nomove))
    print("Low density: " +str(low_dense))
    print("High density: " +str(high_dense))
    print("# perturbation trials: " +str(ptbs))
    print("# microstim trials: " +str(micros))
    print("# landscape trials: " +str(lands))
    print("# replay trials: " +str(repls))
    print("# trials w/controlgain: " +str(controls))
    print("# trials firefly stays on: " +str(full_on))

def make_raster(session, trial_num):
    num_areas = len(session.incl_brain_areas)
    fig, axs = plt.subplots(num_areas, 1, sharex=True)
    for i in range(0, num_areas):
        chans = np.where(session.brain_area == session.incl_brain_areas[i])[0]
        n_inds = []
        n_spikes = []
        for j in range(0, len(chans)):
            n_inds = np.concatenate((n_inds, np.repeat(j, len(session.spike_times[chans[j]][trial_num]))))
            n_spikes = np.concatenate((n_spikes, session.spike_times[chans[j]][trial_num]))
        axs[i].scatter(n_spikes, n_inds, color = "k", s = .5)
        axs[i].set_title(str(session.incl_brain_areas[i]))
        axs[i].set_ylabel("Channel")
        axs[i].axvline(session.t_move[trial_num], label ="t_move", color = "green")
        axs[i].axvline(session.t_stop[trial_num], label="t_stop", color = "red")
        axs[i].axvspan(session.t_flyON[trial_num], session.t_flyOFF[trial_num],
                    color="yellow", zorder=0, label = "firefly on")
        if session.t_reward[trial_num]:
            axs[i].axvline(session.t_reward[trial_num], label="t_reward", color = "purple")
    if session.trial_type[trial_num][1] == 1:
        outcome = "Success"
    else:
        outcome = "Failure"
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    axs[0].set_title(str(session.incl_brain_areas[0] + " - " + outcome))
    axs[-1].set_xlabel("Time (sec)")
    plt.show()

def make_raster_single_chans(session, trial_nums, chans):
    fig, axs = plt.subplots(len(chans), 1, sharex=True)
    for i in range(0, len(chans)):
        n_inds = []
        n_spikes = []
        for j in range(0, len(trial_nums)):
            n_inds = np.concatenate((n_inds, np.repeat(j, len(session.spike_times[chans[i]][trial_nums[j]]))))
            n_spikes = np.concatenate((n_spikes, session.spike_times[chans[i]][trial_nums[j]]))
        axs[i].scatter(n_spikes, n_inds, color = "k", s = .5)
        axs[i].set_title(str(session.brain_area[chans[i]]) + " - " + str(chans[i]))
        axs[i].set_ylabel("Trial")
        axs[i].axvspan(session.t_flyON[trial_nums[0]], session.t_flyOFF[trial_nums[0]],
                    color="yellow", zorder=0, label = "firefly on")

    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    axs[-1].set_xlabel("Time (sec)")
    plt.show()

if __name__ == '__main__':

    base_file = 'path/to/mat/file/m53s114.mat'
    session = Session(base_file)
    print_session_info(session)
    mvmt_graph(session, 7)
    vel_graph(session, 7)
    make_raster(session, 7)
    make_raster_single_chans(session, session.successes[:100], [9, 10, 12, 14])
    x=0