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
from GAM_library import *
import gam_data_handlers as gdh
import matplotlib.pylab as plt
import pandas as pd
import scipy.stats as sts
from post_processing import postprocess_results
import statsmodels.api as sm
from bgam_data_handler import *

def meth3_arbitrary():
    plt.figure()
    np.random.seed(5)
    nobs = 1000#000#2*10**5 bigger does not improve atm
    
    size_knots = 10
    order = 4
    ### values of outcome not input
    
    func1 = lambda x : np.log((10*x - 2)**2 +3)
    func2 = lambda x: 3*x+3
    func3 = lambda x: np.sin(3*x)
    func4 = lambda x: np.log((2*x +3)**2 +5)
    func5 = lambda x: np.round(x/2)
    func6 = lambda x: -3*(x+2)
    func7 = lambda x: x**2
    func8 = lambda x: 5*x
    func9 = lambda x: 2/x
    
    lis_funcs = [func1, func2, func3, func4, func5, func6]#, func7, func8, func9]
    x_ranges = [[-2,2], [-2,2],[-1,1], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [.1,2]]
    xs = []
    xxs = []
    
    sm_handler = smooths_handler()
    var_list = []
    s = []
    for i in range(0, len(lis_funcs)):
        x1 = np.random.uniform(x_ranges[i][0], x_ranges[i][1], size=nobs)
        xs.append(x1)
        xx1 = np.linspace(x_ranges[i][0]+0.001, x_ranges[i][1]-0.001,100)
        xxs.append(xx1)
        name = '1d_var'+str(i)
        var_list.append(name)
        sm_handler.add_smooth(name, [x1], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)
        s.append(lis_funcs[i](x1))
        # fix this
    st = np.sum(s, axis=0)
    
    link = sm.genmod.families.links.logit()
    bernouFam = sm.genmod.families.family.Binomial(link=link)
    p = bernouFam.link.inverse(st)#-np.mean(s)))#/(np.mean(s)/2))
    # rescale to be correct ratio of zeros and one
    plt.figure()
    plt.plot(p)
    plt.plot(st)
    #plt.plot(np.log(curr3(x3)))

    y = np.random.binomial(1, p)
    plt.plot(y)
    plt.show()

    
    link = sm.genmod.families.links.logit()
    
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
    for j in range(0, len(var_list)):
        f1y, f1y_min, f1y_max, la1 = full.smooth_compute([xxs[j]], var_list[j])
        plt.subplot(1, len(var_list), j+1)
        plt.title(var_list[j])
        plt.plot(xxs[j], lis_funcs[j](xxs[j]),color='g',label='f1')
        plt.plot(xxs[j], f1y, 'k-', label='model')
        plt.fill_between(xxs[j], f1y_min,f1y_max)
    
    plt.show()

def meth3():
    plt.figure()
    np.random.seed(5)
    nobs = 1000#000#2*10**5 bigger does not improve atm
    
    size_knots = 10
    order = 4
    
    x1_r = [-2,2]
    x2_r = [-2,2]
    x3_r = [-1,1]
    x4_r = [-2,2]
    x5_r = [-2,2]
    x6_r = [-2,2]
    x7_r = [-2,2]
    
    x1 = np.random.uniform(x1_r[0], x1_r[1], size=nobs) #-.5, .5
    x2 = np.random.uniform(x2_r[0], x2_r[1], size=nobs)
    x3 = np.random.uniform(x3_r[0], x3_r[1], size=nobs) #-2, 2
    x4 = np.random.uniform(x4_r[0], x4_r[1], size=nobs)
    x5 = np.random.uniform(x5_r[0], x5_r[1], size=nobs) #-1, 1
    x6 = np.random.uniform(x6_r[0], x6_r[1], size=nobs)
    x7 = np.random.uniform(x7_r[0], x7_r[1], size=nobs)
    plt.figure()
    plt.plot(np.sort(x7))
    ### values of outcome not input
    
    func1 = lambda x : (10*x - 2)**2 +3
    func2 = lambda x: 3*x+3
    func3 = lambda x: np.sin(3*x)
    func4 = lambda x: (2*x +3)**3 +5
    func5 = lambda x: np.round(x/2)
    func6 = lambda x: -3*(x+2)
    func7 = lambda x: x**2
    
    curr1=func1
    curr2=func2
    curr3=func3
    curr4=func4
    curr5=func5
    curr6=func6
    curr7=func7
    
    ids = np.repeat(np.arange(200),nobs//200)
    
    link = sm.genmod.families.links.logit()
    bernouFam = sm.genmod.families.family.Binomial(link=link)
    s = (np.log(curr1(x1))+curr2(x2) +curr3(x3)+np.log(curr4(x4))+curr5(x5)+curr6(x6)+curr7(x7))
    print(s)
    p = bernouFam.link.inverse(s)#-np.mean(s)))#/(np.mean(s)/2))
    # rescale to be correct ratio of zeros and one
    plt.figure()
    plt.plot(p)
    plt.plot(s)
    #plt.plot(np.log(curr3(x3)))

    y = np.random.binomial(1, p)
    plt.plot(y)
    plt.show()
    
    xx1 = np.linspace(x1_r[0]+0.001, x1_r[1]-0.001,100)
    xx2 = np.linspace(x2_r[0]+0.001, x2_r[1]-0.001,100)
    xx3 = np.linspace(x3_r[0]+0.001, x3_r[1]-0.001,100)
    xx4 = np.linspace(x4_r[0]+0.001, x4_r[1]-0.001,100)
    xx5 = np.linspace(x5_r[0]+0.001, x5_r[1]-0.001,100)
    xx6 = np.linspace(x6_r[0]+0.001, x6_r[1]-0.001,100)
    xx7 = np.linspace(x7_r[0]+0.001, x7_r[1]-0.001,100)


    sm_handler = smooths_handler()
    sm_handler.add_smooth('1d_var', [x1], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False], trial_idx=ids, lam=None,penalty_type='der',der=2)

    sm_handler.add_smooth('1d_var2', [x2], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    
    # TODO - knots only defined -1 to 1 in cyclic, better to construct by hand?
    sm_handler.add_smooth('1d_var3', [x3], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[True],  trial_idx=ids, lam=None,penalty_type='der',der=2, knots_percentiles=(0,100))
    
    sm_handler.add_smooth('1d_var4', [x4], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('1d_var5', [x5], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('1d_var6', [x6], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('1d_var7', [x7], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    
    var_list = ['1d_var','1d_var2','1d_var3','1d_var4','1d_var5','1d_var6','1d_var7']


    link = sm.genmod.families.links.logit()
    
    gam_model = general_additive_model(sm_handler,var_list,y,
                                           bernouFam,fisher_scoring=False)

    full,reduced = gam_model.fit_full_and_reduced(var_list,th_pval=0.001, max_iter=10 ** 4,#, tol=10 ** (-8),
                                  conv_criteria='gcv', trial_num_vec=ids,
                                  method='L-BFGS-B', #smooth_pen=[1] * 4, 
                                  gcv_sel_tol=10 ** (-13), use_dgcv=True, fit_initial_beta=True,pseudoR2_per_variable=True)
    

    
    
    f1y, f1y_min, f1y_max, la1 = full.smooth_compute([xx1], '1d_var', trial_idx=ids)
    f2y, f2y_min, f2y_max, la2 = full.smooth_compute([xx2], '1d_var2', trial_idx=ids)
    f3y, f3y_min, f3y_max, la3 = full.smooth_compute([xx3], '1d_var3', trial_idx=ids)
    f4y, f4y_min, f4y_max, la4 = full.smooth_compute([xx4], '1d_var4', trial_idx=ids)
    f5y, f5y_min, f5y_max, la5 = full.smooth_compute([xx5], '1d_var5', trial_idx=ids)
    f6y, f6y_min, f6y_max, la6 = full.smooth_compute([xx6], '1d_var6', trial_idx=ids)
    f7y, f7y_min, f7y_max, la7 = full.smooth_compute([xx7], '1d_var7', trial_idx=ids)
    
    plt.close('all')
    plt.figure()
    plt.title("log")
    plt.subplot(161)
    plt.title("function 1")
    plt.plot(plt.log(xx1), np.log(curr1(xx1)),color='g',label='f1')
    plt.plot(plt.log(xx1), f1y, 'k-', label='model')
    plt.fill_between(plt.log(xx1), f1y_min,f1y_max)
    plt.subplot(162)
    plt.title("function 2")
    plt.plot(plt.log(xx2), curr2(xx2),color='g',label='f2')
    plt.plot(plt.log(xx2), f2y, 'k-', label='model')
    plt.fill_between(plt.log(xx2), f2y_min, f2y_max)
    plt.subplot(163)
    plt.title("function 3")
    plt.plot(plt.log(xx3), curr3(xx3),color='g',label='f3')
    plt.plot(plt.log(xx3), f3y, 'k-', label='model')
    plt.fill_between(plt.log(xx3), f3y_min, f3y_max)
    plt.subplot(164)
    plt.title("function 4")
    plt.plot(plt.log(xx4), np.log(curr4(xx4)),color='g',label='f4')
    plt.plot(plt.log(xx4), f4y, 'k-', label='model')
    plt.fill_between(plt.log(xx4), f4y_min, f4y_max)
    plt.subplot(165)
    plt.title("function 5")
    plt.plot(plt.log(xx5), curr5(xx5),color='g',label='f5')
    plt.plot(plt.log(xx5), f5y, 'k-', label='model')
    plt.fill_between(plt.log(xx5), f5y_min, f5y_max)
    plt.subplot(166)
    plt.title("function 6")
    plt.plot(plt.log(xx6), curr6(xx6),color='g',label='f6')
    plt.plot(plt.log(xx6), f6y, 'k-', label='model')
    plt.fill_between(plt.log(xx6), f6y_min, f6y_max)
    plt.legend()
    
    
    plt.figure()
    
    
    plt.subplot(171)
    plt.title("Function 1")
    plt.plot(xx1, np.log(curr1(xx1)),color='g',label='f1')
    plt.plot(xx1, f1y, 'k-', label='model')
    plt.fill_between(xx1, f1y_min,f1y_max)
    plt.subplot(172)
    plt.title("Function 2")
    plt.plot(xx2, curr2(xx2),color='g',label='f2')
    plt.plot(xx2, f2y, 'k-', label='model')
    plt.fill_between(xx2, f2y_min,f2y_max)
    plt.subplot(173)
    plt.title("Function 3")
    plt.plot(xx3, curr3(xx3),color='g',label='f3')
    plt.plot(xx3, f3y, 'k-', label='model')
    plt.fill_between(xx3, f3y_min,f3y_max)
    plt.subplot(174)
    plt.title("Function 4")
    plt.plot(xx4, np.log(curr4(xx4)),color='g',label='f4')
    plt.plot(xx4, f4y, 'k-', label='model')
    plt.fill_between(xx4, f4y_min,f4y_max)
    plt.subplot(175)
    plt.title("Function 5")
    plt.plot(xx5, curr5(xx5),color='g',label='f5')
    plt.plot(xx5, f5y, 'k-', label='model')
    plt.fill_between(xx5, f5y_min,f5y_max)
    plt.subplot(176)
    plt.title("Function 6")
    plt.plot(xx6, curr6(xx6),color='g',label='f6')
    plt.plot(xx6, f6y, 'k-', label='model')
    plt.fill_between(xx6, f6y_min,f6y_max)
    plt.subplot(177)
    plt.title("Function 7")
    plt.plot(xx7, curr7(xx7),color='g',label='f7')
    plt.plot(xx7, f7y, 'k-', label='model')
    plt.fill_between(xx7, f7y_min,f7y_max)
    plt.show()

def meth3_subsample():
    
    plt.figure()
    np.random.seed(5)
    nobs = 3000#000#2*10**5 bigger does not improve atm
    
    size_knots = 10
    order = 4
    window = 20
    
    x = np.linspace(-2, 2, 1000)
    low_t = window/2
    high_t = len(x) - window/2
    
    func1 = lambda x: ((x+2)**3)/10
    func2 = lambda x: 1*(x-0.5)**2
    func3 = lambda x: np.sin(x)
    func4 = lambda x: 1*(x)**3
    
    curr1=func2(x)
    '''curr2=func2(x)
    curr3=func3(x)
    curr4=func4(x)'''
    
    ts = np.random.randint(low_t, high_t, size=nobs)
    #tempx = np.array([x[i-int(window/2):int(i+window/2)] for i in ts]).flatten()
    windows1 = np.array([curr1[i-int(window/2):int(i+window/2)] for i in ts])
    c1 = np.average(windows1, axis=1)
    '''windows2 = np.array([curr2[i-int(window/2):int(i+window/2)] for i in ts])
    c2 = np.average(windows2, axis=1)
    windows3 = np.array([curr3[i-int(window/2):int(i+window/2)] for i in ts])
    c3 = np.average(windows3, axis=1)
    windows4 = np.array([curr4[i-int(window/2):int(i+window/2)] for i in ts])
    c4 = np.average(windows4, axis=1)'''
    
    ts = np.repeat(ts, window)
    print(ts)
    ids = np.repeat(np.arange(nobs),window)
    
    link = sm.genmod.families.links.logit()
    bernouFam = sm.genmod.families.family.Binomial(link=link)
    p = bernouFam.link.inverse(c1)#+c3+c1+c4)

    y = np.random.binomial(1, p)
    #print(y.tolist())
    y = np.repeat(y, window)
    #print(y.tolist())
    
    '''print(p.shape)
    print(y.shape)
    print(windows1.flatten().shape)
    print(ids.shape)
    plt.figure()
    plt.plot(y)
    plt.plot(windows1.flatten())
    plt.plot(curr1)
    plt.plot(x)
    plt.show()'''
    # Construct range
    '''knots1 = np.linspace(-2,2,size_knots)
    knots1 = np.hstack(([knots1[0]]*(order-1), knots1, [knots1[-1]]*(order-1)))
    
    # add ends
    xx1 = np.linspace(-2+0.001, knots1[-1]-0.001,100)
    bx1 = gdh.splineDesign(knots1, xx1, 
                       ord=4, der=0,outer_ok=True)
    
    knots2 = np.linspace(-2,2,size_knots)
    knots2 = np.hstack(([knots2[0]]*(order-1), knots2, [knots2[-1]]*(order-1)))
    
    xx2 = np.linspace(-2+0.001, knots2[-1]-0.001,100)
    bx2 = gdh.splineDesign(knots2, xx2, 
                       ord=4, der=0,outer_ok=True)
    
    knots3 = np.linspace(-2,2,size_knots)
    knots3 = np.hstack(([knots3[0]]*(order-1), knots3, [knots3[-1]]*(order-1)))
    
    xx3 = np.linspace(-2+0.001, knots3[-1]-0.001,100)
    bx3 = gdh.splineDesign(knots3, xx3, 
                       ord=4, der=0,outer_ok=True)
    
    knots4 = np.linspace(-2,2,size_knots)
    knots4 = np.hstack(([knots4[0]]*(order-1), knots4, [knots4[-1]]*(order-1)))
    
    xx4 = np.linspace(-2+0.001, knots4[-1]-0.001,100)
    bx4 = gdh.splineDesign(knots4, xx4, 
                       ord=4, der=0,outer_ok=True)'''

    sm_handler = smooths_handler()
    #windows1.flatten()
    sm_handler.add_smooth('1d_var', [x[ts]], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False], trial_idx=ids, lam=None,penalty_type='der',der=2)

    '''sm_handler.add_smooth('1d_var2', [windows2.flatten()], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)

    sm_handler.add_smooth('1d_var3', [windows3.flatten()], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('1d_var4', [windows4.flatten()], ord=order, knots=None, knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=ids, lam=None,penalty_type='der',der=2)
    '''
    var_list = ['1d_var']#,'1d_var2','1d_var3','1d_var4']


    link = sm.genmod.families.links.logit()
    
    gam_model = general_additive_model(sm_handler,var_list,y,
                                           bernouFam,fisher_scoring=True)

    full,reduced = gam_model.fit_full_and_reduced(var_list,th_pval=0.001, max_iter=10 ** 4,#, tol=10 ** (-8),
                                  conv_criteria='gcv', trial_num_vec=ids,
                                  method='L-BFGS-B', #smooth_pen=[1] * 4, 
                                  gcv_sel_tol=10 ** (-13), use_dgcv=True, fit_initial_beta=True,pseudoR2_per_variable=False)
    

    
    xx1 = np.linspace(-2, 2, 100)
    print(len(xx1))
    f1y, f1y_min, f1y_max, la1 = full.smooth_compute([xx1], '1d_var', trial_idx=ids)
    '''f2y, f2y_min, f2y_max, la2 = full.smooth_compute([xx1], '1d_var2', trial_idx=ids)
    f3y, f3y_min, f3y_max, la3 = full.smooth_compute([xx1], '1d_var3', trial_idx=ids)
    f4y, f4y_min, f4y_max, la4 = full.smooth_compute([xx1], '1d_var4', trial_idx=ids)'''
    
    print(len(f1y))
    '''plt.figure()
    plt.plot(f1y)
    plt.plot(tempx, windows1.flatten())
    
    plt.show()
    plt.figure()
    plt.plot(f2y)
    plt.plot(f3y)
    plt.plot(f4y)
    plt.plot(curr2)
    plt.plot(curr3)
    plt.plot(curr4)
    plt.show()'''
    
    
    plt.close('all')
    plt.figure()
    plt.subplot(141)
    plt.title("log")
    plt.plot(plt.log(xx1), func2(xx1),color='g',label='f1')
    plt.plot(plt.log(xx1), f1y, 'k-', label='model')
    plt.legend()
    '''plt.fill_between(plt.log(xx1), f1y_min,f1y_max)
    plt.subplot(142)
    plt.plot(plt.log(xx1), func2(xx1),color='g',label='f2')
    plt.plot(plt.log(xx1), f2y, 'k-', label='model')
    plt.fill_between(plt.log(xx1), f2y_min, f2y_max)
    plt.subplot(143)
    plt.plot(plt.log(xx1), func3(xx1),color='g',label='f3')
    plt.plot(plt.log(xx1), f3y, 'k-', label='model')
    plt.fill_between(plt.log(xx1), f3y_min, f3y_max)
    plt.subplot(144)
    plt.plot(plt.log(xx1), func4(xx1),color='g',label='f4')
    plt.plot(plt.log(xx1), f4y, 'k-', label='model')
    plt.fill_between(plt.log(xx1), f4y_min, f4y_max)
    plt.legend()'''
    
    
    plt.figure()
    
    
    plt.subplot(141)
    plt.title("normal")
    plt.plot(xx1, func2(xx1),color='g',label='f1')
    plt.plot(xx1, f1y, 'k-', label='model')
    plt.legend()
    plt.fill_between(xx1, f1y_min,f1y_max)
    '''plt.subplot(142)
    plt.plot(xx1, func2(xx1),color='g',label='f2')
    plt.plot(xx1, f2y, 'k-', label='model')
    plt.fill_between(xx1, f2y_min,f2y_max)
    plt.subplot(143)
    plt.plot(xx1, func3(xx1),color='g',label='f3')
    plt.plot(xx1, f3y, 'k-', label='model')
    plt.fill_between(xx1, f3y_min,f3y_max)
    plt.subplot(144)
    plt.plot(xx1, func4(xx1),color='g',label='f4')
    plt.plot(xx1, f4y, 'k-', label='model')
    plt.fill_between(xx1, f4y_min,f4y_max)
    plt.legend()'''
    plt.show()
   
def actual_data(data):
    
    
    
    np.random.seed(5)
    nobs = len(data[0])#000#2*10**5 bigger does not improve atm
    
    size_knots = 10
    order = 4
    y=data[0,:]
    #y=(~y.astype(bool)).astype(int) # inverting bool successfully inverts lines
    x0_r = [np.min(data[0,:]),np.max(data[0,:])]
    x1_r = [np.min(data[1,:]),np.max(data[1,:])]
    x2_r = [np.min(data[2,:]),np.max(data[2,:])]
    x3_r = [np.min(data[3,:]),np.max(data[3,:])]
    x4_r = [np.min(data[4,:]),np.max(data[4,:])]
    x5_r = [np.min(data[5,:]),np.max(data[5,:])]
    x6_r = [np.min(data[6,:]),np.max(data[6,:])]
    
    print("===================")
    print(x1_r)
    print(x2_r)
    print(x3_r)
    print(x4_r)
    print(x5_r)
    print(x6_r)
    
    xx1 = np.linspace(x1_r[0], x1_r[1],100)
    xx2 = np.linspace(x2_r[0], x2_r[1],100)
    xx3 = np.linspace(x3_r[0], x3_r[1],100)
    #print(xx3)
    xx4 = np.linspace(x4_r[0], x4_r[1],100)
    xx5 = np.linspace(x5_r[0], x5_r[1],100)
    xx6 = np.linspace(x6_r[0], x6_r[1],100)

    knots1 = np.linspace(x1_r[0],x1_r[1],size_knots)
    #knots1 = np.hstack(([knots1[0]]*(order-1), knots1, [knots1[-1]]*(order-1)))
    knots2 = np.linspace(x2_r[0],x2_r[1],size_knots)
    #knots2 = np.hstack(([knots2[0]]*(order-1), knots2, [knots2[-1]]*(order-1)))
    knots3 = np.linspace(x3_r[0],x3_r[1],size_knots)
    #knots3 = np.hstack(([knots3[0]]*(order-1), knots3, [knots3[-1]]*(order-1)))
    knots4 = np.linspace(x4_r[0],x4_r[1],size_knots)
    #knots4 = np.hstack(([knots4[0]]*(order-1), knots4, [knots4[-1]]*(order-1)))
    knots5 = np.linspace(x5_r[0],x5_r[1],size_knots)
    #knots5 = np.hstack(([knots5[0]]*(order-1), knots5, [knots5[-1]]*(order-1)))
    knots6 = np.linspace(x6_r[0],x6_r[1],size_knots)
    #knots6 = np.hstack(([knots6[0]]*(order-1), knots6, [knots6[-1]]*(order-1)))
    

    sm_handler = smooths_handler()
    sm_handler.add_smooth('t_since_firefly', [data[1,:]], ord=order, knots=[knots1], knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False], trial_idx=None, lam=None,penalty_type='der',der=2)

    sm_handler.add_smooth('t_since_move', [data[2,:]], ord=order, knots=[knots2], knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=None, lam=None,penalty_type='der',der=2)
    
    # TODO - knots only defined -1 to 1 in cyclic, better to construct by hand?
    sm_handler.add_smooth('density', [data[3,:]], ord=order, knots=[knots3], knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=None, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('dist_to_fly', [data[4,:]], ord=order, knots=[knots4], knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=None, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('lin_acc', [data[5,:]], ord=order, knots=[knots5], knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=None, lam=None,penalty_type='der',der=2)
    
    sm_handler.add_smooth('ang_acc', [data[6,:]], ord=order, knots=[knots6], knots_num=size_knots, perc_out_range=0.0,
                          is_cyclic=[False],  trial_idx=None, lam=None,penalty_type='der',der=2)
    
    var_list = ['t_since_firefly', 't_since_move', 'density', 'dist_to_fly', 'lin_acc', 'ang_acc']#,'1d_var2']#,'1d_var3']#,'1d_var4','1d_var5','1d_var6']


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
        
    
    f1y, f1y_min, f1y_max, la1 = full.smooth_compute([xx1], 't_since_firefly') # make sure xx matches
    f2y, f2y_min, f2y_max, la2 = full.smooth_compute([xx1], 't_since_move')
    f3y, f3y_min, f3y_max, la3 = full.smooth_compute([xx3], 'density')
    f4y, f4y_min, f4y_max, la4 = full.smooth_compute([xx4], 'dist_to_fly')
    f5y, f5y_min, f5y_max, la5 = full.smooth_compute([xx5], 'lin_acc')
    f6y, f6y_min, f6y_max, la6 = full.smooth_compute([xx6], 'ang_acc')
    #%%
    plt.close('all')
    '''plt.figure()
    plt.title("log")
    plt.subplot(161)
    plt.title("function 1")
    plt.plot(np.log(xx1), data[1,:],color='g',label='f1')
    plt.plot(plt.log(xx1), f1y, 'k-', label='model')
    plt.fill_between(plt.log(xx1), f1y_min,f1y_max)
    plt.subplot(162)
    plt.title("function 2")
    plt.plot(plt.log(xx2), curr2(xx2),color='g',label='f2')
    plt.plot(plt.log(xx2), f2y, 'k-', label='model')
    plt.fill_between(plt.log(xx2), f2y_min, f2y_max)
    plt.subplot(163)
    plt.title("function 3")
    plt.plot(plt.log(xx3), curr3(xx3),color='g',label='f3')
    plt.plot(plt.log(xx3), f3y, 'k-', label='model')
    plt.fill_between(plt.log(xx3), f3y_min, f3y_max)
    plt.subplot(164)
    plt.title("function 4")
    plt.plot(plt.log(xx4), np.log(curr4(xx4)),color='g',label='f4')
    plt.plot(plt.log(xx4), f4y, 'k-', label='model')
    plt.fill_between(plt.log(xx4), f4y_min, f4y_max)
    plt.subplot(165)
    plt.title("function 5")
    plt.plot(plt.log(xx5), curr5(xx5),color='g',label='f5')
    plt.plot(plt.log(xx5), f5y, 'k-', label='model')
    plt.fill_between(plt.log(xx5), f5y_min, f5y_max)
    plt.subplot(166)
    plt.title("function 6")
    plt.plot(plt.log(xx6), curr6(xx6),color='g',label='f6')
    plt.plot(plt.log(xx6), f6y, 'k-', label='model')
    plt.fill_between(plt.log(xx6), f6y_min, f6y_max)
    plt.legend()'''
    
    
    plt.figure()
    
    plt.title("R2: " + str(reduced.variable_expl_variance[0]['pseudo-R2'])+\
              "Variance expl: " +str(reduced.variable_expl_variance[0]['var_expl']))
    plt.subplots(1,6, sharey='all')
    plt.subplot(161)
    plt.title("t_since_firefly")
    #plt.plot(np.sort(data[1,:]),color='g',label='f1')
    plt.plot(xx2, f1y, 'k-', label='model')
    plt.fill_between(xx2, f1y_min,f1y_max)

    plt.subplot(162)
    plt.title("t_since_move")
    #plt.plot(np.sort(data[2,:]),color='g',label='f2')
    plt.plot(xx2, f2y, 'k-', label='model')
    plt.fill_between(xx2, f2y_min,f2y_max)

    plt.subplot(163)
    plt.title("density")
    #plt.plot(xx3, curr3(xx3),color='g',label='f3')
    plt.plot(xx3, f3y, 'k-', label='model')
    plt.fill_between(xx3, f3y_min,f3y_max)
    plt.subplot(164)
    plt.title("dist_to_fly")
    #plt.plot(xx4, np.log(curr4(xx4)),color='g',label='f4')
    plt.plot(xx4, f4y, 'k-', label='model')
    plt.fill_between(xx4, f4y_min,f4y_max)
    plt.subplot(165)
    plt.title("lin_acc")
    #plt.plot(xx5, curr5(xx5),color='g',label='f5')
    plt.plot(xx5, f5y, 'k-', label='model')
    plt.fill_between(xx5, f5y_min,f5y_max)
    plt.subplot(166)
    plt.title("ang_acc")
    #plt.plot(xx6, curr6(xx6),color='g',label='f6')
    plt.plot(xx6, f6y, 'k-', label='model')
    plt.fill_between(xx6, f6y_min,f6y_max)
    '''plt.subplot(177)
    plt.title("Function 7")
    plt.plot(xx7, curr7(xx7),color='g',label='f7')
    plt.plot(xx7, f7y, 'k-', label='model')
    plt.fill_between(xx7, f7y_min,f7y_max)'''

    #%%
    return [f1y, f2y, f3y, f4y, f5y, f6y]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def asses_data(functions, test_data):
    '''x0_r = [np.min(data[0,:]),np.max(data[0,:])]
    x1_r = [np.min(data[1,:]),np.max(data[1,:])]
    x2_r = [np.min(data[2,:]),np.max(data[2,:])]
    x3_r = [np.min(data[3,:]),np.max(data[3,:])]
    x4_r = [np.min(data[4,:]),np.max(data[4,:])]
    x5_r = [np.min(data[5,:]),np.max(data[5,:])]
    x6_r = [np.min(data[6,:]),np.max(data[6,:])]
    
    xx1 = (np.max(data[1,:])-np.min(data[1,:]))/100
    xx2 = (np.max(data[2,:])- np.min(data[2,:]))/100
    xx3 = (np.max(data[3,:])-np.min(data[3,:])-.004)/100
    xx4 = (np.max(data[4,:])-np.min(data[4,:]))/100
    xx5 = (np.max(data[5,:])-np.min(data[5,:]))/100
    xx6 = (np.max(data[6,:])-np.min(data[6,:]))/100'''
    lowers = [-2.494040012359619, -2.2800000000000002, 0.0001, 6.939641469616705, -72.57864379882812, -83.80699920654297]
    xx1 = (3.629159927368164 + 2.494040012359619)/100
    xx2 = (4.296 +2.2800000000000002)/100
    xx3 = (0.0051 + 0.00011)/100
    xx4 = (466.23953378543354 - 6.939641469616705)/100
    xx5 = (200.83282470703125 +72.57864379882812)/100
    xx6 = (71.20767974853516 + 83.80699920654297)/100
    
    xxs = [xx1, xx2, xx3, xx4, xx5, xx6]
    
    
    # get negative vakues to work
    val = np.zeros((len(xxs)+1, len(test_data[0])))
    for i in [1, 2,3, 4,5,6]:#len(test_data)):
        '''print("test data")
        print(test_data[i])
        print(i)
        print(xxs[i-1])'''
        # this is wrong
        inds = np.rint((test_data[i]-lowers[i-1])/xxs[i-1]).astype(int)
        #np.rint((test_data[i]-lowers[i-1]/xxs[i-1])+(lowers[i-1]/xxs[i-1])*-1).astype(int) # minus lower bound!!!
        '''print("inds")
        print(inds)
        print("funcs")
        print(functions[i-1])
        print(functions[i-1][inds])'''
        temp = functions[i-1][inds]
        val[i,:] = temp

    return val
    
                       
if __name__ == '__main__':
    #oldm()
    print("========is gammmmmmm========")
    #ids, x1, y = isabel_data()
    #run_pgam(ids, x1, y)
    #meth3_arbitrary()
    #meth3()
    #meth3_subsample()
    plt.figure()
    m53s100 = Trial('/Users/isabelgaron/anaconda3/envs/firefly/proj_files/monkey/m53s114.mat')
    final = []
    num_nobs = 0
    num_s = 0
    num_f=0
    # TODO - leave one out check on each variable, labeling axis, at different input windows
    # also confusion matrices, also chekc when reward occurs versus when motion dtops
    # 200, .75, 2786 trials
    for i in range(1, 300):#.75, .9
    #cheating 1, .9, works with .75, 1, and 1, .75
    #starts working qt 60% of total trial, as low as .3, dcsles evenly, dont need to rerun
    # question is how early separation emerges
        samp_s = construct_bgam_downsample_backup(m53s100, m53s100.successes[i], .5, 17)
        samp_f = construct_bgam_downsample_backup(m53s100, m53s100.failures[i], .5, 17)
        final.append(samp_s)
        final.append(samp_f)
        num_nobs+=len(samp_s[0])
        num_nobs+=len(samp_f[0])
        num_s+=len(samp_s[0])
        num_f+=len(samp_f[0])
    print("Failures: " + str(num_f))
    print("Successes: " + str(num_s))
    data = np.hstack(final)
    
    funcs = actual_data(data)
    
    conf_mat = np.zeros((2,2))
    
    plt.figure()
    for i in range(0, 44):

        downsample_s = construct_bgam_downsample_backup(m53s100, m53s100.successes[300+i], .8, 17)
        res1 = asses_data(funcs, downsample_s)
        x1= np.arange(0, len(res1[1]))*20/len(res1[1])
        # time warp so compoletely aligned
        downsample_f =  construct_bgam_downsample_backup(m53s100, m53s100.failures[300+i], .8, 17)
        res2 = asses_data(funcs,downsample_f)
        x2= np.arange(0, len(res2[1]))*20/len(res2[1])
        sum1 =  np.sum(res1, axis=0)
        sum2 = np.sum(res2, axis=0)
        if sum1[-1] >0:
            conf_mat[0,0] +=1
        else:
            conf_mat[0,1] +=1
        if sum2[-1] <0:
            conf_mat[1,0] +=1
        else:
            conf_mat[1,1] +=1
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
        ax.set_xticklabels(np.around(np.linspace(0, 99, 10)))
    
        
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
        #plt.legend()
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
    plt.show()
    #knots6 = np.linspace(x6_r[0],x6_r[1],size_knots)
    #knots6 = np.hstack(([knots6[0]]*(order-1), knots6, [knots6[-1]]*(order-1)))
    
    