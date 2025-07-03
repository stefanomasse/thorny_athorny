import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from network_code.utils import *
from network_code.parameters import *
from network_code.network import *
from network_code.simulations import *
from functions_for_parameter_sweeps import *
import multiprocessing
import json

def run_balance_1param(args):
    conn1, i, plastic_ee, heterogenous = args
    find_balance_1parameter(conn1, i, plastic_ee=plastic_ee, heterogenous=heterogenous, itlen=20, itmax=60)

def run_sim_1param(args):
    conn1, i, plastic_ee, heterogenous = args
    simulate_1parameter(conn1, i, plastic_ee=plastic_ee, heterogenous=heterogenous, simlen=200)

def run_balance_2param(args):
    conn1, conn2, i = args
    find_balance_2parameters(conn1, conn2, i, itlen=20, itmax=60)

def run_sim_2param(args):
    conn1, conn2, i = args
    simulate_2parameters(conn1, conn2, i, simlen=200)
    

def run_all_1param(conn1, plastic_ee=False, heterogenous=False):
    '''
    Runs the following sequence
        run_balance_1param, which is find_balance_1parameter on a multiprocessing pool
        save_conductances_1parameter
        run_sim_1param, which is simulate_1parameter on a multiprocessing pool
        open_and_group_data_1parameter
    '''
    indices = np.arange(41)
    args_list = [(conn1, i, plastic_ee, heterogenous) for i in indices]

    with multiprocessing.Pool(41) as pool:
        pool.map(run_balance_1param, args_list)
        
    save_conductances_1parameter(conn1, plastic_ee=plastic_ee, heterogenous=heterogenous)
    
    with multiprocessing.Pool(41) as pool:
        pool.map(run_sim_1param, args_list)

    if conn1 == ['tt']:
        # Connectivity tt requires a lower threshold on the population firing needed to qualify as a "peak",
        # because very little firing of a or t happens for extreme values of this connectivity
        open_and_group_data_1parameter(conn1, thr=3, plastic_ee=plastic_ee, heterogenous=heterogenous)
    else:
        open_and_group_data_1parameter(conn1, plastic_ee=plastic_ee, heterogenous=heterogenous)


def run_all_2param(conn1, conn2):
    '''
    Runs the following sequence
        run_balance_2param, which is find_balance_2parameters on a multiprocessing pool
        save_conductances_2parameterw
        run_sim_2param, which is simulate_2parameters on a multiprocessing pool
        open_and_group_data_1parameters
    '''
    indices = np.arange(82)
    args_list = [(conn1, conn2, i) for i in indices]
    
    with multiprocessing.Pool(82) as pool:
        pool.map(run_balance_2param, args_list)

    save_conductances_2parameters(conn1, conn2)

    with multiprocessing.Pool(82) as pool:
        pool.map(run_sim_2param, args_list)

    open_and_group_data_2parameters(conn1, conn2)


''' Main for the Cluster run '''

# Adjust connectivities as needed - more time efficient if different combinations are run on different computers
combos_one = ['aa','ta','at','tt']
combos_two = [['aa','tt'],['aa','at'],['aa','ta'],['ta','tt'],['ta','at'],['at','tt'],['aa','bt'],['aa','ba'],['aa','tb'],['aa','ab'],['ta','bt'],['ta','ba'],['ta','tb'],['ta','ab'],['ab','bt'],['ab','ba'],['ab','tb'],['tb','bt'],['tb','ba'],['ba','bt']]

for combo in combos_one:
    run_all_1param(combo)

for combo in combos_one:
    run_all_1param(combo, plastic_ee=True)

for combo in combos_one:
    run_all_1param(combo, heterogenous=True)

for combo1, combo2 in combos_two:
    run_all_2param(combo1,combo2)



