#%%

import numpy as np
import time
import pickle
from numpy import newaxis as na
import utm

# import LO-GA
from input_folder.loga.loga_par_matlab_v2 import Boundaries
from input_folder.loga.loga_par_matlab_v2 import LayoutOptimizationGA_MO

# import obj function wrappers
from input_folder.obj_funcs_v4 import ObjFunc_HKNscaled
from input_folder.obj_funcs_v4 import ObjFunc_HKNscaled_GY

from input_folder.utils.wflop_utils import min_distance

if __name__ == '__main__':

    # extract HKN data
    with open(f'input_folder/HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_boundaries_x = HKN_data['hkn_boundaries_x']
    hkn_boundaries_y = HKN_data['hkn_boundaries_y']
    hkn_wt_x = HKN_data['hkn_wt_x']
    hkn_wt_y = HKN_data['hkn_wt_y']
    diameter = 283.2
    n_wt = len(hkn_wt_x)

    # scale HKN data - boundaries and initial positions
    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]
    diameter_hkn = 200.
    hkn_wt_x_scaled = x_sub + (hkn_wt_x-x_sub)*(diameter/diameter_hkn)
    hkn_wt_y_scaled = y_sub + (hkn_wt_y-y_sub)*(diameter/diameter_hkn)
    hkn_boundaries_x_scaled = x_sub + (hkn_boundaries_x-x_sub)*(diameter/diameter_hkn)
    hkn_boundaries_y_scaled = y_sub + (hkn_boundaries_y-y_sub)*(diameter/diameter_hkn)
    boundaries = Boundaries([hkn_boundaries_x_scaled],[hkn_boundaries_y_scaled])

    # optimization parameters
    n_pop = 350                         # size of the population
    n_gen = 800                        # number of generations
    n_keep_parents = int(0.3*n_pop)     # number of parents kept from the previous generation
    p_s_min = 0.7                       # min percentage of selection
    p_s_max = 0.7                       # max percentage of selection
    p_m_min = 0.1                       # min percentage of mutation
    p_m_max = 0.3                       # max percentage of mutation
    s_m_min = 0*diameter                # min step of mutation
    s_m_max = 3*diameter                # max step of mutation
    d_limit = 1*diameter                # distance limit for turbine association during crossover
    p_m_array = np.flip(np.linspace(p_m_min,p_m_max,n_gen))
    s_m_array = s_m_min+(s_m_max-s_m_min)*(np.exp(-5*np.linspace(0,1,n_gen)))

    # extract initial population
    with open(f'input_folder/initial_layouts/initial_pop_v1.pkl', 'rb') as f:
        data_init_pop = pickle.load(f)
    x_mat_initial = data_init_pop['x_mat_initial'][:n_pop,:]
    y_mat_initial = data_init_pop['y_mat_initial'][:n_pop,:]

    # define objective function wrapper
    obj_func = ObjFunc_HKNscaled(n_cpu=1,                   # select the number of CPUs in case of parallel execution
                                 parallel_execution=False,  # select sequential (seto to False) or parallel (set to True) programming
                                 obj=['IOE','LCOE'],        # select objectives
                                 maximize=[False,False])    # define whether the goal is maximize (set to True) or minimize (set to False) th eobjectives
    

    # create optimization object (MULTI-OBJECTIVE)
    layout_optimization = LayoutOptimizationGA_MO(obj_func,
                                                  n_wt,
                                                  boundaries,
                                                  n_obj=2,      # match the number of objectives deifned in the wrapper
                                                  n_gen=n_gen,
                                                  n_pop=n_pop,
                                                  perc_s=0.7,
                                                  perc_m=p_m_array,
                                                  step_m=s_m_array,
                                                  distance_limit=d_limit,
                                                  full_pop_evaluation=True,
                                                  x_mat_initial=x_mat_initial,
                                                  y_mat_initial=y_mat_initial,
                                                  gen_to_f = True
                                                  )

    
    # run optmization
    print('Multi-objective optimization')
    t_1 = time.time()
    x_mat_output_pareto,y_mat_output_pareto,fitness_val_output_pareto = layout_optimization.optimize()
    t_2 = time.time()
    print(f'Total time: {t_2-t_1}')


    # calculate min distance to check convergence (must be satisfied by all layouts in the Pareto front)
    min_d_D_opt = np.inf
    for i in np.arange(x_mat_output_pareto.shape[0]):
        min_d_D_temp = min_distance(x_mat_output_pareto[i,:],y_mat_output_pareto[i,:],diameter)
        if min_d_D_temp<min_d_D_opt:
            min_d_D_opt = min_d_D_temp


    # postprocess optimization results

    data_results = {}
    data_results['x_opt_Pareto'] = x_mat_output_pareto
    data_results['y_opt_Pareto'] = y_mat_output_pareto
    data_results['x_mat_initial']  = x_mat_initial
    data_results['y_mat_initial'] = y_mat_initial
    data_results['f_opt_Pareto'] = fitness_val_output_pareto

    data_results['converged_flag'] = 4.<=min_d_D_opt
    data_results['min_d_D_opt'] = min_d_D_opt

    parallel_evaluation = False
    n_cpu_evaluation = 1
    f_wrapper_list = [ObjFunc_HKNscaled(obj=['AEP'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled(obj=['AEPnet'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled(obj=['LCOE'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled(obj=['IOE'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled_GY(obj=['AEP'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled_GY(obj=['AEPnet'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled_GY(obj=['LCOE'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ObjFunc_HKNscaled_GY(obj=['IOE'],maximize=[True],n_cpu=n_cpu_evaluation,parallel_execution=parallel_evaluation),
                      ]

    f_name_list = ['AEP',
                   'AEP_net',
                   'LCOE',
                   'IOE',
                   'AEP_GY',
                   'AEPnet_GY',
                   'LCOE_GY',
                   'IOE_GY',
                   ]

    print('Postprocessing...')
    t_1 = time.time()
    for name,wrapper in zip(f_name_list,f_wrapper_list):
        f_eval = wrapper
        data_results[name] = np.sum(f_eval(x_mat_output_pareto,y_mat_output_pareto),axis=(1))
    t_2 = time.time()
    print(f'Total time: {t_2-t_1}')

    # store data
    with open('data_HKNscaled_example.pkl', 'wb') as f:
        pickle.dump(data_results, f)





















