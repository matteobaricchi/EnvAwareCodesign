# import main packages
import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
from numpy import newaxis as na

# import tools and data
from input_folder.utils.wflop_utils import filter_turbines

# import geometric yaw
from input_folder.geometric_yaw.geometric_yaw import calculate_geomYaw_ExpCorr

# cabling optimization
import networkx as nx
from optiwindnet.api import WindFarmNetwork, HGSRouter, EWRouter

from pathos.multiprocessing import ProcessPool



class AEPCalculator():

    def __init__(self,**kwargs):

        self.wfm = kwargs.get('wfm')
        self.wd_array = kwargs.get('wd_array')
        self.ws_array = kwargs.get('ws_array')
        self.ws_rated = kwargs.get('ws_rated',None)
        self.wind_turbine = kwargs.get('wind_turbine',None)
        self.use_geomYaw = kwargs.get('use_geomYaw',False)
        self.use_inputYaw = kwargs.get('use_inputYaw',False)
        self.yaw_input = kwargs.get('yaw_input',False)

    def __call__(self,x,y):

        # calculate yaw angles
        if self.use_geomYaw:
            yaw_temp = calculate_geomYaw_ExpCorr(x,y,self.wd_array,self.ws_array,self.ws_rated,self.wind_turbine,self.wfm)
        elif self.use_inputYaw:
            yaw_temp = self.yaw_input
        else:
            yaw_temp = 0

        # compute AEP
        simres = self.wfm(x,y,wd=self.wd_array,ws=self.ws_array,yaw=yaw_temp,tilt=0)
        aep = simres.aep().sum().values
        aep_per_turbine = np.sum(simres.aep().values,axis=(1,2))
        aep_no_wake = simres.aep(with_wake_loss=False).sum().values
        power_ilk = simres.Power.values

        #AEP_per_turbine = np.sum(sim_res.aep().values,axis=(1,2))
        #detect_aep = sim_res.aep(normalize_probabilities=False).sum().values
        #detect_aep_nowake = sim_res.aep(normalize_probabilities=False,with_wake_loss=False).sum().values
        #detect_power = sim_res.Power.values

        # save data

        ds = xr.Dataset(
            {
                "x": ('i', x),
                "y": ('i', y),
                "wd_array": ('l', self.wd_array),
                "ws_array": ('k', self.ws_array),
                "power_ilk": (('i','l','k'), power_ilk),
                "AEP_per_turbine": ('i', aep_per_turbine),
                "AEP": aep,
                "AEP_no_wake": aep_no_wake,
            },
            coords={
                "i": np.arange(len(x)),
                "l": np.arange(len(self.wd_array)),
                "k": np.arange(len(self.ws_array)),
            }
        )

        return ds



class CableCalculator():

    def __init__(self,**kwargs):
        self.time_limit = kwargs.get('time_limit',0.3)
        self.x_sub = kwargs.get('x_sub')
        self.y_sub = kwargs.get('y_sub')
        cable_specs = kwargs.get('cable_specs')
        self.cables = np.array([(int(c["capacity_NrT"]), float(c["cost_€_m"])) for c in cable_specs],dtype=[("capacity", int), ("cost", float)])
        #self.x_init = kwargs.get('x_init')
        #self.y_init = kwargs.get('y_init')


    def Postprocess_Cable_Optimizer(self,wfn):
        cab_data = wfn.get_network()
        # get connection matrix
        u_fnt = []
        v_fnt = []
        fnT = wfn.G.graph.get('fnT', None)
        fnT_state = fnT is None
        for u, v in wfn.G.edges():
             u_fnt.append(u if fnT_state else fnT[u])
             v_fnt.append(v if fnT_state else fnT[v])
        
        # Postprocess to deal with obstacles
        # Step 1: Build undirected graph
        H = nx.Graph()
        length_lookup = {}
        type_lookup = {}
        load_lookup = {}
        idx_ct = 4
        substation = -1
        for row in cab_data:
            a, b = int(row[0]), int(row[1])
            length = row[2]
            ct = row[idx_ct]
            l = row[3]
            H.add_edge(a, b)
            length_lookup[(a, b)] = length
            length_lookup[(b, a)] = length  # undirected
            type_lookup[(a, b)] = ct
            type_lookup[(b, a)] = ct  # undirected
            load_lookup[(a, b)] = l
            load_lookup[(b, a)] = l  # undirected
        # Step 2: Traverse from substation
        visited = set()
        edges = []
        lengths = []
        types = []
        loads = []
        def dfs(node, parent=None):
            visited.add(node)
            for neighbor in H.neighbors(node):
                if neighbor not in visited:
                    # Reverse direction to point to substation
                    edges.append((neighbor, node))
                    lengths.append(length_lookup[(neighbor, node)])
                    types.append(type_lookup[(neighbor, node)])
                    loads.append(load_lookup[(neighbor, node)])
                    dfs(neighbor, node)
        
        dfs(substation)
        
        # Now `edges` contains tuples of form (source, target) pointing to substation
        source_nodes = [int(src) for src, dst in edges]
        target_nodes = [int(dst) for src, dst in edges]
        lengths = [float(l) for l in lengths]
        types = [int(t) for t in types]
        loads = [int(t) for t in loads]
        
        # Step 3: Combine and sort by source
        combined = sorted(zip(source_nodes, target_nodes, loads), key=lambda x: x[0])
        s, t, load_sorted = zip(*combined)
        return t, [int(x) for x in u_fnt], [int(x) for x in v_fnt], np.array([cab_data['src'].tolist(), cab_data['tgt'].tolist()])


    def __call__(self,x,y):
        # First try metaheuristic solver, if it fails, go with heuristics.
        try:
            wfn = WindFarmNetwork(turbinesC=np.column_stack((x,y)), substationsC=np.column_stack((self.x_sub, self.y_sub)), cables=self.cables, router=HGSRouter(time_limit=self.time_limit))
            wfn.optimize()
        except:
            #x_init_fil = self.x_init[:len(x)]
            #y_init_fil = self.y_init[:len(y)]
            print("Metaheuristic solver failed, trying with heuristics...")
            wfn = WindFarmNetwork(turbinesC=np.column_stack((x,y)), substationsC=np.column_stack((self.x_sub, self.y_sub)), cables=self.cables, router=EWRouter())
            wfn.optimize()
        TurCon, u, v, cab_data = self.Postprocess_Cable_Optimizer(wfn)

        ds = xr.Dataset(
            {
                "x": ('i', x),
                "y": ('i', y),
                "TurCon": ('c', np.array(TurCon)),
                "u": ('c', u),
                "v": ('c', v),
                "cab_data": (('c_type','c'), cab_data),
            },
            coords={
                "i": np.arange(len(x)),
                "c": np.arange(len(TurCon)),
                "c_type": np.arange(cab_data.shape[0]),
            }
        )

        return ds



class WaterDepthCalculator():

    def __init__(self,**kwargs):
        self.site_x_grid = kwargs.get('site_x_grid')
        self.site_y_grid = kwargs.get('site_y_grid')
        self.site_bathymetry_grid = kwargs.get('site_bathymetry_grid')

    def __call__(self,x,y):
        site_x_coord = self.site_x_grid[:,0]
        site_y_coord = self.site_y_grid[0,:]
        interp_function = RegularGridInterpolator((site_x_coord,site_y_coord),self.site_bathymetry_grid)
        water_depth = interp_function((x,y))    
        return water_depth

        
#%%
#
#class EcoEnvCalculator():
#
#    def __init__(self,**kwargs):
#        self.matlab_eng = kwargs.get('matlab_eng')
#
#
#    def __call__(self,x,y,ds_aep,ds_cable,water_depth_per_turbine):
#
#        # extract input for DETECT
#        aep_input = ds_aep["AEP"].values
#        aep_no_wake = ds_aep["AEP_no_wake"].values
#        power_ilk = ds_aep["power_ilk"].values
#        wd_array = ds_aep["wd_array"].values
#        aep_per_turbine = ds_aep["AEP_per_turbine"].values
#
#        TurCon = tuple(ds_cable["TurCon"].values)
#        u = ds_cable["u"].values.tolist()
#        v = ds_cable["v"].values.tolist()
#        cab_data = ds_cable["cab_data"].values
#
#        # run DETECT
#        print(TurCon)
#        print(type(TurCon))
#        print(len(TurCon))
#        lcoe, ioe, AnnualCosts, AnnualEmissions, AEPnetFactor = self.matlab_eng.RunDetectWithoutPython(x, y, -water_depth_per_turbine, 'precalculated', aep_input, aep_no_wake, power_ilk, 'precalculated', TurCon, u, v, cab_data, float(np.diff(wd_array)[0]), nargout=5)
#        AnnualCosts_per_turbine = np.array(AnnualCosts).flatten()
#        AnnualEmissions_per_turbine = np.array(AnnualEmissions).flatten()
#        
#        # Apply AEP net factor (=losses due to availability, load-dependent powertrain, load-dependent array-cabling, export cabling, performance, transformers in substations)
#        aep_per_turbine = aep_per_turbine*AEPnetFactor
#        aep = np.sum(aep_per_turbine)
#                
#        # calculate obj functions per turbine
#        eps = 1e-6
#        n_wt = len(x)
#        EUR_cost = np.sum(AnnualCosts_per_turbine/1000)/np.sum(aep_per_turbine)/n_wt
#        CO2_cost = np.sum(AnnualEmissions_per_turbine/1000)/np.sum(aep_per_turbine)/n_wt
#        EUR_cost_per_turbine = np.sum(AnnualCosts_per_turbine/1000)/np.sum(aep_per_turbine)/n_wt + eps*(AnnualCosts_per_turbine/1000/aep_per_turbine)
#        CO2_cost_per_turbine = np.sum(AnnualEmissions_per_turbine/1000)/np.sum(aep_per_turbine)/n_wt + eps*(AnnualEmissions_per_turbine/1000/aep_per_turbine)
#
#        # save data
#
#        ds = xr.Dataset(
#            {
#                "x": ('i', x),
#                "y": ('i', y),
#                "AEP_per_turbine": ('i', aep_per_turbine),
#                "LCOE_per_turbine": ('i', EUR_cost_per_turbine),
#                "IOE_per_turbine": ('i', CO2_cost_per_turbine),
#                "AEP": aep,
#                "LCOE": EUR_cost,
#                "IOE": CO2_cost,
#            },
#            coords={
#                "i": np.arange(len(x)),
#            }
#        )
#        
#        return ds
#        
#
#
#    def evaluate_population():
#        ...
#
#
#
#
#class EvalFunction_XYwrapper():
#
#    def __init__(self,aep_calculator,cable_calculator,water_depth_calculator,eco_env_calculator,min_d,**kwargs):
#
#        self.aep_calculator = aep_calculator
#        self.cable_calculator = cable_calculator
#        self.water_depth_calculator = water_depth_calculator
#        self.eco_env_calculator = eco_env_calculator
#        self.min_d = min_d
#
#        # attributes for optmization
#        self.evaluate_population = kwargs.get('evaluate_population',False)
#        self.parallel_execution = kwargs.get('parallel_execution',False)
#        self.n_cpu = kwargs.get('n_cpu',None)
#        self.value_per_turbine = kwargs.get('value_per_turbine',False)
#        self.ouput = kwargs.get('output',['AEP','EUR','CO2'])
#
#
#
#    def python_wrapper(self,x,y):
#
#        # filter turbines
#        x_fil,y_fil,ind_keep = filter_turbines(x,y,self.min_d)
#
#        # run simulation (python)
#        ds_aep = self.aep_calculator(x_fil,y_fil)
#        ds_cable = self.cable_calculator(x_fil,y_fil)
#        water_depth_per_turbine = self.water_depth_calculator(x_fil,y_fil)
#
#        # save output
#        dict_output = {
#            'x_fil' : x_fil,
#            'y_fil' : y_fil,
#            'ind_keep' : ind_keep,
#            'ds_aep' : ds_aep,
#            'ds_cable' : ds_cable,
#            'water_depth_per_turbine' : water_depth_per_turbine,
#        }
#
#        return dict_output
#
#
#    def __call__(self,x,y):
#        
#        if self.evaluate_population:
#
#            # extract dimension
#            if len(x.shape)<2:
#                raise TypeError('Population not in the right input format: check dimensions -> x: (n_pop,n_wt)')
#            n_pop = x.shape[0]
#            n_wt = x.shape[1]
#
#            # initialize results
#            EUR_cost_per_turbine = np.ones((n_pop,n_wt))*1e20
#            CO2_cost_per_turbine = np.ones((n_pop,n_wt))*1e20
#            aep_per_turbine = np.ones((n_pop,n_wt))*1e-10
#
#            # run python wrapper
#            if self.parallel_execution:
#                with ProcessPool(self.n_cpu) as pool:
#                    results =  pool.map(self.python_wrapper,list(x),list(y))
#            else:
#                results = map(self.python_wrapper,list(x),list(y))
#
#            # unpack results
#            res_list = list(results)
#            x_fil_list = [None]*n_pop
#            y_fil_list = [None]*n_pop
#            ind_keep_list = [None]*n_pop
#            ds_aep_list = [None]*n_pop
#            ds_cable_list = [None]*n_pop
#            water_depth_per_turbine_list = [None]*n_pop
#            for n in np.arange(n_pop):
#                x_fil_list[n] = res_list[n]['x_fil']
#                y_fil_list[n] = res_list[n]['y_fil']
#                ind_keep_list[n] = res_list[n]['ind_keep']
#                ds_aep_list[n] = res_list[n]['ds_aep']
#                ds_cable_list[n] = res_list[n]['ds_cable']
#                water_depth_per_turbine_list[n] = res_list[n]['water_depth_per_turbine']
#
#
#            #fitness_val_mat_initial = np.array(list(results)).transpose(0,2,1)
#
#
#            return ds_aep_list
#
#
#
#
#        else:
#
#            # initialize results
#            EUR_cost_per_turbine = np.ones(len(x))*1e20
#            CO2_cost_per_turbine = np.ones(len(x))*1e20
#            aep_per_turbine = np.ones(len(x))*1e-10
#
#            # filter turbines
#            x_fil,y_fil,ind_keep = filter_turbines(x,y,self.min_d)
#
#            # run simulation (python)
#            ds_aep = self.aep_calculator(x_fil,y_fil)
#            ds_cable = self.cable_calculator(x_fil,y_fil)
#            water_depth_per_turbine = self.water_depth_calculator(x_fil,y_fil)
#
#            # run simulation (matlab)
#            ds_ecoenv = self.eco_env_calculator(x_fil,y_fil,ds_aep,ds_cable,water_depth_per_turbine)
#
#            # save results
#            aep_per_turbine[ind_keep] = ds_ecoenv["AEP_per_turbine"].values
#            EUR_cost_per_turbine[ind_keep] = ds_ecoenv["LCOE_per_turbine"].values
#            CO2_cost_per_turbine[ind_keep] = ds_ecoenv["IOE_per_turbine"].values
#
#            ds = xr.Dataset(
#                {
#                    "x": ('i', x),
#                    "y": ('i', y),
#                    "AEP_per_turbine": ('i', aep_per_turbine),
#                    "LCOE_per_turbine": ('i', EUR_cost_per_turbine),
#                    "IOE_per_turbine": ('i', CO2_cost_per_turbine),
#                    "AEP": ds_ecoenv["AEP"].values,
#                    "LCOE": ds_ecoenv["LCOE"].values,
#                    "IOE": ds_ecoenv["IOE"].values,
#                },
#                coords={
#                    "i": np.arange(len(x)),
#                }
#            )
#
#            return ds
#

#%%




class PopEval_python_XYwrapper():

    def __init__(self,aep_calculator,cable_calculator,water_depth_calculator,min_d,**kwargs):

        self.aep_calculator = aep_calculator
        self.cable_calculator = cable_calculator
        self.water_depth_calculator = water_depth_calculator
        self.min_d = min_d

        # attributes for optmization
        self.parallel_execution = kwargs.get('parallel_execution',False)
        self.n_cpu = kwargs.get('n_cpu',None)


    def func_wrapper(self,x,y):

        # filter turbines
        x_fil,y_fil,ind_keep = filter_turbines(x,y,self.min_d)

        # run simulation (python)
        ds_aep = self.aep_calculator(x_fil,y_fil)
        ds_cable = self.cable_calculator(x_fil,y_fil)
        water_depth_per_turbine = self.water_depth_calculator(x_fil,y_fil)

        # save output
        dict_output = {
            'x_fil' : x_fil,
            'y_fil' : y_fil,
            'ind_keep' : ind_keep,
            'ds_aep' : ds_aep,
            'ds_cable' : ds_cable,
            'water_depth_per_turbine' : water_depth_per_turbine,
        }

        return dict_output


    def __call__(self,x_mat,y_mat):
        
        # extract dimension
        if len(x_mat.shape)<2:
            raise TypeError('Population not in the right input format: check dimensions -> x: (n_pop,n_wt)')
        n_pop = x_mat.shape[0]
        n_wt = x_mat.shape[1]

        ## initialize results
        #EUR_cost_per_turbine = np.ones((n_pop,n_wt))*1e20
        #CO2_cost_per_turbine = np.ones((n_pop,n_wt))*1e20
        #aep_per_turbine = np.ones((n_pop,n_wt))*1e-10

        # run python wrapper
        if self.parallel_execution:
            with ProcessPool(self.n_cpu) as pool:
                results =  pool.map(self.func_wrapper,list(x_mat),list(y_mat))
        else:
            results = map(self.func_wrapper,list(x_mat),list(y_mat))

        # unpack results
        res_list = list(results)
        x_fil_list = [None]*n_pop
        y_fil_list = [None]*n_pop
        ind_keep_list = [None]*n_pop
        ds_aep_list = [None]*n_pop
        ds_cable_list = [None]*n_pop
        water_depth_per_turbine_list = [None]*n_pop
        for n in np.arange(n_pop):
            x_fil_list[n] = res_list[n]['x_fil']
            y_fil_list[n] = res_list[n]['y_fil']
            ind_keep_list[n] = res_list[n]['ind_keep']
            ds_aep_list[n] = res_list[n]['ds_aep']
            ds_cable_list[n] = res_list[n]['ds_cable']
            water_depth_per_turbine_list[n] = res_list[n]['water_depth_per_turbine']

        # save output
        dict_python_output = {
            'x_fil_list' : x_fil_list,
            'y_fil_list' : y_fil_list,
            'ind_keep_list' : ind_keep_list,
            'ds_aep_list' : ds_aep_list,
            'ds_cable_list' : ds_cable_list,
            'water_depth_per_turbine_list' : water_depth_per_turbine_list,
        }

        return dict_python_output



class PopEval_matlab_XYwrapper():

    def __init__(self,matlab_eng,**kwargs):

        self.matlab_eng = matlab_eng

        # attributes for optmization
        self.parallel_execution = kwargs.get('parallel_execution',False)
        self.n_cpu = kwargs.get('n_cpu',None)


    def __call__(self,dict_python_output):

        # unpack python output
        x_fil_list = dict_python_output['x_fil_list']
        y_fil_list = dict_python_output['y_fil_list']
        ind_keep_list = dict_python_output['ind_keep_list']

        n_pop = len(x_fil_list)
        z_fil_list = [None]*n_pop
        aep_input_list = [None]*n_pop
        aep_no_wake_list = [None]*n_pop
        power_ilk_list = [None]*n_pop
        TurCon_list = [None]*n_pop
        u_list = [None]*n_pop
        v_list = [None]*n_pop
        cab_data_list = [None]*n_pop
        wd_step_list = [None]*n_pop
        for n in np.arange(n_pop):
            z_fil_list[n] = -dict_python_output['water_depth_per_turbine_list'][n]
            aep_input_list[n] = dict_python_output['ds_aep_list'][n]['AEP'].values
            aep_no_wake_list[n] = dict_python_output['ds_aep_list'][n]['AEP_no_wake'].values
            power_ilk_list[n] = dict_python_output['ds_aep_list'][n]['power_ilk'].values
            wd_step_list[n] = float(np.diff(dict_python_output['ds_aep_list'][n]['wd_array'].values)[0])
            TurCon_list[n] = dict_python_output['ds_cable_list'][n]['TurCon'].values
            u_list[n] = dict_python_output['ds_cable_list'][n]['u'].values
            v_list[n] = dict_python_output['ds_cable_list'][n]['v'].values
            cab_data_list[n] = dict_python_output['ds_cable_list'][n]['cab_data'].values

        # process these data with the matlab engine running DETECT for the all the elements in the lists
        lcoe_list,ioe_list,cost_per_turbine_list,emission_per_turbine_list,aep_net_factor_list,add_output_list = self.matlab_eng.evaluate_population(x_fil_list,
                                                                                                                                     y_fil_list,
                                                                                                                                     z_fil_list,
                                                                                                                                     aep_input_list,
                                                                                                                                     aep_no_wake_list,
                                                                                                                                     power_ilk_list,
                                                                                                                                     wd_step_list,
                                                                                                                                     TurCon_list,
                                                                                                                                     u_list,
                                                                                                                                     v_list,
                                                                                                                                     cab_data_list,
                                                                                                                                     self.parallel_execution,
                                                                                                                                     self.n_cpu,
                                                                                                                                     nargout=6)
        # postprocess matlab variables
        for n in np.arange(n_pop):
            cost_per_turbine_list[n] = np.array(cost_per_turbine_list[n]).reshape(-1)
            emission_per_turbine_list[n] = np.array(emission_per_turbine_list[n]).reshape(-1)
            add_output_list[n] = np.array(add_output_list[n]).reshape(-1)
            
        # save output
        dict_matlab_output = {
            'lcoe_list' : lcoe_list,
            'ioe_list' : ioe_list,
            'cost_per_turbine_list' : cost_per_turbine_list,
            'emission_per_turbine_list' : emission_per_turbine_list,
            'aep_net_factor_list' : aep_net_factor_list,
            'add_output_list' : add_output_list
        }

        return dict_matlab_output




class ObjFunction_XYwrapper():

    def __init__(self,python_xy_wrapper,matlab_xy_wrapper,**kwargs):
        
        self.python_xy_wrapper = python_xy_wrapper
        self.matlab_xy_wrapper = matlab_xy_wrapper

        self.value_per_turbine = kwargs.get('value_per_turbine',False)
        self.output_keys = kwargs.get('output_keys',['AEP','LCOE','IOE'])
        self.n_obj = len(self.output_keys)
        self.maximize = kwargs.get('maximize',[True]*self.n_obj)



    def __call__(self,x_mat,y_mat):

        # initialize results
        n_pop,n_wt = x_mat.shape
        aep_array = np.zeros((n_pop))
        aep_net_array = np.zeros((n_pop))
        lcoe_array = np.zeros((n_pop))
        ioe_array = np.zeros((n_pop))
        aep_per_turbine_mat = np.ones((n_pop,n_wt))*1e-10
        aep_net_per_turbine_mat = np.ones((n_pop,n_wt))*1e-10
        cost_per_turbine_mat = np.ones((n_pop,n_wt))*1e20
        emission_per_turbine_mat = np.ones((n_pop,n_wt))*1e20
        eps = 1e-6
        # additional variables (for postprocessing)
        cab_emissions_mat = np.zeros((n_pop))
        mp_emissions_mat =  np.zeros((n_pop))
        tow_emissions_mat = np.zeros((n_pop))
        tot_emissions_mat = np.zeros((n_pop))
        cab_costs_mat     = np.zeros((n_pop))
        mp_costs_mat      = np.zeros((n_pop))
        tow_costs_mat     = np.zeros((n_pop))
        tot_cost_mat      = np.zeros((n_pop))

        # run python and matlab functions
        dict_python_output = self.python_xy_wrapper(x_mat,y_mat)
        dict_matlab_output = self.matlab_xy_wrapper(dict_python_output)

        # postprocess results
        for n in np.arange(n_pop):
            aep_array[n] = dict_python_output['ds_aep_list'][n]['AEP'].values
            aep_net_array[n] = dict_python_output['ds_aep_list'][n]['AEP'].values*dict_matlab_output['aep_net_factor_list'][n]
            lcoe_array[n] = dict_matlab_output['lcoe_list'][n]
            ioe_array[n] = dict_matlab_output['ioe_list'][n]
            ind_keep = dict_python_output['ind_keep_list'][n]
            
            cab_emissions_mat[n] = dict_matlab_output['add_output_list'][n][0]
            mp_emissions_mat[n] = dict_matlab_output['add_output_list'][n][1]
            tow_emissions_mat[n] = dict_matlab_output['add_output_list'][n][2]
            tot_emissions_mat[n] = dict_matlab_output['add_output_list'][n][3]
            cab_costs_mat[n] = dict_matlab_output['add_output_list'][n][4]
            mp_costs_mat[n] = dict_matlab_output['add_output_list'][n][5]
            tow_costs_mat[n] = dict_matlab_output['add_output_list'][n][6]
            tot_cost_mat[n] = dict_matlab_output['add_output_list'][n][7]
            
            aep_per_turbine_mat[n,ind_keep] = dict_python_output['ds_aep_list'][n]['AEP_per_turbine'].values
            aep_net_per_turbine_mat[n,ind_keep] = dict_python_output['ds_aep_list'][n]['AEP_per_turbine'].values*dict_matlab_output['aep_net_factor_list'][n]
            cost_per_turbine_mat[n,ind_keep] = dict_matlab_output['cost_per_turbine_list'][n]
            emission_per_turbine_mat[n,ind_keep] = dict_matlab_output['emission_per_turbine_list'][n]
        #lcoe_per_turbine_mat = (np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1)))[:,na] + eps*(cost_per_turbine_mat/1000/aep_net_per_turbine_mat)
        #ioe_per_turbine_mat = (np.sum(emission_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1)))[:,na] + eps*(emission_per_turbine_mat/1000/aep_net_per_turbine_mat)
        lcoe_per_turbine_mat = (np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(cost_per_turbine_mat/1000/aep_net_per_turbine_mat)
        ioe_per_turbine_mat = (np.sum(emission_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(emission_per_turbine_mat/1000/aep_net_per_turbine_mat)

        # save output
        dict_output = {
            'AEP' : aep_array,
            'LCOE' : lcoe_array,
            'IOE' : ioe_array,
            'AEP_per_turbine' : aep_per_turbine_mat,
            'LCOE_per_turbine' : lcoe_per_turbine_mat,
            'IOE_per_turbine' : ioe_per_turbine_mat,
            'Cable_emissions' : cab_emissions_mat,
            'Monopile_emissions' : mp_emissions_mat,
            'Tower_emissions': tow_emissions_mat,
            'Total_lifecycle_emissions' : tot_emissions_mat,
            'Cable_costs' : cab_costs_mat,
            'Monopile_costs' : mp_costs_mat,
            'Tower_costs' : tow_costs_mat,
            'Total_lifecycle_costs' : tot_cost_mat,
            'AEPnet' : aep_net_array
        }

        # define sign
        sign_array = np.where(np.array(self.maximize),1,-1)

        # select output
        if self.value_per_turbine:
            f_out = np.zeros((n_pop,n_wt,self.n_obj))
            for n_obj_ind in np.arange(self.n_obj):
                f_out[:,:,n_obj_ind] = dict_output[self.output_keys[n_obj_ind]+'_per_turbine']
            f_out = f_out*sign_array[na,na,:]
        else:
            f_out = np.zeros((n_pop,self.n_obj))
            for n_obj_ind in np.arange(self.n_obj):
                f_out[:,n_obj_ind] = dict_output[self.output_keys[n_obj_ind]]
            f_out = f_out*sign_array[na,:]

        # fix dimensions
        if self.value_per_turbine:
            if self.n_obj<2:
                f_out = np.reshape(f_out,(n_pop,n_wt))
        else:
            if self.n_obj<2:
                f_out = np.reshape(f_out,(n_pop))
            else:
                f_out = np.reshape(f_out,(n_pop,self.n_obj))

        return f_out







