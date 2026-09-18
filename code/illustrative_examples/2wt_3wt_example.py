#%% Preamble
# import main packages
import numpy as np
import utm
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle
from numpy import newaxis as na

from input_folder.obj_functions_v3 import AEPCalculator,CableCalculator,WaterDepthCalculator,PopEval_python_XYwrapper,PopEval_matlab_XYwrapper,ObjFunction_XYwrapper

# import py_wake packages
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.superposition_models import SquaredSum
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.site import XRSite
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.ground_models.ground_models import Mirror
from py_wake.deficit_models.utils import ct2a_mom1d

# import matlab engine (you can uncomment this in case you only load+plot data)
import matlab.engine

#%% Inputs
# Main flags
Run = False                 # True to create data, False to load it from pickle file
hbd = False                 # True for high bathymetry case, False for low bathymetry
WritePkl = False            # To write figures into pickle file
AlignYaxis2turbine = True   # True to manually align axes of 2-turbine case
y_lim_2t = (-1.2,7.4)       # associated interval
AlignYaxis3turbine = False  # True to manually align axes of 3-turbine case
y_lim_3t = (-4.5,5)         # associated interval

#colors and font
FS1 = 8  # titles + axis labels
FS2 = 7  # tick labels + legends

color_0_shades = ['#ffce91','#f48800','#a35a00']
color_1_shades = ['#ffc1b7','#ff4426','#c31c00']
color_2_shades = ['#98bbef','#1f63cb','#144288']
color_3_shades = ['#94d9a2','#359548','#236330']
color_4_shades = ['#f2c3e6','#d94bb3','#a1227f']
color_5_shades = ['#c9c9c9','#7a7a7a','#3f3f3f']
color_6_shades = ['#fff3a6','#fbc02d','#a67c00']
color_7_shades = ['#d6b3ff','#7b2cff','#4a148c']

#%% update font to latex
if not WritePkl:
    try:
        import matplotlib.font_manager as font_manager
        font_dir = [r"input_folder\font\Serif"]
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)
        plt.rcParams.update({
            # Regular text
            "font.family": "CMU Serif",
        
            # Math text
            "mathtext.fontset": "custom",
            "mathtext.rm": "CMU Serif",
            "mathtext.it": "CMU Serif:italic",
            "mathtext.bf": "CMU Serif:bold",
            "mathtext.sf": "CMU Serif",
            "mathtext.tt": "CMU Serif",
        
            # Your existing options
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
        })
    except ValueError:
        print("Latex font not available.")

#%% Set up models
# define wind turbine
ws_turbine = np.array([0, 2, 3, 3.54953237, 4.067900771, 4.553906848, 5.006427063, 5.424415288, 5.806905228, 6.153012649, 6.461937428, 6.732965398, 6.965470002, 7.158913742, 7.312849418, 7.426921164, 7.500865272, 7.534510799, 7.541241633, 7.58833327, 7.675676842, 7.803070431, 7.970219531, 8.176737731, 8.422147605, 8.70588182, 9.027284445, 9.385612468, 9.780037514, 10.20964776, 10.67345004, 11.13492728, 11.17037214, 11.6992653, 12.25890683, 12.84800295, 13.46519181, 14.10904661, 14.77807889, 15.470742, 16.18543466, 16.92050464, 17.67425264, 18.44493615, 19.23077353, 20.02994808, 20.8406123, 21.66089211, 22.4888912, 23.32269542, 24.1603772, 25])
p_turbine = np.array([0, 0, 0.375538649, 0.635644519, 1.014237244, 1.473497284, 1.996315118, 2.571170922, 3.179901342, 3.803380108, 4.421435886, 5.013265683, 5.55992078, 6.043266596, 6.446694086, 6.756659289, 6.962663346, 7.057781739, 7.076914762, 7.211728426, 7.466186181, 7.847828144, 8.367545594, 9.039659135, 9.882192126, 10.91883042, 12.17633054, 13.68438108, 15.97152541, 18.12124994, 20.20010629, 22, 21.99998265, 21.99998772, 22.00000215, 22.0000002, 21.99990199, 22.00008648, 22.0000445, 22.0000247, 22.00001503, 22.00000988, 22.00000673, 22.00000464, 22.00000305, 22.00000173, 22.00000068, 22.000396, 21.99989367, 22.00000109, 22.00000282, 22.00000531])*1e3
ct_turbine = np.array([0, 0, 0.781330462, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.774831279, 0.751237387, 0.687365371, 0.63157364, 0.620285375, 0.500707415, 0.417418767, 0.352496626, 0.300020593, 0.256857131, 0.220980221, 0.190956234, 0.165702375, 0.144374588, 0.126299006, 0.110930462, 0.09782423, 0.086614854, 0.07700031, 0.068731176, 0.061595491, 0.055424038, 0.050069939, 0.045412126])
wind_turbine = WindTurbine(name='IEA22MW',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=PowerCtTabular(ws_turbine,p_turbine,'kW',ct_turbine))    
diameter = wind_turbine.diameter()
ws_rated = 11.

# extract HKN data
with open(r'input_folder/HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
    HKN_data = pickle.load(f)
hkn_site = HKN_data['hkn_site']
hkn_site_bathymetry_grid = HKN_data['hkn_site_bathymetry_grid']
hkn_site_x_grid = HKN_data['hkn_site_x_grid']
hkn_site_y_grid = HKN_data['hkn_site_y_grid']
hkn_wt_x = HKN_data['hkn_wt_x']
hkn_wt_y = HKN_data['hkn_wt_y']
coord_sub = utm.from_latlon(52.70,4.29)
x_sub = coord_sub[0]
y_sub = coord_sub[1]
diameter_hkn = 200.

# scale HKN data - boundaries
coord_sub = utm.from_latlon(52.70,4.29)
x_sub = coord_sub[0]
y_sub = coord_sub[1]
diameter_hkn = 200.
hkn_wt_x_scaled = x_sub + (hkn_wt_x-x_sub)*(diameter/diameter_hkn)
hkn_wt_y_scaled = y_sub + (hkn_wt_y-y_sub)*(diameter/diameter_hkn)

# scale HKN data - wind resource (create new pywake site object)
wd = np.arange(0,360,3)
ds_hkn_scaled = xr.Dataset(
    data_vars={
        'Sector_frequency':(['wd'],np.where(wd == 270, 1, 0)),
        'Weibull_A':(['wd'],[float(hkn_site.ds['Weibull_A'].mean(dim=["x", "y"]).values[0])]*len(wd)),
        'Weibull_k':(['wd'],[float(hkn_site.ds['Weibull_k'].mean(dim=["x", "y"]).values[0])]*len(wd)),
        'TI':0.04    
        },
    coords={
        'wd':wd
        }
    )
site = XRSite(ds_hkn_scaled)

# define wind farm model (TurboPark, as implemented in Nygaard, 2022)
wake_deficitModel = TurboGaussianDeficit(
    ct2a=ct2a_mom1d,
    groundModel=Mirror(superpositionModel=SquaredSum()),
    rotorAvgModel=GaussianOverlapAvgModel(),
    ctlim=0.96)
wake_deficitModel.WS_key = 'WS_jlk'     # Ørsted scales the deficit with respect to the ambient wind speed of the downstream turbine
wfm = PropagateDownwind(site, wind_turbine,
                        wake_deficitModel=wake_deficitModel,        # Nygaard 2022
                        superpositionModel=SquaredSum(),            # Nygaard 2022
                        deflectionModel=JimenezWakeDeflection(),    # added
                        rotorAvgModel=GaussianOverlapAvgModel())    # added, but probably not required  

# wind rose discretization and minimum spacing
wd_array = np.arange(0,360,3)
ws_array = np.arange(3,26)
min_d = 1.*diameter

# coordinates turbine 1 (fixed)
x_1 = (np.max(hkn_wt_x_scaled)+np.min(hkn_wt_x_scaled))/2 #hkn_wt_x_scaled[0]
y_1 = (np.max(hkn_wt_y_scaled)+np.min(hkn_wt_y_scaled))/2 #hkn_wt_y_scaled[0]

# define bathymetry
if hbd:
    # high bathymetry difference 10-40 m
    distance_max = 10. # [D]
    bathymetry_ub = -20.
    bathymetry_lb = -60.#-60.
    slope_bathymetry = (bathymetry_ub-bathymetry_lb)/distance_max
    site_x_dim = x_1+np.linspace(-distance_max,distance_max,21,endpoint=True)*diameter
    site_y_dim = y_1+np.linspace(-distance_max,distance_max,21,endpoint=True)*diameter
    site_x_grid,site_y_grid = np.meshgrid(site_x_dim,site_y_dim,indexing='ij')
    site_bathymetry_grid = bathymetry_ub-slope_bathymetry*np.sqrt(((site_x_grid-x_1)/diameter)**2+((site_y_grid-y_1)/diameter)**2)

## define bathymetry
else:
    # HKN bathymetry difference 19-26 m
    distance_max = 10. # [D]
    bathymetry_ub = -19.
    bathymetry_lb = -28.
    slope_bathymetry = (bathymetry_ub-bathymetry_lb)/distance_max
    site_x_dim = x_1+np.linspace(-distance_max,distance_max,21,endpoint=True)*diameter
    site_y_dim = y_1+np.linspace(-distance_max,distance_max,21,endpoint=True)*diameter
    site_x_grid,site_y_grid = np.meshgrid(site_x_dim,site_y_dim,indexing='ij')
    site_bathymetry_grid = bathymetry_ub-slope_bathymetry*np.sqrt(((site_x_grid-x_1)/diameter)**2+((site_y_grid-y_1)/diameter)**2)

#%% Calculate optimal yaw angles
if Run:
    from msr.msr import ObjFuncComponent,MSR_optimizer
    
    # function to calculate the optimal yaw angles for one wind direction
    def calc_yaw_opt_onewd(x,y,wd_yaw_opt,ws_array):
    
        # initialization
        yaw_opt = np.zeros((len(x),1,len(ws_array)))
    
        # define objective function
        def calculate_power_wake_steering(x,y,wfm,wd,ws,yaw):
            power = np.sum(wfm(x,y,wd=wd,ws=ws,yaw=yaw,tilt=0).Power.values)
            return power
    
        # iterate for each flow case
        for k in np.arange(len(ws_array)):
    
            # create objective function component
            f_obj = ObjFuncComponent(obj_func = calculate_power_wake_steering,
                                    input_keys = ['yaw'],
                                    x = x,
                                    y = y,
                                    wfm = wfm,
                                    wd = wd_yaw_opt,
                                    ws = ws_array[k])
    
            # create optimizer object
            optimizer_MSR = MSR_optimizer(x = x,
                                        y = y,
                                        wd = wd_yaw_opt,
                                        f_obj = f_obj,
                                        n_step = 3)
    
            # add strategy (Wake steering - Refine)
            optimizer_MSR.add_strategy(str_name = 'Wake steering',
                                        var_name = 'yaw',
                                        opt_method = 'Refine',
                                        n_values = 5,
                                        cmin = -30.,
                                        cmax = 30.)
    
            # run optimization
            optimizer_MSR.optimize()
            c_opt = optimizer_MSR.c_opt
            yaw_opt_temp = c_opt['yaw']
    
            # store output
            yaw_opt[:,0,k] = yaw_opt_temp
    
            #print(f'Optimization completed for the flow case: wd={wd_yaw_opt} - ws={ws_array[k]}')
    
        return yaw_opt
    
    
    # 2 turbines -----------------------------------------------------------
    
    # wind direction for which the yaw angles are optimized
    wd_yaw_opt = 270.
    ind_wd = np.where(wd_array==wd_yaw_opt)[0]
    
    # define turbine positions
    n_samples = 20
    distance_turbine_array = np.linspace(2,8,n_samples,endpoint=True,dtype=float)
    x_mat = np.zeros((n_samples,2),dtype=float)+x_1
    y_mat = np.zeros((n_samples,2),dtype=float)+y_1
    x_mat[:,1] = x_1+distance_turbine_array*diameter
    
    yaw_opt_2wt_list = [None]*n_samples
    for i in np.arange(n_samples):
        yaw_temp = calc_yaw_opt_onewd(x_mat[i,:],y_mat[i,:],wd_yaw_opt,ws_array)
        yaw_ilk = np.zeros((2,len(wd_array),len(ws_array)))
        yaw_ilk[:,ind_wd,:] = yaw_temp
        yaw_opt_2wt_list[i] = yaw_ilk
    
    # 3 turbines -----------------------------------------------------------
    
    # wind direction for which the yaw angles are optimized
    wd_yaw_opt = 270.
    
    # define turbine positions
    n_samples = 20
    distance_turbine_array = np.linspace(2,8,n_samples,endpoint=True,dtype=float)
    x_mat = np.zeros((n_samples,3),dtype=float)+x_1
    y_mat = np.zeros((n_samples,3),dtype=float)+y_1
    x_mat[:,1] = x_1+distance_turbine_array*diameter
    x_mat[:,2] = x_1+10.*diameter
    
    yaw_opt_3wt_list = [None]*n_samples
    for i in np.arange(n_samples):
        yaw_temp = calc_yaw_opt_onewd(x_mat[i,:],y_mat[i,:],wd_yaw_opt,ws_array)
        yaw_ilk = np.zeros((3,len(wd_array),len(ws_array)))
        yaw_ilk[:,ind_wd,:] = yaw_temp
        yaw_opt_3wt_list[i] = yaw_ilk

    #%% Setup DETECT deifntion and inputs
    matlab_eng = matlab.engine.start_matlab()
    matlab_eng.addpath(matlab_eng.genpath('input_folder/matlab_funcs'), nargout=0)
    matlab_eng.addpath(matlab_eng.genpath('../../../detect'), nargout=0)
    
    cable_specs = [
        {"diameter_mm2": 185, "capacity_NrT": 3, "cost_€_m": 368.9},
        {"diameter_mm2": 400, "capacity_NrT": 5, "cost_€_m": 428.9},
        {"diameter_mm2": 1000, "capacity_NrT": 7, "cost_€_m": 737.1}
    ]
    
    # initialization
    objfun_xy_wrapper_2wt_list = [None]*n_samples
    objfun_xy_wrapper_2wtYawOpt_list = [None]*n_samples
    objfun_xy_wrapper_3wt_list = [None]*n_samples
    objfun_xy_wrapper_3wtYawOpt_list = [None]*n_samples
    
    # define main blocks (iterate for each case) - 2wt
    for i in np.arange(n_samples):
    
        aep_calculator = AEPCalculator(wfm=wfm,
                                        wd_array=wd_array,
                                        ws_array=ws_array,
                                        ws_rated=ws_rated,
                                        wind_turbine=wind_turbine,
                                        use_geomYaw=False)
    
        aep_calculator_geomYaw = AEPCalculator(wfm=wfm,
                                            wd_array=wd_array,
                                            ws_array=ws_array,
                                            ws_rated=ws_rated,
                                            wind_turbine=wind_turbine,
                                            use_geomYaw=True)
    
        aep_calculator_optYaw = AEPCalculator(wfm=wfm,
                                            wd_array=wd_array,
                                            ws_array=ws_array,
                                            ws_rated=ws_rated,
                                            wind_turbine=wind_turbine,
                                            use_inputYaw=True,
                                            yaw_input=np.flip(yaw_opt_2wt_list[i],axis=0))
    
        cable_calculator = CableCalculator(x_sub=x_sub,
                                            y_sub=y_sub,
                                            cable_specs=cable_specs)
    
        water_depth_calculator = WaterDepthCalculator(site_x_grid=site_x_grid,
                                                    site_y_grid=site_y_grid,
                                                    site_bathymetry_grid=site_bathymetry_grid)
    
        # define wrappers
    
        python_xy_wrapper = PopEval_python_XYwrapper(aep_calculator,
                                                    cable_calculator,
                                                    water_depth_calculator,
                                                    min_d,
                                                    parallel_execution=False,
                                                    n_cpu=1)
    
        python_xy_wrapper_geomYaw = PopEval_python_XYwrapper(aep_calculator_geomYaw,
                                                            cable_calculator,
                                                            water_depth_calculator,
                                                            min_d,
                                                            parallel_execution=False,
                                                            n_cpu=1)
        python_xy_wrapper_optYaw = PopEval_python_XYwrapper(aep_calculator_optYaw,
                                                            cable_calculator,
                                                            water_depth_calculator,
                                                            min_d,
                                                            parallel_execution=False,
                                                            n_cpu=1)
        matlab_xy_wrapper = PopEval_matlab_XYwrapper(matlab_eng,
                                                    parallel_execution=False,
                                                    n_cpu=1)
        objfun_xy_wrapper = ObjFunction_XYwrapper(python_xy_wrapper,
                                                matlab_xy_wrapper,
                                                maximize=[True]*12,
                                                value_per_turbine=False,
                                                output_keys=['AEP','LCOE','IOE','Cable_emissions','Monopile_emissions','Tower_emissions','Total_lifecycle_emissions','Cable_costs','Monopile_costs','Tower_costs','Total_lifecycle_costs','AEPnet'])
    
        objfun_xy_wrapper_geomYaw = ObjFunction_XYwrapper(python_xy_wrapper_geomYaw,
                                                        matlab_xy_wrapper,
                                                        maximize=[True]*12,
                                                        value_per_turbine=False,
                                                        output_keys=['AEP','LCOE','IOE','Cable_emissions','Monopile_emissions','Tower_emissions','Total_lifecycle_emissions','Cable_costs','Monopile_costs','Tower_costs','Total_lifecycle_costs','AEPnet'])
        objfun_xy_wrapper_optYaw = ObjFunction_XYwrapper(python_xy_wrapper_optYaw,
                                                        matlab_xy_wrapper,
                                                        maximize=[True]*12,
                                                        value_per_turbine=False,
                                                        output_keys=['AEP','LCOE','IOE','Cable_emissions','Monopile_emissions','Tower_emissions','Total_lifecycle_emissions','Cable_costs','Monopile_costs','Tower_costs','Total_lifecycle_costs','AEPnet'])
    
        objfun_xy_wrapper_2wt_list[i] = objfun_xy_wrapper
        objfun_xy_wrapper_2wtYawOpt_list[i] = objfun_xy_wrapper_optYaw
    
    
    # define main blocks (iterate for each case) - 3wt
    for i in np.arange(n_samples):
    
        aep_calculator = AEPCalculator(wfm=wfm,
                                        wd_array=wd_array,
                                        ws_array=ws_array,
                                        ws_rated=ws_rated,
                                        wind_turbine=wind_turbine,
                                        use_geomYaw=False)
    
        aep_calculator_geomYaw = AEPCalculator(wfm=wfm,
                                            wd_array=wd_array,
                                            ws_array=ws_array,
                                            ws_rated=ws_rated,
                                            wind_turbine=wind_turbine,
                                            use_geomYaw=True)
    
        aep_calculator_optYaw = AEPCalculator(wfm=wfm,
                                            wd_array=wd_array,
                                            ws_array=ws_array,
                                            ws_rated=ws_rated,
                                            wind_turbine=wind_turbine,
                                            use_inputYaw=True,
                                            yaw_input=yaw_opt_3wt_list[i])
    
        cable_calculator = CableCalculator(x_sub=x_sub,
                                            y_sub=y_sub,
                                            cable_specs=cable_specs)
    
        water_depth_calculator = WaterDepthCalculator(site_x_grid=site_x_grid,
                                                    site_y_grid=site_y_grid,
                                                    site_bathymetry_grid=site_bathymetry_grid)
    
        # define wrappers
    
        python_xy_wrapper = PopEval_python_XYwrapper(aep_calculator,
                                                    cable_calculator,
                                                    water_depth_calculator,
                                                    min_d,
                                                    parallel_execution=False,
                                                    n_cpu=1)
    
        python_xy_wrapper_geomYaw = PopEval_python_XYwrapper(aep_calculator_geomYaw,
                                                            cable_calculator,
                                                            water_depth_calculator,
                                                            min_d,
                                                            parallel_execution=False,
                                                            n_cpu=1)
        python_xy_wrapper_optYaw = PopEval_python_XYwrapper(aep_calculator_optYaw,
                                                            cable_calculator,
                                                            water_depth_calculator,
                                                            min_d,
                                                            parallel_execution=False,
                                                            n_cpu=1)
        matlab_xy_wrapper = PopEval_matlab_XYwrapper(matlab_eng,
                                                    parallel_execution=False,
                                                    n_cpu=1)
        objfun_xy_wrapper = ObjFunction_XYwrapper(python_xy_wrapper,
                                                matlab_xy_wrapper,
                                                maximize=[True]*12,
                                                value_per_turbine=False,
                                                output_keys=['AEP','LCOE','IOE','Cable_emissions','Monopile_emissions','Tower_emissions','Total_lifecycle_emissions','Cable_costs','Monopile_costs','Tower_costs','Total_lifecycle_costs','AEPnet'])
    
        objfun_xy_wrapper_geomYaw = ObjFunction_XYwrapper(python_xy_wrapper_geomYaw,
                                                        matlab_xy_wrapper,
                                                        maximize=[True]*12,
                                                        value_per_turbine=False,
                                                        output_keys=['AEP','LCOE','IOE','Cable_emissions','Monopile_emissions','Tower_emissions','Total_lifecycle_emissions','Cable_costs','Monopile_costs','Tower_costs','Total_lifecycle_costs','AEPnet'])
        objfun_xy_wrapper_optYaw = ObjFunction_XYwrapper(python_xy_wrapper_optYaw,
                                                        matlab_xy_wrapper,
                                                        maximize=[True]*12,
                                                        value_per_turbine=False,
                                                        output_keys=['AEP','LCOE','IOE','Cable_emissions','Monopile_emissions','Tower_emissions','Total_lifecycle_emissions','Cable_costs','Monopile_costs','Tower_costs','Total_lifecycle_costs','AEPnet'])
    
        objfun_xy_wrapper_3wt_list[i] = objfun_xy_wrapper
        objfun_xy_wrapper_3wtYawOpt_list[i] = objfun_xy_wrapper_optYaw

    #%%
    # debug
    
    n_samples = 20
    distance_turbine_array = np.linspace(2,8,n_samples,endpoint=True,dtype=float)
    x_mat = np.zeros((n_samples,3),dtype=float)+x_1
    y_mat = np.zeros((n_samples,3),dtype=float)+y_1
    x_mat[:,1] = x_1+distance_turbine_array*diameter
    x_mat[:,2] = x_1+10.*diameter
    
    for i in np.arange(n_samples):
    
        aep_calculator_optYaw = AEPCalculator(wfm=wfm,
                                        wd_array=wd_array,
                                        ws_array=ws_array,
                                        ws_rated=ws_rated,
                                        wind_turbine=wind_turbine,
                                        use_inputYaw=True,
                                        yaw_input=yaw_opt_3wt_list[i])
    
        # calculate evaluation function
        f_mat_BL = aep_calculator(x_mat[i,:],y_mat[i,:])
        f_mat_OY = aep_calculator_optYaw(x_mat[i,:],y_mat[i,:])
    
        aep_BL = f_mat_BL['AEP'].values
        aep_OY = f_mat_OY['AEP'].values
    
        #print(f'BL: {aep_BL}')
        #print(f'OY: {aep_OY}')
        print(f'gain: {100*(aep_OY-aep_BL)/aep_BL}')

#%% Plot function
normalize = True
wrt_respective_baseline = False
def plot_results(distance_turbine_array,res_yaw,res_BL,water_depth_array,threetur=False):
    res_BL_org = res_BL.copy()
    # Normalize
    if normalize:
        if wrt_respective_baseline:
            # compared to resepective baseline
            res_yaw = res_yaw / res_BL[0] * 100 - 100
            res_BL = res_BL / res_BL[0] * 100 - 100
        else:
            # KPIs
            res_yaw[:,0:3] = res_yaw[:,0:3] / res_BL[0,0:3] * 100 - 100
            res_BL[:,0:3] = res_BL[:,0:3] / res_BL[0,0:3] * 100 - 100
            # AEPnet
            res_yaw[:,-1] = res_yaw[:,-1] / res_BL[0,-1] * 100 - 100
            res_BL[:,-1] = res_BL[:,-1] / res_BL[0,-1] * 100 - 100
            # cable + monopile changes
            # (a) wrt overall emission changes
            res_BL[:,3:6] = res_BL[:,3:6] - res_BL[0,3:6]            # emission change in cables + monopiles
            res_BL[:,3:7] = res_BL[:,3:7] / res_BL[0,6] * 100        # relative change w.r.t. overall emissions
            res_BL[:,6]   = res_BL[:,6] - 100
            # (b) wrt overall cost changes
            res_BL[:,7:10] = res_BL[:,7:10] - res_BL[0,7:10]            # emission change in cables + monopiles
            res_BL[:,7:11] = res_BL[:,7:11] / res_BL[0,10] * 100        # relative change w.r.t. overall emissions
            res_BL[:,10]   = res_BL[:,10] - 100
    else:
        if wrt_respective_baseline:
            res_BL[:,3:-1] = res_BL[:,3:] - res_BL[0,3:-1]

    # extract values
    # KPIs
    aep_BL_array        = res_BL[:,0]
    aepnet_BL_array     = res_BL[:,-1]
    lcoe_BL_array       = res_BL[:,1]
    ioe_BL_array        = res_BL[:,2]
    
    # emissions
    em_cab_BL_array     = res_BL[:,3]
    em_mp_BL_array      = res_BL[:,4]
    em_tow_BL_array     = res_BL[:,5]
    em_tot_BL_array     = res_BL[:,6]
    if normalize and wrt_respective_baseline:
        em_others_BL_array = res_BL_org[:,6] - (res_BL_org[:,5] + res_BL_org[:,4] + res_BL_org[:,3])
        em_others_BL_array = em_others_BL_array / em_others_BL_array[0] * 100 - 100
    else:
        em_others_BL_array  = em_tot_BL_array - (em_mp_BL_array + em_cab_BL_array + em_tow_BL_array)
    # costs
    cost_cab_BL_array   = res_BL[:,7]
    cost_mp_BL_array    = res_BL[:,8]
    cost_tow_BL_array    = res_BL[:,9]
    cost_tot_BL_array   = res_BL[:,10]
    if normalize and wrt_respective_baseline:
        cost_others_BL_array = res_BL_org[:,10] - (res_BL_org[:,9] + res_BL_org[:,8] + res_BL_org[:,7])
        cost_others_BL_array = cost_others_BL_array / cost_others_BL_array[0] * 100 - 100
    else:
        cost_others_BL_array = cost_tot_BL_array - (cost_mp_BL_array + cost_cab_BL_array + cost_tow_BL_array)
    #
    aep_GY_array    = res_yaw[:,0]
    aepnet_GY_array    = res_yaw[:,-1]
    lcoe_GY_array   = res_yaw[:,1]
    ioe_GY_array    = res_yaw[:,2]
        
    # Prepare Plot
    s1 = 30
    s2 = 40
    savefig = False
    name_path = 'figures/initial_examples/'
    name_fig = ('3wt_' if threetur else '2wt_') + ('highB' if hbd else 'lowB')
    name_format = 'svg'
    
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(3.12,4.44*6/5),sharex=True)
    
    # ---------------------------------------------------------------------------------
    # Plot
    #
    # AEP
    if hbd:
        axes[0].set_title('High bathymetry difference',fontsize=FS1)
    else:
        axes[0].set_title('Low bathymetry difference',fontsize=FS1)
    axes[0].plot(distance_turbine_array,aep_BL_array,c=color_1_shades[1],label='WFLO')
    axes[0].scatter(distance_turbine_array[np.argmax(aep_BL_array)],aep_BL_array[np.argmax(aep_BL_array)],marker='o',s=s1,c=color_1_shades[1])#,label='WFLO optimum' if hbd else None)
    axes[0].plot(distance_turbine_array,aep_GY_array,c=color_1_shades[1],linestyle='--',label='WFCO')
    axes[0].scatter(distance_turbine_array[np.argmax(aep_GY_array)],aep_GY_array[np.argmax(aep_GY_array)],marker='*',s=s2,c=color_1_shades[1])#,label='WFLO optimum' if hbd else None)
    if normalize:
        axes[0].set_ylabel(r'$\Delta$AEP [%]')
    else:
        axes[0].set_ylabel(r'AEP'+'\n'+r'[GWh]')
    if not hbd and not threetur:
            axes[0].legend(loc='upper left',fontsize=FS2, ncol=1)
    if hbd and threetur:
            axes[0].legend(loc='lower center',fontsize=FS2, ncol=2)
            
    # AEPnet
    axes[1].plot(distance_turbine_array,aepnet_BL_array,c=color_0_shades[1])#,label='Greedy')
    axes[1].scatter(distance_turbine_array[np.argmax(aepnet_BL_array)],aepnet_BL_array[np.argmax(aepnet_BL_array)],marker='o',s=s1,c=color_0_shades[1],label='WFLO optimum')
    axes[1].plot(distance_turbine_array,aepnet_GY_array,c=color_0_shades[1],linestyle='--')#,label='Wake steering')
    axes[1].scatter(distance_turbine_array[np.argmax(aepnet_GY_array)],aepnet_GY_array[np.argmax(aepnet_GY_array)],marker='*',s=s2,c=color_0_shades[1],label='WFCO optimum')
    if normalize:
        axes[1].set_ylabel(r'$\mathrm{\Delta AEP_{net}}$ [%]')
    else:
        axes[1].set_ylabel(r'$\mathrm{AEP_{net}}$'+'\n'+r'[GWh]')  
    if not hbd and not threetur:
        axes[1].legend(loc='upper left',fontsize=FS2, ncol=1)
    if hbd and threetur:
        axes[1].legend(loc='lower center',fontsize=FS2, ncol=2)
     
    # COE_EUR
    axes[2].plot(distance_turbine_array,lcoe_BL_array,c=color_2_shades[1])#,label='Greedy')
    axes[2].scatter(distance_turbine_array[np.argmin(lcoe_BL_array)],lcoe_BL_array[np.argmin(lcoe_BL_array)],marker='o',s=s1,c=color_2_shades[1],label='WFLO optimum')
    axes[2].plot(distance_turbine_array,lcoe_GY_array,c=color_2_shades[1],linestyle='--')#,label='Wake steering')
    axes[2].scatter(distance_turbine_array[np.argmin(lcoe_GY_array)],lcoe_GY_array[np.argmin(lcoe_GY_array)],marker='*',s=s2,c=color_2_shades[1],label='WFCO optimum')
    if normalize:
        axes[2].set_ylabel(r'$\mathrm{\Delta COE_{€}}$ [%]')
    else:
        axes[2].set_ylabel(r'$\mathrm{COE_{€}}$'+'\n'+r'[€/MWh]')
    
    # COE_CO2
    axes[3].plot(distance_turbine_array,ioe_BL_array,c=color_3_shades[1])#,label='Greedy')
    axes[3].scatter(distance_turbine_array[np.argmin(ioe_BL_array)],ioe_BL_array[np.argmin(ioe_BL_array)],marker='o',s=s1,c=color_3_shades[1],label='WFLO optimum')
    axes[3].plot(distance_turbine_array,ioe_GY_array,c=color_3_shades[1],linestyle='--')#,label='Wake steering')
    axes[3].scatter(distance_turbine_array[np.argmin(ioe_GY_array)],ioe_GY_array[np.argmin(ioe_GY_array)],marker='*',s=s2,c=color_3_shades[1],label='WFCO optimum')
    if normalize:
        axes[3].set_ylabel(r'$\mathrm{\Delta COE_{CO2}}$ [%]')
    else:
        axes[3].set_ylabel(r'$\mathrm{COE_{CO2}}$'+'\n'+r'$[\mathrm{kgCO_2eq/MWh}]$')
    
    # Total costs
    if wrt_respective_baseline:
        axes[4].plot(distance_turbine_array,cost_cab_BL_array,c=color_4_shades[1],label='Cable')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_cab_BL_array)],cost_cab_BL_array[np.argmin(cost_cab_BL_array)],marker='o',s=s1,c=color_4_shades[1])
        axes[4].plot(distance_turbine_array,cost_mp_BL_array,c=color_5_shades[1],linestyle='--',label='Monopile')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_mp_BL_array)],cost_mp_BL_array[np.argmin(cost_mp_BL_array)],marker='*',s=s2,c=color_5_shades[1])
        axes[4].plot(distance_turbine_array,cost_tow_BL_array,c=color_6_shades[1],linestyle='--',label='Tower')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_tow_BL_array)],cost_tow_BL_array[np.argmin(cost_tow_BL_array)],marker='*',s=s2,c=color_6_shades[1])
        axes[4].plot(distance_turbine_array,cost_others_BL_array,c=color_7_shades[1],linestyle='--',label='Others')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_others_BL_array)],cost_others_BL_array[np.argmin(cost_others_BL_array)],marker='*',s=s2,c=color_7_shades[1])
    elif normalize:
        axes[4].stackplot(
            distance_turbine_array,
            cost_cab_BL_array,
            cost_mp_BL_array,
            cost_tow_BL_array,
            cost_others_BL_array,
            labels=['Cable', 'Monopile', 'Tower', 'Others']
        )
    else:
        axes[4].plot(distance_turbine_array,cost_cab_BL_array,c=color_4_shades[1],label='Cable')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_cab_BL_array)],cost_cab_BL_array[np.argmin(cost_cab_BL_array)],marker='o',s=s1,c=color_4_shades[1])
        axes[4].plot(distance_turbine_array,cost_mp_BL_array,c=color_5_shades[1],linestyle='--',label='Monopile')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_mp_BL_array)],cost_mp_BL_array[np.argmin(cost_mp_BL_array)],marker='*',s=s2,c=color_5_shades[1])
        axes[4].plot(distance_turbine_array,cost_tow_BL_array,c=color_6_shades[1],linestyle='--',label='Towers')
        axes[4].scatter(distance_turbine_array[np.argmin(cost_tow_BL_array)],cost_tow_BL_array[np.argmin(cost_tow_BL_array)],marker='*',s=s2,c=color_6_shades[1])
        
    if normalize:
        axes[4].set_ylabel(r'$\mathrm{\Delta TLCC}$ [%]')
    else:
        if wrt_respective_baseline:
            axes[4].set_ylabel(r'$\mathrm{\Delta Costs}$'+'\n'+r'[€]')
        else:
            axes[4].set_ylabel(r'$\mathrm{Costs}$'+'\n'+r'[€]')
    if not hbd:
        axes[4].legend(loc='upper left',fontsize=FS2, ncol=2)

    # Total emissions
    if wrt_respective_baseline:
        axes[5].plot(distance_turbine_array,em_cab_BL_array,c=color_4_shades[1],label='Cable')
        axes[5].scatter(distance_turbine_array[np.argmin(em_cab_BL_array)],em_cab_BL_array[np.argmin(em_cab_BL_array)],marker='o',s=s1,c=color_4_shades[1])
        axes[5].plot(distance_turbine_array,em_mp_BL_array,c=color_5_shades[1],linestyle='--',label='Monopile')
        axes[5].scatter(distance_turbine_array[np.argmin(em_mp_BL_array)],em_mp_BL_array[np.argmin(em_mp_BL_array)],marker='*',s=s2,c=color_5_shades[1])
        axes[5].plot(distance_turbine_array,em_tow_BL_array,c=color_6_shades[1],linestyle='--',label='Tower')
        axes[5].scatter(distance_turbine_array[np.argmin(em_tow_BL_array)],em_tow_BL_array[np.argmin(em_tow_BL_array)],marker='*',s=s2,c=color_6_shades[1])
        axes[5].plot(distance_turbine_array,em_others_BL_array,c=color_7_shades[1],linestyle='--',label='Others')
        axes[5].scatter(distance_turbine_array[np.argmin(em_others_BL_array)],em_others_BL_array[np.argmin(em_others_BL_array)],marker='*',s=s2,c=color_7_shades[1])
    elif normalize:
        axes[5].stackplot(
            distance_turbine_array,
            em_cab_BL_array,
            em_mp_BL_array,
            em_tow_BL_array,
            em_others_BL_array,
            labels=['Cable', 'Monopile', 'Tower', 'Others']
        )
    else:
        axes[5].plot(distance_turbine_array,em_cab_BL_array,c=color_4_shades[1],label='Cable')
        axes[5].scatter(distance_turbine_array[np.argmin(em_cab_BL_array)],em_cab_BL_array[np.argmin(em_cab_BL_array)],marker='o',s=s1,c=color_4_shades[1])
        axes[5].plot(distance_turbine_array,em_mp_BL_array,c=color_5_shades[1],linestyle='--',label='Monopile')
        axes[5].scatter(distance_turbine_array[np.argmin(em_mp_BL_array)],em_mp_BL_array[np.argmin(em_mp_BL_array)],marker='*',s=s2,c=color_5_shades[1])
        axes[5].plot(distance_turbine_array,em_tow_BL_array,c=color_6_shades[1],linestyle='--',label='Tower')
        axes[5].scatter(distance_turbine_array[np.argmin(em_tow_BL_array)],em_tow_BL_array[np.argmin(em_tow_BL_array)],marker='*',s=s2,c=color_6_shades[1])
        
    if normalize:
        axes[5].set_ylabel(r'$\mathrm{\Delta TLCE}$ [%]')
    else:
        if wrt_respective_baseline:
            axes[5].set_ylabel(r'$\mathrm{\Delta Emissions}$'+'\n'+r'[kgCO2e]')
        else:
            axes[5].set_ylabel(r'$\mathrm{Emissions}$'+'\n'+r'[kgCO2e]')
    
    # Decorate
    axes[5].set_xlabel(r'Distance [D]')
    puffer = 0.2
    axes[5].set_xlim([min(distance_turbine_array)-puffer,max(distance_turbine_array)+puffer])
    # align axes ranges
    if normalize:
        for ax in axes:
            # if threetur and not hbd:
            #     ax.yaxis.set_major_locator(MultipleLocator(1))
            # else:
            ax.yaxis.set_major_locator(MultipleLocator(2))
    if normalize:
        data = [aep_BL_array,aep_GY_array,lcoe_BL_array,lcoe_GY_array,ioe_BL_array,ioe_GY_array,cost_tot_BL_array,em_tot_BL_array]
        min_lim = 0
        max_lim = 0
        for dat in data:
            if abs(min(dat)) < abs(max(dat)):
                # positive case
                if max(dat) > max_lim:
                    max_lim = max(dat)
                elif -(min(dat)) > min_lim:
                    min_lim = -(min(dat))
            else:
                # negative case
                if max(dat) < min_lim:
                    min_lim = max(dat)
                elif abs(min(dat)) > max_lim:
                    max_lim = abs(min(dat))
                elif abs(min(dat)) > min_lim:
                    min_lim = abs(min(dat))
                    
        # set ylim to have some puffer
        for ax in axes:
            puffer = 0.15 * (max_lim+min_lim)
            if abs(min(ax.get_ylim())) < abs(max(ax.get_ylim())):
                # positive case
                ax.set_ylim(-min_lim-puffer, max_lim+puffer)
                if AlignYaxis2turbine and not threetur:
                    ax.set_ylim(y_lim_2t[0], y_lim_2t[1])
                elif AlignYaxis3turbine and threetur:
                    ax.set_ylim(y_lim_3t[0], y_lim_3t[1])
            else:
                ax.set_ylim(-max_lim-puffer, min_lim+puffer)
                if AlignYaxis2turbine and not threetur:
                    ax.set_ylim(-y_lim_2t[1], -y_lim_2t[0])
                elif AlignYaxis3turbine and threetur:
                    ax.set_ylim(y_lim_3t[0], y_lim_3t[1])
                
    # constant lines to support visuals
    for ax in axes:
        ax.axhline(y=0,color='black',linewidth=0.8,linestyle=':',zorder=0)
        ax.axvline(min(distance_turbine_array),color='black',linewidth=0.5,zorder=0)
        ax.axvline(max(distance_turbine_array),color='black',linewidth=0.5,zorder=0)
        # ax.grid('true')
     
    # Apply font sizes consistently
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(alpha=0.6,linewidth=0.5)
        # Axis labels
        ax.xaxis.label.set_size(FS1)
        ax.yaxis.label.set_size(FS1)
    
        # Title (if any)
        ax.title.set_size(FS1)
    
        # Tick labels
        ax.tick_params(axis='both', labelsize=FS2)
    
        # Legend
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(leg.get_title().get_text(), prop={'size': FS2})
            for text in leg.get_texts():
                text.set_fontsize(FS2)
        
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.96,
        bottom=0.082,
        left=0.117,
        right=0.983,
        hspace=0.345,
        wspace=0.2,
    )
    
    if savefig: plt.savefig(name_path+name_fig+name_format,format=name_format)
    if WritePkl:
        with open(name_path+name_fig+".pkl", "wb") as f:
            pickle.dump(fig, f)
    plt.show()

    # print size pareto front
    ind_best_lcoe_BL = np.argmin(lcoe_BL_array)
    ind_best_lcoe_GY = np.argmin(lcoe_GY_array)
    ind_best_ioe_BL = np.argmin(ioe_BL_array)
    ind_best_ioe_GY = np.argmin(ioe_GY_array)
    delta_lcoe_BL = lcoe_BL_array[ind_best_ioe_BL]-lcoe_BL_array[ind_best_lcoe_BL]
    delta_ioe_BL = ioe_BL_array[ind_best_lcoe_BL]-ioe_BL_array[ind_best_ioe_BL]
    delta_lcoe_GY = lcoe_GY_array[ind_best_ioe_GY]-lcoe_GY_array[ind_best_lcoe_GY]
    delta_ioe_GY = ioe_GY_array[ind_best_lcoe_GY]-ioe_GY_array[ind_best_ioe_GY]
    
    if normalize:
        print(' ')
        print ('########################')
        print('Three turbines case' if threetur else 'Two turbines case')
        print('with High bathymetry difference' if hbd else 'with Low bathymetry difference')
        print('Size Pareto front ----------------')
        print('Baseline')
        print(f'Delta LCOE: {delta_lcoe_BL} %')
        print(f'Delta IOE: {delta_ioe_BL} %')
        print('Yaw')
        print(f'Delta LCOE: {delta_lcoe_GY} %')
        print(f'Delta IOE: {delta_ioe_GY} %')
        print('Y-axis limits')
        y_min, y_max = axes[0].get_ylim()
        print(f"Y-axis limits: {y_min:.2f} to {y_max:.2f}")
    else:
        print('Size Pareto front ----------------')
        print('Baseline')
        print(f'Delta LCOE: {delta_lcoe_BL} \t -> {100*delta_lcoe_BL/lcoe_BL_array[ind_best_lcoe_BL]} %')
        print(f'Delta IOE: {delta_ioe_BL} \t -> {100*delta_ioe_BL/ioe_BL_array[ind_best_ioe_BL]} %')
        print('Geom yaw')
        print(f'Delta LCOE: {delta_lcoe_GY} \t -> {100*delta_lcoe_GY/lcoe_GY_array[ind_best_lcoe_GY]} %')
        print(f'Delta IOE: {delta_ioe_GY} \t -> {100*delta_ioe_GY/ioe_GY_array[ind_best_ioe_GY]} %')

# %% 2wt scenario
# create data
if Run:
    # define turbine positions
    n_samples = 20
    distance_turbine_array = np.linspace(2,8,n_samples,endpoint=True,dtype=float)
    x_mat = np.zeros((n_samples,2),dtype=float)+x_1
    y_mat = np.zeros((n_samples,2),dtype=float)+y_1
    x_mat[:,1] = x_1+distance_turbine_array*diameter
    
    # calculate evaluation function
    # flip so detect works without substation!
    f_mat_BL = objfun_xy_wrapper(x_mat[:, ::-1],y_mat)
    f_mat_geomYaw = objfun_xy_wrapper_geomYaw(x_mat[:, ::-1],y_mat)
    
    f_mat_optYaw = np.zeros_like(f_mat_BL)
    for i in np.arange(n_samples):
        x_temp = x_mat[i, ::-1]
        y_temp = y_mat[i,:]
        f_mat_optYaw[i,:] = objfun_xy_wrapper_2wtYawOpt_list[i](x_temp[na,:],y_temp[na,:])
    
    # extract bathymetry
    water_depth_array = np.zeros(n_samples)
    for i in np.arange(len(distance_turbine_array)):
        x = x_mat[i,:]
        y = y_mat[i,:]
        water_depth_array[i] = water_depth_calculator(x,y)[1]
        
    with open("2wt_hbd_results.pkl" if hbd else "2wt_lbd_results.pkl", "wb") as f:
            pickle.dump((distance_turbine_array, f_mat_BL, f_mat_geomYaw, f_mat_optYaw, water_depth_array), f)
else: 
    with open("2wt_hbd_results.pkl" if hbd else "2wt_lbd_results.pkl", "rb") as f:
        distance_turbine_array, f_mat_BL, f_mat_geomYaw, f_mat_optYaw, water_depth_array = pickle.load(f)

# plot
if True:
    plot_results(distance_turbine_array,f_mat_optYaw,f_mat_BL,water_depth_array)

#%% 3wt scenario
# create data
if Run:
    # define turbine positions
    n_samples = 20
    distance_turbine_array = np.linspace(2,8,n_samples,endpoint=True,dtype=float)
    x_mat = np.zeros((n_samples,3),dtype=float)+x_1
    y_mat = np.zeros((n_samples,3),dtype=float)+y_1
    x_mat[:,1] = x_1+distance_turbine_array*diameter
    x_mat[:,2] = x_1+10.*diameter
    
    # calculate evaluation function
    f_mat_BL = objfun_xy_wrapper(x_mat,y_mat)
    f_mat_geomYaw = objfun_xy_wrapper_geomYaw(x_mat,y_mat)

    f_mat_optYaw = np.zeros_like(f_mat_BL)
    for i in np.arange(n_samples):
        x_temp = x_mat[i,:]
        y_temp = y_mat[i,:]
        f_mat_optYaw[i,:] = objfun_xy_wrapper_3wtYawOpt_list[i](x_temp[na,:],y_temp[na,:])
    
    # extract bathymetry
    water_depth_array = np.zeros(n_samples)
    for i in np.arange(len(distance_turbine_array)):
        x = x_mat[i,:]
        y = y_mat[i,:]
        water_depth_array[i] = water_depth_calculator(x,y)[1]

    with open("3wt_hbd_results.pkl" if hbd else "3wt_lbd_results.pkl", "wb") as f:
            pickle.dump((distance_turbine_array, f_mat_BL, f_mat_geomYaw, f_mat_optYaw, water_depth_array), f)
else: 
    with open("3wt_hbd_results.pkl" if hbd else "3wt_lbd_results.pkl", "rb") as f:
        distance_turbine_array, f_mat_BL, f_mat_geomYaw, f_mat_optYaw, water_depth_array = pickle.load(f)
            
# plot
if True:
    plot_results(distance_turbine_array,f_mat_optYaw,f_mat_BL,water_depth_array,threetur=True)