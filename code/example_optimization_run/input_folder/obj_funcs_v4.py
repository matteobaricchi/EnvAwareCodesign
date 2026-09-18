#%%
# import main packages
import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
from numpy import newaxis as na
import pickle
import utm

# import tools and data
from input_folder.utils.wflop_utils import filter_turbines

# import geometric yaw
from input_folder.geometric_yaw.geometric_yaw import calculate_geomYaw_ExpCorr


# import py_wake packages
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_farm_models import PropagateDownwind,All2AllIterative
from py_wake.deficit_models import BastankhahGaussianDeficit,ZongGaussianDeficit,NOJDeficit
from py_wake.superposition_models import SquaredSum,LinearSum
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.rotor_avg_models import GaussianOverlapAvgModel,CGIRotorAvg,RotorCenter
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.turbulence_models import STF2017TurbulenceModel
from py_wake.site import XRSite,UniformWeibullSite
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.ground_models.ground_models import Mirror
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.deficit_models import Rathmann

# import DETECT wrappers
from input_folder.obj_functions_DETECT import AEPCalculator_DETECT
from input_folder.obj_functions_DETECT import AEPCalculator_DETECT_Weibull
from input_folder.obj_functions_DETECT import CableCalculator_DETECT
from input_folder.obj_functions_DETECT import CableCalculator_inputCab
from input_folder.obj_functions_DETECT import WaterDepthCalculator_DETECT
from input_folder.obj_functions_DETECT import PopEval_python_XYwrapper
from input_folder.obj_functions_DETECT import PopEval_matlab_XYwrapper
from input_folder.obj_functions_DETECT import ObjFunction_XYwrapper
import matlab.engine



#%%


def generate_HKNscaled_site(f_scaling_density=1):

    # extract HKN data
    with open(f'input_folder/HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_site = HKN_data['hkn_site']
    diameter = 283.2
    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]
    diameter_hkn = 200.

    # scale HKN data - wind resource (create new pywake site object)
    ds_hkn_scaled = xr.Dataset(
        data_vars={
            'Sector_frequency':(['x','y','wd'],hkn_site.ds['Sector_frequency'].values),
            'Weibull_A':(['x','y','wd'],hkn_site.ds['Weibull_A'].values*((170./115.)**0.1)),
            'Weibull_k':(['x','y','wd'],hkn_site.ds['Weibull_k'].values),
            'TI':0.04    
            },
        coords={
            'x':x_sub + (hkn_site.ds['x'].values-x_sub)*(diameter/diameter_hkn)*f_scaling_density,
            'y':y_sub + (hkn_site.ds['y'].values-y_sub)*(diameter/diameter_hkn)*f_scaling_density,
            'wd':hkn_site.ds['wd'].values
            }
        )
    hkn_site_scaled = XRSite(ds_hkn_scaled)

    return hkn_site_scaled




def generate_HKNscaled_site_aWR():

    # extract HKN data
    with open(f'input_folder/HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_site = HKN_data['hkn_site']

    # average site
    wd_site = np.linspace(0,360,16,endpoint=False)
    p_wd_site = np.mean(hkn_site.ds.Sector_frequency.values,axis=(0,1))/np.sum(np.mean(hkn_site.ds.Sector_frequency.values,axis=(0,1)))
    a_site = np.ones_like(wd_site)*np.mean(hkn_site.ds.Weibull_A.values*((170./115.)**0.1),axis=(0,1))
    k_site = np.ones_like(wd_site)*np.mean(hkn_site.ds.Weibull_k.values,axis=(0,1))
    hkn_site_average = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.1)

    return hkn_site_average


def generate_HKNscaled_bathymetry(f_scaling_density=1,bathymetry_range=None,use_sloped_profile=False):

    # extract HKN data
    with open(f'input_folder/HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_site = HKN_data['hkn_site']

    # extract HKN bathymetry data (high resolution from EMODnet)
    with open(r'input_folder/HKN_data_and_tools/HKN_bathymetry_highres.pkl', 'rb') as f:
        HKN_bathymetry_data = pickle.load(f)
    hkn_site_bathymetry_grid = HKN_bathymetry_data['hkn_site_bathymetry_grid']
    hkn_site_x_grid = HKN_bathymetry_data['hkn_site_x_grid']
    hkn_site_y_grid = HKN_bathymetry_data['hkn_site_y_grid']

    # scale HKN data - bathymetry
    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]
    diameter_hkn = 200.
    diameter = 283.2
    hkn_site_x_grid_scaled = x_sub + (hkn_site_x_grid-x_sub)*(diameter/diameter_hkn)*f_scaling_density
    hkn_site_y_grid_scaled = y_sub + (hkn_site_y_grid-y_sub)*(diameter/diameter_hkn)*f_scaling_density

    if use_sloped_profile:

        # max and min values within the entire grid (not limited within the farm boundaries)
        bathymetry_ub = bathymetry_range[0]
        bathymetry_lb = bathymetry_range[1]

        # domaninat wind direction
        wd_slope = hkn_site.ds['wd'].values[np.argmax(np.mean(hkn_site.ds['Sector_frequency'].values,axis=(0,1)))]

        # scale bathymetry grid
        x_grid_rot = hkn_site_x_grid_scaled*np.cos(np.deg2rad(wd_slope))+hkn_site_y_grid_scaled*np.sin(np.deg2rad(wd_slope))
        y_grid_rot = -hkn_site_x_grid_scaled*np.sin(np.deg2rad(wd_slope))+hkn_site_y_grid_scaled*np.cos(np.deg2rad(wd_slope))
        hkn_site_bathymetry_grid_scaled = bathymetry_lb+(bathymetry_ub-bathymetry_lb)*(np.max(x_grid_rot)-x_grid_rot)/(np.max(x_grid_rot)-np.min(x_grid_rot))

    else:

        if bathymetry_range is None:
            hkn_site_bathymetry_grid_scaled = hkn_site_bathymetry_grid
        else:
            # max and min values within the entire grid (not limited within the farm boundaries)
            bathymetry_ub = bathymetry_range[0]
            bathymetry_lb = bathymetry_range[1]

            # calculate scaling values
            f_scaling_bathymetry = (bathymetry_ub-bathymetry_lb)/(np.max(hkn_site_bathymetry_grid)-np.min(hkn_site_bathymetry_grid))
            offset_bathymetry = (bathymetry_ub-np.mean(hkn_site_bathymetry_grid))-(np.max(hkn_site_bathymetry_grid)-np.mean(hkn_site_bathymetry_grid))*(bathymetry_ub-bathymetry_lb)/(np.max(hkn_site_bathymetry_grid)-np.min(hkn_site_bathymetry_grid))

            # scale bathymetry grid
            hkn_site_bathymetry_grid_scaled = offset_bathymetry+np.mean(hkn_site_bathymetry_grid)+(hkn_site_bathymetry_grid-np.mean(hkn_site_bathymetry_grid))*f_scaling_bathymetry


    return hkn_site_x_grid_scaled,hkn_site_y_grid_scaled,hkn_site_bathymetry_grid_scaled



def generate_HKNscaled_substation_cables():

    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]

    cable_specs = [
        {"diameter_mm2": 185, "capacity_NrT": 3, "cost_€_m": 368.9},
        {"diameter_mm2": 400, "capacity_NrT": 5, "cost_€_m": 428.9},
        {"diameter_mm2": 1000, "capacity_NrT": 7, "cost_€_m": 737.1}
    ]

    return x_sub,y_sub,cable_specs


def generate_HKNscaled_turbine():
    
    # define wind turbine
    ws_turbine = np.array([0, 2, 3, 3.54953237, 4.067900771, 4.553906848, 5.006427063, 5.424415288, 5.806905228, 6.153012649, 6.461937428, 6.732965398, 6.965470002, 7.158913742, 7.312849418, 7.426921164, 7.500865272, 7.534510799, 7.541241633, 7.58833327, 7.675676842, 7.803070431, 7.970219531, 8.176737731, 8.422147605, 8.70588182, 9.027284445, 9.385612468, 9.780037514, 10.20964776, 10.67345004, 11.13492728, 11.17037214, 11.6992653, 12.25890683, 12.84800295, 13.46519181, 14.10904661, 14.77807889, 15.470742, 16.18543466, 16.92050464, 17.67425264, 18.44493615, 19.23077353, 20.02994808, 20.8406123, 21.66089211, 22.4888912, 23.32269542, 24.1603772, 25])
    p_turbine = np.array([0, 0, 0.375538649, 0.635644519, 1.014237244, 1.473497284, 1.996315118, 2.571170922, 3.179901342, 3.803380108, 4.421435886, 5.013265683, 5.55992078, 6.043266596, 6.446694086, 6.756659289, 6.962663346, 7.057781739, 7.076914762, 7.211728426, 7.466186181, 7.847828144, 8.367545594, 9.039659135, 9.882192126, 10.91883042, 12.17633054, 13.68438108, 15.97152541, 18.12124994, 20.20010629, 22, 21.99998265, 21.99998772, 22.00000215, 22.0000002, 21.99990199, 22.00008648, 22.0000445, 22.0000247, 22.00001503, 22.00000988, 22.00000673, 22.00000464, 22.00000305, 22.00000173, 22.00000068, 22.000396, 21.99989367, 22.00000109, 22.00000282, 22.00000531])*1e3
    ct_turbine = np.array([0, 0, 0.781330462, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.856986677, 0.774831279, 0.751237387, 0.687365371, 0.63157364, 0.620285375, 0.500707415, 0.417418767, 0.352496626, 0.300020593, 0.256857131, 0.220980221, 0.190956234, 0.165702375, 0.144374588, 0.126299006, 0.110930462, 0.09782423, 0.086614854, 0.07700031, 0.068731176, 0.061595491, 0.055424038, 0.050069939, 0.045412126])
    wind_turbine = WindTurbine(name='IEA22MW',
                    diameter=283.2,
                    hub_height=170.0,
                    powerCtFunction=PowerCtTabular(ws_turbine,p_turbine,'kW',ct_turbine))    
    return wind_turbine


def get_HKNscaled_price_data(year=2030,norm=False):
    with open(f'input_folder/HKN_data_and_tools/HKN_price_data.pkl', 'rb') as f:
        HKN_price_data = pickle.load(f)
    if norm:
        return HKN_price_data[f'price_mat_{year}_norm']
    else:
        return HKN_price_data[f'price_mat_{year}']





class CaseStudy():

    def __init__(self,id_case,het_wr_data):

        if het_wr_data:
            match id_case:
                case 'HKNscaled':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry()
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_lowD':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site(f_scaling_density=np.sqrt(2))
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(f_scaling_density=np.sqrt(2))
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_highD':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site(f_scaling_density=1/np.sqrt(3))
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(f_scaling_density=1/np.sqrt(3))
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_highB':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(bathymetry_range=[-10.,-60.])
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_highSlopedB':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(bathymetry_range=[-10.,-60.],use_sloped_profile=True)
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case _:
                    raise TypeError('Case study not defined')

        else:
            match id_case:
                case 'HKNscaled':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site_aWR()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry()
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_lowD':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site_aWR()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(f_scaling_density=np.sqrt(2))
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_highD':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site_aWR()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(f_scaling_density=1/np.sqrt(3))
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_highB':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site_aWR()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(bathymetry_range=[-10.,-60.])
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case 'HKNscaled_highSlopedB':
                    self.wind_turbine = generate_HKNscaled_turbine()
                    self.ws_rated = 11.
                    self.site = generate_HKNscaled_site_aWR()
                    self.site_x_grid,self.site_y_grid,self.site_bathymetry_grid = generate_HKNscaled_bathymetry(bathymetry_range=[-10.,-60.],use_sloped_profile=True)
                    self.x_sub,self.y_sub,self.cable_specs = generate_HKNscaled_substation_cables()
                    self.price_lk = get_HKNscaled_price_data(year=2030,norm=False)       # assumed that price_mat_lk has the same ws and wd discretization
                    self.price_norm_lk = get_HKNscaled_price_data(year=2030,norm=True)   # assumed that price_mat_lk has the same ws and wd discretization
                case _:
                    raise TypeError('Case study not defined')
                

def construct_wfm(case_study,deficit_model,superposition_model,rotavg_model,deflection_model,turbulence_model):

    turbo_park_flag = False
    match deficit_model:
        case 'TurbOPark':
            turbo_park_flag = True
        case 'Jensen':
            wake_deficitModel = NOJDeficit()
        case 'Bastankhah':
            wake_deficitModel = BastankhahGaussianDeficit()
        case 'Zong':
            wake_deficitModel = ZongGaussianDeficit()
        case _:
            raise TypeError('Wake deficit model not defined')
        
    match superposition_model:
        case 'SquaredSum':
            wake_superpositionModel = SquaredSum()
        case 'LinearSum':
            wake_superpositionModel = LinearSum()
        case _:
            raise TypeError('Superposition model not defined')
        
    match rotavg_model:
        case 'Gaussian':
            rotor_avgModel = GaussianOverlapAvgModel()
        case 'RotorCenter':
            rotor_avgModel = RotorCenter()
        case 'CGI':
            rotor_avgModel = CGIRotorAvg()
        case _:
            raise TypeError('Rotor average model not defined')
        
    match deflection_model:
        case 'Jimenez':
            wake_deflectionModel = JimenezWakeDeflection()
        case None:
            wake_deflectionModel = None
        case _:
            raise TypeError('Deflection model not defined')
    
    match turbulence_model:
        case 'STF':
            turbulenceModel = STF2017TurbulenceModel()
        case None:
            turbulenceModel = None
        case _:
            raise TypeError('Turbulence model not defined')
    
    
    if turbo_park_flag:
        wake_deficitModel = TurboGaussianDeficit(
            ct2a=ct2a_mom1d,
            groundModel=Mirror(superpositionModel=SquaredSum()),#wake_superpositionModel),
            rotorAvgModel=rotor_avgModel,
            ctlim=0.96)
        wake_deficitModel.WS_key = 'WS_jlk'     # Ørsted scales the deficit with respect to the ambient wind speed of the downstream turbine
        wfm = PropagateDownwind(case_study.site, case_study.wind_turbine,
                                    wake_deficitModel=wake_deficitModel,
                                    superpositionModel=wake_superpositionModel,
                                    deflectionModel=wake_deflectionModel,
                                    rotorAvgModel=rotor_avgModel)
    else:
        wfm = PropagateDownwind(case_study.site, case_study.wind_turbine,
                                    wake_deficitModel=wake_deficitModel,
                                    superpositionModel=wake_superpositionModel,
                                    deflectionModel=wake_deflectionModel,
                                    rotorAvgModel=rotor_avgModel,
                                    turbulenceModel=turbulenceModel)
    return wfm



def discretize_flow(wd_binsize,ws_binsize):
    wd_array = np.arange(0,360,wd_binsize)
    if ws_binsize=='8ms':
        ws_array = np.array([8.])
    else:
        ws_array = np.arange(3,26,ws_binsize)
    return wd_array,ws_array
    



class ObjFuncWrapper():

    def __init__(self,
                 id_case,
                 het_wr_data=True,
                 deficit_model='TurbOPark',
                 superposition_model='SquaredSum',
                 rotavg_model='Gaussian',
                 deflection_model=None,
                 turbulence_model=None,
                 wd_binsize=5,
                 ws_binsize=1,
                 use_WeibullSamples=False,
                 use_geomYaw=False,
                 run_OptiWindNet=False,
                 run_DETECT=False,
                 obj=['AEP'],
                 maximize=[True],
                 n_cpu=None,
                 parallel_execution=False,
                 min_d_D=1.,
                 min_d_D_soft=0.,
                 value_per_turbine=True,
                 use_input_cables=False,
                 TurCon=None,
                 u=None,
                 v=None,
                 cab_data=None,
                 use_inputYaw=False,
                 yaw_input=None,
                 ):
        
        # define case study
        case_study = CaseStudy(id_case=id_case,het_wr_data=het_wr_data)

        # define wind farm model
        wfm = construct_wfm(case_study,deficit_model,superposition_model,rotavg_model,deflection_model,turbulence_model)

        # define flow discretization
        wd_array,ws_array = discretize_flow(wd_binsize,ws_binsize)
        
        # define aep wrapper
        if use_WeibullSamples:
            aep_calculator = AEPCalculator_DETECT_Weibull(wfm=wfm,
                                                wd_array=wd_array,
                                                ws_array=ws_array,
                                                ws_rated=case_study.ws_rated,
                                                wind_turbine=case_study.wind_turbine,
                                                use_geomYaw=use_geomYaw,
                                                price_lk=case_study.price_lk,
                                                price_norm_lk=case_study.price_norm_lk,
                                                x_sub=case_study.x_sub,
                                                y_sub=case_study.y_sub)
        else:
            aep_calculator = AEPCalculator_DETECT(wfm=wfm,
                                                wd_array=wd_array,
                                                ws_array=ws_array,
                                                ws_rated=case_study.ws_rated,
                                                wind_turbine=case_study.wind_turbine,
                                                use_geomYaw=use_geomYaw,
                                                price_lk=case_study.price_lk,
                                                price_norm_lk=case_study.price_norm_lk,
                                                use_inputYaw=use_inputYaw,
                                                yaw_input=yaw_input,
            )
        
        # define water depth calculator wrapper
        water_depth_calculator = WaterDepthCalculator_DETECT(site_x_grid=case_study.site_x_grid,
                                                             site_y_grid=case_study.site_y_grid,
                                                             site_bathymetry_grid=case_study.site_bathymetry_grid)
        # define cable calculator wrapper
        if run_OptiWindNet:
            if use_input_cables:
                cable_calculator = CableCalculator_inputCab(x_sub=case_study.x_sub,
                                                            y_sub=case_study.y_sub,
                                                            cable_specs=case_study.cable_specs,
                                                            TurCon=TurCon,
                                                            u=u,
                                                            v=v,
                                                            cab_data=cab_data)
            else:
                cable_calculator = CableCalculator_DETECT(x_sub=case_study.x_sub,
                                                        y_sub=case_study.y_sub,
                                                        cable_specs=case_study.cable_specs)
        else:
            cable_calculator = None

        # define python wrapper
        python_xy_wrapper = PopEval_python_XYwrapper(aep_calculator,
                                                    cable_calculator,
                                                    water_depth_calculator,
                                                    min_d=min_d_D*case_study.wind_turbine.diameter(),
                                                    parallel_execution=parallel_execution,
                                                    n_cpu=n_cpu)
        # define matlab wrapper
        if run_DETECT:
            matlab_eng = matlab.engine.start_matlab()
            matlab_eng.addpath(matlab_eng.genpath('input_folder/matlab_funcs'), nargout=0) # IMPORTANT: here you need to add the location of the DETECT FOLDER
            matlab_xy_wrapper = PopEval_matlab_XYwrapper(matlab_eng,
                                                        parallel_execution=False, # always run Matlab in series
                                                        n_cpu=n_cpu)
        else:
            matlab_xy_wrapper = None

        # define objective function wrapper
        self.obj_func_XYwrapper = ObjFunction_XYwrapper(python_xy_wrapper=python_xy_wrapper,
                                                   matlab_xy_wrapper=matlab_xy_wrapper,
                                                   value_per_turbine=value_per_turbine,
                                                   output_keys = obj,
                                                   maximize=maximize,
                                                   min_d_bounds=np.array([min_d_D,min_d_D_soft])*case_study.wind_turbine.diameter())


    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.obj_func_XYwrapper(x_mat,y_mat,gen_number_norm)



#%%
# import specific wrappers

#%%
# HKNscaled


class ObjFunc_HKNscaled():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNscaled_GY():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)




class EvalFunc_HKNscaled():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_GY():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)




class EvalFunc_HKNscaled_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_GY_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)



#%%
# HKNscaled - high power density (HPD)

class ObjFunc_HKNscaled_HPD():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNscaled_HPD_GY():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)
    

class EvalFunc_HKNscaled_HPD_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_HPD_GY_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)



#%%
# HKNscaled - low power density (LPD)

class ObjFunc_HKNscaled_LPD():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_lowD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNscaled_LPD_GY():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_lowD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_LPD_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_lowD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_LPD_GY_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_lowD',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


#%%
# HKNscaled - high bathymetry difference (HBD)

class ObjFunc_HKNscaled_HBD():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNscaled_HBD_GY():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_HBD_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_HBD_GY_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


#%%
# HKNscaled - high sloped bathymetry difference (HSBD)

class ObjFunc_HKNscaled_HSBD():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highSlopedB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNscaled_HSBD_GY():

    def __init__(self,obj,maximize,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highSlopedB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)

class EvalFunc_HKNscaled_HSBD_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highSlopedB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNscaled_HSBD_GY_inputCab():

    def __init__(self,obj,maximize,TurCon,u,v,cab_data,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case='HKNscaled_highSlopedB',
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=3,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


#%%
# input yaw and cables - generic functions for all case studies

class ObjFunc_HKNallCases():

    def __init__(self,id_case,obj,maximize,wd_binsize=3,ws_binsize=1,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case=id_case,
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=wd_binsize,
                                           ws_binsize=ws_binsize,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNallCases_GY():

    def __init__(self,id_case,obj,maximize,wd_binsize=3,ws_binsize=1,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case=id_case,
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=wd_binsize,
                                           ws_binsize=ws_binsize,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize)

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class ObjFunc_HKNallCases_OY():

    def __init__(self,id_case,obj,maximize,yaw_input,wd_binsize=3,ws_binsize=1,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case=id_case,
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=4.,
                                           wd_binsize=wd_binsize,
                                           ws_binsize=ws_binsize,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=False,
                                           obj=obj,
                                           maximize=maximize,
                                           use_inputYaw=True,
                                           yaw_input=yaw_input
                                           )

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)
    


#%%
# evaluation functions

#%%
# input yaw and cables - generic functions for all case studies

class EvalFunc_HKNallCases_inputCab():

    def __init__(self,id_case,obj,maximize,TurCon,u,v,cab_data,wd_binsize=3,ws_binsize=1,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case=id_case,
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=wd_binsize,
                                           ws_binsize=ws_binsize,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data,
                                           )

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNallCases_inputCab_GY():

    def __init__(self,id_case,obj,maximize,TurCon,u,v,cab_data,wd_binsize=3,ws_binsize=1,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case=id_case,
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=wd_binsize,
                                           ws_binsize=ws_binsize,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=True,
                                           obj=obj,
                                           maximize=maximize,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data,
                                           )

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)


class EvalFunc_HKNallCases_inputCabYaw():

    def __init__(self,id_case,obj,maximize,TurCon,u,v,cab_data,yaw_input,wd_binsize=3,ws_binsize=1,n_cpu=1,parallel_execution=False):
        self.f_obj_wrapper= ObjFuncWrapper(id_case=id_case,
                                           n_cpu=n_cpu,
                                           parallel_execution=parallel_execution,
                                           min_d_D=1.,
                                           min_d_D_soft=2.,
                                           wd_binsize=wd_binsize,
                                           ws_binsize=ws_binsize,
                                           run_OptiWindNet=True,
                                           run_DETECT=True,
                                           deflection_model='Jimenez',
                                           use_geomYaw=False,
                                           obj=obj,
                                           maximize=maximize,
                                           use_inputYaw=True,
                                           yaw_input=yaw_input,
                                           value_per_turbine=False,
                                           use_input_cables=True,
                                           TurCon=TurCon,
                                           u=u,
                                           v=v,
                                           cab_data=cab_data,
                                           )

    def __call__(self,x_mat,y_mat,gen_number_norm=1.):
        return self.f_obj_wrapper(x_mat,y_mat,gen_number_norm)
    
