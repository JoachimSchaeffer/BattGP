from typing import Optional

import numpy as np
import torch

# Configuration file for the LFP battery data
# If you want to load data from different battery systems, create a new config file
# and adapt the paths and settings accordingly


# #####################MAKE CHANGES HERE TO USE OTHER DATA################################
# Path to OCV data
PATH_FIELDDATA_CELL_CHARACTERISTIC = "data/ocv_linear_approx.csv"
# Folder with battery raw data
# PATH_FIELDDATA_DATA: Optional[str] = "data/field_data"
PATH_FIELDDATA_DATA: Optional[str] = "data/field_data.zip"


# #############################HIGH LEVEL MODEL SETTINGS##################################
# Options for hyperparameter optimization: "random_data"

PATH_RESULTS = "results/"
PATH_FIGURES_DATA_VIS = "results/data_vis/"
PATH_DATA_CACHE: Optional[str] = "data/cache"

GAP_REMOVAL = True
GAP_REMOVAL_THRESHOLD = 90  # Days

NB_DATAPOINTS = 1000  # 16000
# Options for the GP model
GP_SETTINGS = {
    "mode": "random",
    "nb_data_points": NB_DATAPOINTS,
}

# Hyperpamerters suitable for the LFP battery systems analysed in the associated article

# # Run: 6, 8: Median:
NOISE_VARIANCE = (2.33e-6,)
OUTPUTSCALE_WIENER = 4.23e-13
# OUTPUTSCALE_WIENER = 1e-11
OUTPUTSCALE_RBF = 0.0099
LENGTHSCALE_RBF = (12.11, 33.75, 45.14)


PARAMS_LFP = {
    "noise_variance": NOISE_VARIANCE,
    "outputscale_wiener": OUTPUTSCALE_WIENER,
    "outputscale_rbf": OUTPUTSCALE_RBF,
    "lengthscale_rbf": LENGTHSCALE_RBF,
}

NOISE_VARIANCE_RANGE = (0, 1e5)
OUTPUTSCALE_WIENER_RANGE = (1e-15, 1e4)
OUTPUTSCALE_RBF_RANGE = (1e-10, 1e6)
LENGTHSCALE_RBF_RANGE = ((1e-5, 1e4), (1e-5, 1e4), (1e-5, 1e4))

# Hyperparameter thresholds for the GP Kernels
HYPER_OPT_PARAMS = {
    "optimize": False,  # True or False: If false the hyperparameters from above are used
    "parallelize": False,  # If true hyperparameter optimization is parallelized for cells
    "opt_algorithm": "torch_adam",  # "torch_lbfgs", "botorch_lbfgs_B", "torch_adam"
    "mode": "random_data",
    "nb_data_points": NB_DATAPOINTS,
    "verbose": True,
    "save": True,
    "save_path": PATH_RESULTS,
}

# GPtorch config
OPTIM_MAX_ITER = 500
OPTIM_REL_TOL = 1e-5
OPTIM_LR = 1
DTYPE = torch.float64


# ####################################FAULT PARAMETERS####################################
BAND = 0.25


# ########################CHANGES BELOW CHANGE THE MODEL BEHAVIOUR########################
##########################################################################################
# Global paths for saving models, visualizations and data, no need to change

# control boolean for SMOOTH MIN WINDOW application
SMOOTHING_ENABLED = False

# used to replace the original time series with a X minute mean to reduce computational
# load
SMOOTH_MIN_WINDOW = "5s"

# [minutes] sampling every X minutes of the GP mean prediction
# Warning: lower value is PERFORMANCE CRITICAL
SAMPLING_R0_MEAN = 60

# ################ DATA SELECTION CRITERIA ##################
# Initial trials:
MODE = "discharge"
SOC_UPPER_LIMIT = 95
SOC_LOWER_LIMIT = 40
# The cell shouldnt be at rest, becasue at too low currents the diferences between cells will be harder to detect.
Ibat_UPPER_LIMIT = -5
Ibat_LOWER_LIMIT = -80
# Converter limit! Will filter out segments wehrer the balacing current exceed this value!
CNV_LIMIT = 20
# number of decimals to round to for operating point values (T, I, SOC)
DECIMALS_OP = 2
# number of decimals to round to for R0 values
DECIMALS_R0 = 7
T_UPPER_LIMIT = 100
T_LOWER_LIMIT = 10

# map cell number to temperature sensor column
TEMP_MAP = {
    1: "T_Cell_1_2",
    2: "T_Cell_1_2",
    3: "T_Cell_3_4",
    4: "T_Cell_3_4",
    5: "T_Cell_5_6",
    6: "T_Cell_5_6",
    7: "T_Cell_7_8",
    8: "T_Cell_7_8",
}

# Color list of 8 Colors as an iterable
# Source: https://sunlightfoundation.com/2014/03/12/datavizguide/
COLORS_RGB = [
    (227, 186, 34),
    (230, 132, 42),
    (19, 123, 128),
    (142, 109, 138),
    (154, 62, 37),
    (21, 107, 144),
    (112, 130, 89),
    (92, 129, 0),
    (160, 183, 0),
]

# Turn the RGB values Above into hex values
COLORS = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for (r, g, b) in COLORS_RGB]


# List of line styles for plotting
LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (5, (10, 3)),
    (0, (3, 1, 1, 1, 1, 1, 1, 1)),
    (0, (5, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
]

PLOT_WEAKEST_LINK = True

# ### Test ref point and basis vectors to compare implementations

REF_POINT_TEST = [-15, 90, 25]

BASIS_VECTORS_TEST = np.array(
    [
        REF_POINT_TEST,
        [-64.25477443, 83.28817391, 39.4224058],
        [-10.80410898, 78.6666667, 30.08442331],
        [-74.24327292, 74.78568326, 33.21828561],
        [-47.06353377, 91.02846591, 32.09091656],
        [-62.51796286, 65.82102257, 45.00314765],
        [-20.53822455, 88.61687243, 21.99115226],
        [-64.7591052, 91.59194317, 28.27214496],
        [-36.49631582, 78.47533602, 39.66440937],
        [-43.27929647, 53.47728045, 36.8471867],
        [-13.26760761, 66.52815601, 50.67186305],
        [-64.96876897, 72.27563088, 27.35068929],
        [-74.63171892, 90.40885569, 41.19193989],
        [-8.43191017, 89.64784873, 33.55010045],
        [-74.47809595, 81.70302401, 25.28060273],
        [-51.61732558, 78.25407242, 40.23841932],
        [-16.82556806, 89.7177934, 32.80815529],
        [-27.37164598, 54.04143692, 36.1206507],
        [-74.55399196, 50.77467882, 38.18493576],
        [-62.7071066, 90.03861202, 44.779653],
        [-12.68433632, 89.38658803, 43.94535107],
        [-64.28531008, 83.03207683, 24.55407413],
        [-58.41834649, 52.11911844, 51.19643772],
        [-62.69200037, 79.02408074, 47.98075758],
        [-73.66109677, 84.29672819, 49.18488824],
        [-44.63641919, 88.60949694, 43.45261949],
        [-9.75936422, 90.2207237, 22.95667786],
        [-50.05753047, 69.34597067, 29.38482871],
        [-11.38552943, 68.1434364, 37.23700286],
        [-64.6681922, 91.65758198, 36.03602746],
        [-31.29355722, 88.7670797, 22.07695578],
        [-40.92882773, 68.94495651, 53.76951164],
        [-26.08149943, 90.34392493, 31.4753645],
        [-74.46498508, 91.13693204, 21.65863125],
        [-23.45882257, 79.88596948, 33.45257488],
        [-74.60733971, 76.11734541, 44.07854291],
        [-26.97420399, 66.27269173, 42.0986053],
        [-47.63959782, 65.32687204, 41.53494188],
        [-62.4613436, 78.30348479, 33.33329078],
        [-42.67652057, 90.0387609, 22.39529299],
        [-23.59419915, 78.03181963, 46.16586314],
        [-65.32850793, 85.38823594, 32.03891998],
        [-61.56847982, 55.76017233, 27.39420775],
        [-63.64399519, 69.86498301, 55.5036387],
        [-11.14923698, 78.98622712, 42.27885969],
        [-26.35246082, 89.20071015, 42.18575233],
        [-63.40160386, 56.53539561, 39.18640621],
        [-55.36961726, 89.0019949, 34.49248688],
        [-73.87717318, 61.14566455, 49.26031806],
        [-74.46847159, 91.14428401, 30.9517958],
        [-65.1092191, 91.00156096, 20.32630171],
        [-36.75855745, 89.31133987, 33.15853565],
        [-74.09558331, 64.73437291, 26.98745763],
        [-54.22461264, 89.91098734, 22.91821479],
        [-74.2718682, 65.74005687, 38.9614417],
        [-13.81476031, 49.74455959, 47.20048935],
        [-19.35472261, 68.25852681, 27.44853683],
        [-63.66202788, 66.33476976, 34.65687235],
        [-64.66196833, 73.90300559, 39.79419603],
        [-12.08794868, 55.97650701, 35.15132122],
        [-64.90876312, 45.79684571, 43.5637439],
        [-74.64509977, 83.33409859, 35.53681094],
        [-46.34072761, 80.9354565, 30.70971338],
        [-34.08768165, 70.72751672, 29.09534578],
        [-33.11856894, 54.26479758, 52.74260125],
    ]
)
