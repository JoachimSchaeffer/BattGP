# %%

from src.batt_data.batt_data import BattData
from src.batt_data.data_utils import build_data_cache, read_cell_characteristics
from src.batt_models.battgp_full import BattGP_Full
from src.batt_models.battgp_spatiotemporal import BattGP_SpatioTemporal
from src.batt_models.fault_probabilities import calc_fault_probabilities
from src.batt_models.plotting import plot_cell_r0_predictions, plot_fault_probabilities
from src.batt_models.ref_strategy import RefStrategy
from src.operating_point import Op
from src.path_setup import setup_paths

# %% Cache battery data

# The battery data is cached automatically at the first usage of each battery.
# However one may prefer to wait for the caching one time upfront.
# The constant PATH_FIELDDATA_DATA in the module config must point to the folder
# containing the unzipped data or directly to the zip file with the data.
# Also, check that the constant PATH_DATA_CACHE points to an appropriate path for the
# cache.

setup_paths()
build_data_cache()

# %% Load dataset

cell_characteristics = read_cell_characteristics()
batt_data = BattData("18", cell_characteristics)


# %% Spatiotemporal GP

# construct spatiotemporal GP wrapper (no work is done by this)
bgp = BattGP_SpatioTemporal(
    batt_data,
    sampling_time_sec=3600,
    ref_strategy=RefStrategy(Op(-15, 90, 25)),
    max_batch_size=1000,
    basis_vector_strategy="kmeans",
    nbasis=[60],
)
# Calculate resistance at reference point

# Currently one cannot/should not call the predict method several times on the
# same object, as the state of the kalman filter is not resetted before each call.
# (With smooth=True this will also cause an exception at the end of all calculations.)
gp_res = bgp.predict_cell_r0_op(smooth=True)

plot_cell_r0_predictions(gp_res, single_plot=True)


# %% Perform fault analysis

# The values for the band and the threshold are expected to be in Ohm (and not
# mOhm) in order to be consistent with the GP results
df_faults = calc_fault_probabilities(
    gp_res, causal=True, r0_band=0.55e-3, r0_upper_threshold=2.0e-3
)

plot_fault_probabilities(df_faults, gp_res, causal=True)

# %% Full GP

# Load a smaller dataset
cell_characteristics = read_cell_characteristics()
batt_data = BattData("3", cell_characteristics)


# %% Construct full GP wrapper (no work is done by this)

bgp = BattGP_Full(batt_data, ref_strategy=RefStrategy(Op(-15, 90, 25)))

# %% Optional: Train hyperparameter
# Depending on the system (available data) and number of datapoints,
# the hyperparameters might be physically not very meaninguful.
# For best results use a GPU, 40K datapoints (e.g., an A100 GPU) and a system with
# enough data points (e.g., system 6 or 8).
train_hyperparameters = False

if train_hyperparameters:
    bgp.train_hyperparameters()

# %% Calculate resistance at reference point
gp_res = bgp.predict_cell_r0_op()
plot_cell_r0_predictions(gp_res, single_plot=True)

# Perform fault analysis
# Note: The fault probabilities of the full GP can not be used in an online setting due
# to the dependy of p(t) on data points in the future.
df_faults = calc_fault_probabilities(
    gp_res, causal=False, r0_band=0.55e-3, r0_upper_threshold=2.0e-3
)

plot_fault_probabilities(df_faults, gp_res, causal=True)


# %% Use trained hyperparameter for spatial temporal GP

# Here, we use the parameters of cell 1 for all cells as well as the pack model.
# If one of the BattGP classes is used, you can only specifiy "global"
# parameters. In case of the full GP, it is possible to change the GP parameters for
# each cell individually after the construction of the BattGP object. In the spatio-
# temporal case this is not possible, as the spatio-temporal cell model doesn't provide
# a set_parameters method (yet). This is also an interesting avenue for future work.

params = bgp.get_parameters()

# Only keep hyperparameters of the GP
# (This is a little unfortunate, that the parameters contains also the parameters used
# for the hyperparameter optimization. One could provide a get_hyperparameter method?)
gp_param_names = [
    "noise_variance",
    "outputscale_wiener",
    "outputscale_rbf",
    "lengthscale_rbf",
]
gp_params = {k: v for (k, v) in params.items() if k in gp_param_names}

bgp_st = BattGP_SpatioTemporal(
    batt_data,
    sampling_time_sec=3600,
    ref_strategy=RefStrategy(Op(-15, 90, 25)),
    **gp_params
)

gp_res = bgp_st.predict_cell_r0_op()

plot_cell_r0_predictions(gp_res, single_plot=True)

# %%
