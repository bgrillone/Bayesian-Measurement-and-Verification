import arviz  as az
import matplotlib.pyplot as plt
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings
from sklearn.metrics import mean_squared_error
from math import sqrt
from pymc3.variational.callbacks import CheckParametersConvergence

RANDOM_SEED = 8924

# Data import
df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/Id50_preprocessed2.csv", index_col=0)

# Check if there's NAs
df.isna().sum()

# Preprocessing
df["log_v"] = log_electricity = np.log(df["total_electricity"]).values
total_electricity = df.total_electricity.values

# Multilevel model
# Create local variables (clusters need to start from 0)
n_hours = len(df.index)
df.t = pd.to_datetime(pd.Series(df.t))
dayhour = df['t'].dt.hour
temperature = df.outdoor_temp
outdoor_temp_c = df.outdoor_temp_c
outdoor_temp_h = df.outdoor_temp_h
coords = {"obs_id": np.arange(temperature.size)}
daypart_fs_sin_1 = df.daypart_fs_sin_1
daypart_fs_sin_2 = df.daypart_fs_sin_2
daypart_fs_sin_3 = df.daypart_fs_sin_3
daypart_fs_sin_4 = df.daypart_fs_sin_4
daypart_fs_sin_5 = df.daypart_fs_sin_5
daypart_fs_cos_1 = df.daypart_fs_cos_1
daypart_fs_cos_2 = df.daypart_fs_cos_2
daypart_fs_cos_3 = df.daypart_fs_cos_3
daypart_fs_cos_4 = df.daypart_fs_cos_4
daypart_fs_cos_5 = df.daypart_fs_cos_5

# Intercept, Fourier, temperatures, with profile and temperature clustering

with pm.Model(coords=coords) as complete_pooling:

    fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1, dims="obs_id")
    fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2, dims="obs_id")
    fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3, dims="obs_id")
    fs_sin_4 = pm.Data("fs_sin_4", daypart_fs_sin_4, dims="obs_id")
    fs_sin_5 = pm.Data("fs_sin_5", daypart_fs_sin_5, dims="obs_id")
    fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1, dims="obs_id")
    fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2, dims="obs_id")
    fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3, dims="obs_id")
    fs_cos_4 = pm.Data("fs_cos_4", daypart_fs_cos_4, dims="obs_id")
    fs_cos_5 = pm.Data("fs_cos_5", daypart_fs_cos_5, dims="obs_id")

    cooling_temp = pm.Data("cooling_temp", outdoor_temp_c, dims="obs_id")
    heating_temp = pm.Data("heating_temp", outdoor_temp_h, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    btc = pm.Normal("btc", mu=0.0, sigma=1.0)
    bth = pm.Normal("bth", mu=0.0, sigma=1.0)

    # Varying slopes:
    bs1 = pm.Normal("bs1", mu=0.0, sigma=1.0)
    bs2 = pm.Normal("bs2", mu=0.0, sigma=1.0)
    bs3 = pm.Normal("bs3", mu=0.0, sigma=1.0)
    bs4 = pm.Normal("bs4", mu=0.0, sigma=1.0)
    bs5 = pm.Normal("bs5", mu=0.0, sigma=1.0)
    bc1 = pm.Normal("bc1", mu=0.0, sigma=1.0)
    bc2 = pm.Normal("bc2", mu=0.0, sigma=1.0)
    bc3 = pm.Normal("bc3", mu=0.0, sigma=1.0)
    bc4 = pm.Normal("bc4", mu=0.0, sigma=1.0)
    bc5 = pm.Normal("bc5", mu=0.0, sigma=1.0)

    # Expected value per county:
    mu = a + bs1 * fs_sin_1 + bs2 * fs_sin_2 + bs3 * fs_sin_3 + bs4 * fs_sin_4 + bs5 * fs_sin_5 + bc1 * fs_cos_1 + \
         bc2 * fs_cos_2 + bc3 * fs_cos_3 + bc4 * fs_cos_4 + bc5 * fs_cos_5 + btc * cooling_temp + bth * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

# Graphviz visualisation
complete_pooling_graph = pm.model_to_graphviz(complete_pooling)
complete_pooling_graph.render(filename='img/complete_pooling_hourly')

#Fitting without sampling
with complete_pooling:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    complete_pooling_trace = approx.sample(1000)
    complete_pooling_idata = az.from_pymc3(complete_pooling_trace)

az.summary(complete_pooling_idata, round_to=2)
az.plot_trace(complete_pooling_idata)
plt.show()


complete_pooling_a_means = np.mean(complete_pooling_trace['a'])
complete_pooling_bs1_means = np.mean(complete_pooling_trace['bs1'])
complete_pooling_bs2_means = np.mean(complete_pooling_trace['bs2'], axis =0)
complete_pooling_bs3_means = np.mean(complete_pooling_trace['bs3'], axis =0)
complete_pooling_bs4_means = np.mean(complete_pooling_trace['bs4'], axis =0)
complete_pooling_bs5_means = np.mean(complete_pooling_trace['bs5'], axis =0)
complete_pooling_bc1_means = np.mean(complete_pooling_trace['bc1'], axis =0)
complete_pooling_bc2_means = np.mean(complete_pooling_trace['bc2'], axis =0)
complete_pooling_bc3_means = np.mean(complete_pooling_trace['bc3'], axis =0)
complete_pooling_bc4_means = np.mean(complete_pooling_trace['bc4'], axis =0)
complete_pooling_bc5_means = np.mean(complete_pooling_trace['bc5'], axis =0)

complete_pooling_bth_means = np.mean(complete_pooling_trace['bth'], axis = 0)
complete_pooling_btc_means = np.mean(complete_pooling_trace['btc'], axis = 0)
# Create array with predictions
complete_pooling_predictions = []
# Create array with bounds
complete_pooling_hdi = az.hdi(complete_pooling_idata)
complete_pooling_mean_lower = []
complete_pooling_mean_higher= []
complete_pooling_lower = []
complete_pooling_higher= []


for hour, row in df.iterrows():
                                    complete_pooling_predictions.append(complete_pooling_a_means + \
                                                                       complete_pooling_bs1_means* daypart_fs_sin_1[hour] + \
                                                                       complete_pooling_bs2_means * daypart_fs_sin_2[hour] + \
                                                                       complete_pooling_bs3_means * daypart_fs_sin_3[hour] + \
                                                                       complete_pooling_bs4_means * daypart_fs_sin_4[hour] + \
                                                                       complete_pooling_bs5_means * daypart_fs_sin_5[hour] + \
                                                                       complete_pooling_bc1_means * daypart_fs_cos_1[hour] + \
                                                                       complete_pooling_bc2_means * daypart_fs_cos_2[hour] + \
                                                                       complete_pooling_bc3_means * daypart_fs_cos_3[hour] + \
                                                                       complete_pooling_bc4_means * daypart_fs_cos_4[hour] + \
                                                                       complete_pooling_bc5_means * daypart_fs_cos_5[hour] + \
                                                                       complete_pooling_bth_means * outdoor_temp_h[hour] + \
                                                                       complete_pooling_btc_means * outdoor_temp_c[hour])



# Calculate prediction error
predictions = np.exp(complete_pooling_predictions)
mse = mean_squared_error(df.total_electricity, predictions)
rmse = sqrt(mse)
cvrmse = rmse/df.total_electricity.mean()

# PLOTS

# output to static HTML file
output_file("predictions.html")

# predictions vs real log scale
p = figure(plot_width=800, plot_height=400)

p.circle(df.index, complete_pooling_predictions, size=5, color="navy", alpha=0.5)
p.circle(df.index, log_electricity, size=5, color="orange", alpha=0.2)
show(p)

# predictions vs real

p2 = figure(plot_width=800, plot_height=400)

p2.circle(df.index, predictions, size=5, color="navy", alpha=0.5)
p2.circle(df.index, df.total_electricity, size = 5, color="orange", alpha=0.2)
show(p2)

# predictions vs real lineplot

p3 = figure(plot_width=800, plot_height=400)

p3.line(df.index, predictions, color="navy", alpha=0.8)
p3.line(df.index, df.total_electricity, color="orange", alpha=0.8)
show(p3)

