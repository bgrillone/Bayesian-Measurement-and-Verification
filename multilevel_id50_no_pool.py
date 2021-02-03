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

# Daypart 06 17 24
df.t = pd.to_datetime(pd.Series(df.t))
df["daypart"] = np.where(df['t'].dt.hour <= 19,
                         np.where(df['t'].dt.hour <= 15,
                                  np.where(df['t'].dt.hour <= 11,
                                           np.where(df['t'].dt.hour <= 7,
                                                    np.where(df['t'].dt.hour <= 3,0,1),2),3),4),5)
# Create local variables (clusters need to start from 0)
df.s = df.s - 1
clusters = df.s
unique_clusters = clusters.unique()
heat_clusters = df.temp_h_cluster
cool_clusters = df.temp_c_cluster
dayparts = df.daypart
unique_heat_clusters = heat_clusters.unique()
unique_cool_clusters = cool_clusters.unique()
unique_dayparts = dayparts.unique()
n_hours = len(df.index)
temperature = df.outdoor_temp
outdoor_temp_c = df.outdoor_temp_c
outdoor_temp_h = df.outdoor_temp_h
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

coords = {"obs_id": np.arange(temperature.size)}
coords["profile_cluster"] = unique_clusters
coords["heat_cluster"] = unique_heat_clusters
coords["cool_cluster"] = unique_cool_clusters
coords["daypart"] = unique_dayparts


# Intercept, Fourier, temperatures, with profile and temperature clustering -  MINIBATCH

with pm.Model(coords=coords) as no_pooling:
    profile_cluster_idx = pm.Data("profile_cluster_idx", clusters, dims="obs_id")
    heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters, dims="obs_id")
    cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters, dims="obs_id")
    daypart = pm.Data("daypart", dayparts, dims = "obs_id")

    fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1, dims = "obs_id")
    fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2, dims = "obs_id")
    fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3, dims = "obs_id")
    fs_sin_4 = pm.Data("fs_sin_4", daypart_fs_sin_4, dims = "obs_id")
    fs_sin_5 = pm.Data("fs_sin_5", daypart_fs_sin_5, dims = "obs_id")
    fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1, dims = "obs_id")
    fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2, dims = "obs_id")
    fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3, dims = "obs_id")
    fs_cos_4 = pm.Data("fs_cos_4", daypart_fs_cos_4, dims = "obs_id")
    fs_cos_5 = pm.Data("fs_cos_5", daypart_fs_cos_5, dims = "obs_id")

    cooling_temp = pm.Data("cooling_temp", outdoor_temp_c, dims="obs_id")
    heating_temp = pm.Data("heating_temp", outdoor_temp_h, dims="obs_id")


    # Priors:
    a = pm.Normal("a", mu=0.0, sigma=1.0, dims = ("daypart", "profile_cluster"))
    btc = pm.Normal("btc", mu=0.0, sigma=1.0, dims = ("daypart", "cool_cluster"))
    bth = pm.Normal("bth", mu=0.0, sigma=1.0, dims = ("daypart", "heat_cluster"))

    bs1 = pm.Normal("bs1", mu=0.0, sigma=1.0, dims="profile_cluster")
    bs2 = pm.Normal("bs2", mu=0.0, sigma=1.0, dims="profile_cluster")
    bs3 = pm.Normal("bs3", mu=0.0, sigma=1.0, dims="profile_cluster")
    bs4 = pm.Normal("bs4", mu=0.0, sigma=1.0, dims="profile_cluster")
    bs5 = pm.Normal("bs5", mu=0.0, sigma=1.0, dims="profile_cluster")
    bc1 = pm.Normal("bc1", mu=0.0, sigma=1.0, dims="profile_cluster")
    bc2 = pm.Normal("bc2", mu=0.0, sigma=1.0, dims="profile_cluster")
    bc3 = pm.Normal("bc3", mu=0.0, sigma=1.0, dims="profile_cluster")
    bc4 = pm.Normal("bc4", mu=0.0, sigma=1.0, dims="profile_cluster")
    bc5 = pm.Normal("bc5", mu=0.0, sigma=1.0, dims="profile_cluster")


    # Expected value per county:
    mu = a[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[profile_cluster_idx] * fs_sin_2 + \
         bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
         bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
         bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
         bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
         btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims = 'obs_id')


# Graphviz visualisation
no_pooling_graph = pm.model_to_graphviz(no_pooling)
no_pooling_graph.render(filename='img/no_pooling_hourly_nomb')

#Fitting without sampling
with no_pooling:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    no_pooling_trace = approx.sample(1000)
    no_pooling_idata = az.from_pymc3(no_pooling_trace)

az.summary(no_pooling_idata, round_to=2)
az.plot_trace(no_pooling_idata)
plt.show()



# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

no_pooling_a_means = np.mean(no_pooling_trace['a'], axis =0)
no_pooling_bs1_means = np.mean(no_pooling_trace['bs1'], axis =0)
no_pooling_bs2_means = np.mean(no_pooling_trace['bs2'], axis =0)
no_pooling_bs3_means = np.mean(no_pooling_trace['bs3'], axis =0)
no_pooling_bs4_means = np.mean(no_pooling_trace['bs4'], axis =0)
no_pooling_bs5_means = np.mean(no_pooling_trace['bs5'], axis =0)
no_pooling_bc1_means = np.mean(no_pooling_trace['bc1'], axis =0)
no_pooling_bc2_means = np.mean(no_pooling_trace['bc2'], axis =0)
no_pooling_bc3_means = np.mean(no_pooling_trace['bc3'], axis =0)
no_pooling_bc4_means = np.mean(no_pooling_trace['bc4'], axis =0)
no_pooling_bc5_means = np.mean(no_pooling_trace['bc5'], axis =0)

no_pooling_bth_means = np.mean(no_pooling_trace['bth'], axis = 0)
no_pooling_btc_means = np.mean(no_pooling_trace['btc'], axis = 0)
# Create array with predictions
no_pooling_predictions = []
# Create array with bounds
no_pooling_hdi = az.hdi(no_pooling_idata)
no_pooling_mean_lower = []
no_pooling_mean_higher= []
no_pooling_lower = []
no_pooling_higher= []


for hour, row in df.iterrows():
    for cluster_idx in unique_clusters:
        if clusters[hour] == cluster_idx:
            for heat_cluster_idx in unique_heat_clusters:
                if heat_clusters[hour] == heat_cluster_idx:
                    for cool_cluster_idx in unique_cool_clusters:
                        if cool_clusters[hour] == cool_cluster_idx:
                            for daypart_idx in unique_dayparts:
                                if dayparts[hour] == daypart_idx:

                                    no_pooling_predictions.append(no_pooling_a_means[daypart_idx, cluster_idx] + \
                                                                       no_pooling_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                                                       no_pooling_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                                                       no_pooling_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                                                       no_pooling_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                                                       no_pooling_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                                                       no_pooling_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                                                       no_pooling_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                                                       no_pooling_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                                                       no_pooling_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                                                       no_pooling_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                                                       no_pooling_bth_means[daypart_idx, heat_cluster_idx] * outdoor_temp_h[hour] + \
                                                                       no_pooling_btc_means[daypart_idx, cool_cluster_idx] * outdoor_temp_c[hour])


# Calculate prediction error
predictions = np.exp(no_pooling_predictions)
mse = mean_squared_error(df.total_electricity, predictions)
rmse = sqrt(mse)
cvrmse = rmse/df.total_electricity.mean()

# PLOTS

# output to static HTML file
output_file("predictions.html")

# predictions vs real log scale
p = figure(plot_width=800, plot_height=400)

p.circle(df.index, no_pooling_predictions, size=5, color="navy", alpha=0.5)
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

