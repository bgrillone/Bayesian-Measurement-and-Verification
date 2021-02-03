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
df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/Id50_preprocessed2.csv", index_col = 0)

# Check if there's NAs
df.isna().sum()

# Preprocessing
df["log_v"] = log_electricity = np.log(df["total_electricity"]).values
total_electricity = df.total_electricity.values

# Plotting data
measured = df[np.isfinite(df["total_electricity"])].total_electricity
hist, edges = np.histogram(measured, density = True, bins = 50)

def make_plot(title, hist, edges, x):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)

    p.y_range.start = 0
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    return p
  
x = np.linspace (0, 30000, num=3000)
p1 = make_plot("Electricity hist", hist, edges, x)

measured_log = df[np.isfinite(df["total_electricity"])].log_v
hist_l, edges_l = np.histogram(measured_log, density = True, bins = 50)
x_l = np.linspace (0, 12, num=20)
p2 = make_plot("Log Electricity Hist", hist_l, edges_l, x_l)
show(gridplot([p1,p2], ncols = 2))

# They're not normal, even after logging: what do we do? GLM?

# Multilevel model
# Create local variables (clusters need to start from 0)
df.s = df.s -1
clusters = df.s
unique_clusters = clusters.unique()
heat_clusters = df.temp_h_cluster
cool_clusters = df.temp_c_cluster
unique_heat_clusters = heat_clusters.unique()
unique_cool_clusters = cool_clusters.unique()
n_hours = len(df.index)
df.t = pd.to_datetime(pd.Series(df.t))
dayhour = df['t'].dt.hour
temperature = df.outdoor_temp
outdoor_temp_c = df.outdoor_temp_c
outdoor_temp_h = df.outdoor_temp_h
coords = {"obs_id": np.arange(temperature.size)}
coords["profile_cluster"] = unique_clusters
coords["heat_cluster"] = unique_heat_clusters
coords["cool_cluster"] = unique_cool_clusters
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

with pm.Model(coords=coords) as partial_pooling:
    profile_cluster_idx = pm.Data("profile_cluster_idx", clusters, dims="obs_id")
    heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters, dims="obs_id")
    cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters, dims="obs_id")

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

    # Hyperpriors:
    bf = pm.Normal("bf", mu=0.0, sigma=1.0)
    sigma_bf = pm.Exponential("sigma_bf", 1.0)
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    btc = pm.Normal("btc", mu=0.0, sigma=1.0)
    bth = pm.Normal("bth", mu=0.0, sigma=1.0)
    sigma_btc = pm.Exponential("sigma_btc", 1.0)
    sigma_bth = pm.Exponential("sigma_bth", 1.0)

    # Varying intercepts
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="profile_cluster")

    # Varying slopes:
    bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs4 = pm.Normal("bs4", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs5 = pm.Normal("bs5", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc4 = pm.Normal("bc4", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc5 = pm.Normal("bc5", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    btc_cluster = pm.Normal("btc_cluster", mu=btc, sigma=sigma_btc, dims="cool_cluster")
    bth_cluster = pm.Normal("bth_cluster", mu=bth, sigma=sigma_bth, dims="heat_cluster")

    # Expected value per county:
    mu = a_cluster[profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[profile_cluster_idx] * fs_sin_2 + \
         bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
         bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
         bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
         bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
         btc_cluster[cool_temp_cluster_idx] * cooling_temp + bth_cluster[heat_temp_cluster_idx] * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

#Fitting without sampling
with partial_pooling:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    partial_pooling_trace = approx.sample(1000)
    partial_pooling_idata = az.from_pymc3(partial_pooling_trace)


# Intercept, Fourier, temperatures, with profile and temperature clustering -  MINIBATCH

with pm.Model(coords=coords) as partial_pooling_mb:
    profile_cluster_idx = pm.Data("profile_cluster_idx", clusters, dims="obs_id")
    heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters, dims="obs_id")
    cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters, dims="obs_id")

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

    # Minibatch replacements
    batch_size = 1000
    profile_cluster_mb = pm.Minibatch(profile_cluster_idx.get_value(), batch_size)
    heat_cluster_mb = pm.Minibatch(heat_temp_cluster_idx.get_value(), batch_size)
    cool_cluster_mb = pm.Minibatch(cool_temp_cluster_idx.get_value(), batch_size)
    fs1_mb = pm.Minibatch(fs_sin_1.get_value(), batch_size)
    fs2_mb = pm.Minibatch(fs_sin_2.get_value(), batch_size)
    fs3_mb = pm.Minibatch(fs_sin_3.get_value(), batch_size)
    fs4_mb = pm.Minibatch(fs_sin_4.get_value(), batch_size)
    fs5_mb = pm.Minibatch(fs_sin_5.get_value(), batch_size)
    fc1_mb = pm.Minibatch(fs_cos_1.get_value(), batch_size)
    fc2_mb = pm.Minibatch(fs_cos_2.get_value(), batch_size)
    fc3_mb = pm.Minibatch(fs_cos_3.get_value(), batch_size)
    fc4_mb = pm.Minibatch(fs_cos_4.get_value(), batch_size)
    fc5_mb = pm.Minibatch(fs_cos_5.get_value(), batch_size)
    cooling_temp_mb = pm.Minibatch(cooling_temp.get_value(), batch_size)
    heating_temp_mb = pm.Minibatch(heating_temp.get_value(), batch_size)
    log_electricity_mb = pm.Minibatch(log_electricity, batch_size)


    # Hyperpriors:
    bf = pm.Normal("bf", mu=0.0, sigma=1.0)
    sigma_bf = pm.Exponential("sigma_bf", 1.0)
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    btc = pm.Normal("btc", mu=0.0, sigma=1.0)
    bth = pm.Normal("bth", mu=0.0, sigma=1.0)
    sigma_btc = pm.Exponential("sigma_btc", 1.0)
    sigma_bth = pm.Exponential("sigma_bth", 1.0)

    # Varying intercepts
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="profile_cluster")

    # Varying slopes:
    bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs4 = pm.Normal("bs4", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bs5 = pm.Normal("bs5", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc4 = pm.Normal("bc4", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    bc5 = pm.Normal("bc5", mu=bf, sigma=sigma_bf, dims="profile_cluster")
    btc_cluster = pm.Normal("btc_cluster", mu=btc, sigma=sigma_btc, dims="cool_cluster")
    bth_cluster = pm.Normal("bth_cluster", mu=bth, sigma=sigma_bth, dims="heat_cluster")

    # Expected value per county:
    mu = a_cluster[profile_cluster_mb] + bs1[profile_cluster_mb] * fs1_mb + bs2[profile_cluster_mb] * fs2_mb + \
         bs3[profile_cluster_mb] * fs3_mb + bs4[profile_cluster_mb] * fs4_mb + \
         bs5[profile_cluster_mb] * fs5_mb + bc1[profile_cluster_mb] * fc1_mb + \
         bc2[profile_cluster_mb] * fc2_mb + bc3[profile_cluster_mb] * fc3_mb + \
         bc4[profile_cluster_mb] * fc4_mb + bc5[profile_cluster_mb] * fc5_mb + \
         btc_cluster[cool_cluster_mb] * cooling_temp_mb + bth_cluster[heat_cluster_mb] * heating_temp_mb

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_mb, total_size = n_hours)


# Graphviz visualisation
varying_intercept_and_temp_graph = pm.model_to_graphviz(partial_pooling)
varying_intercept_and_temp_graph.render(filename='img/varying_intercept_and_temp_hourly')

#Fitting without sampling
with partial_pooling_mb:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    partial_pooling_trace = approx.sample(1000)
    partial_pooling_idata = az.from_pymc3(partial_pooling_trace)

az.summary(partial_pooling_idata, round_to=2)
az.plot_trace(partial_pooling_idata)
plt.show()

# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

partial_pooling_acluster_means = np.mean(partial_pooling_trace['a_cluster'], axis =0)
partial_pooling_bs1_means = np.mean(partial_pooling_trace['bs1'], axis =0)
partial_pooling_bs2_means = np.mean(partial_pooling_trace['bs2'], axis =0)
partial_pooling_bs3_means = np.mean(partial_pooling_trace['bs3'], axis =0)
partial_pooling_bs4_means = np.mean(partial_pooling_trace['bs4'], axis =0)
partial_pooling_bs5_means = np.mean(partial_pooling_trace['bs5'], axis =0)
partial_pooling_bc1_means = np.mean(partial_pooling_trace['bc1'], axis =0)
partial_pooling_bc2_means = np.mean(partial_pooling_trace['bc2'], axis =0)
partial_pooling_bc3_means = np.mean(partial_pooling_trace['bc3'], axis =0)
partial_pooling_bc4_means = np.mean(partial_pooling_trace['bc4'], axis =0)
partial_pooling_bc5_means = np.mean(partial_pooling_trace['bc5'], axis =0)

partial_pooling_bth_means = np.mean(partial_pooling_trace['btc_cluster'], axis = 0)
partial_pooling_btc_means = np.mean(partial_pooling_trace['btc_cluster'], axis = 0)
# Create array with predictions
partial_pooling_predictions = []
# Create array with bounds
partial_pooling_hdi = az.hdi(partial_pooling_idata)
partial_pooling_mean_lower = []
partial_pooling_mean_higher= []
partial_pooling_lower = []
partial_pooling_higher= []

for hour, row in df.iterrows():
    for cluster_idx in unique_clusters:
        if clusters[hour] == cluster_idx:
            for heat_cluster_idx in unique_heat_clusters:
                if heat_clusters[hour] == heat_cluster_idx:
                    for cool_cluster_idx in unique_cool_clusters:
                        if cool_clusters[hour] == cool_cluster_idx:

                            partial_pooling_predictions.append(partial_pooling_acluster_means[cluster_idx] + \
                                                               partial_pooling_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                                               partial_pooling_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                                               partial_pooling_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                                               partial_pooling_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                                               partial_pooling_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                                               partial_pooling_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                                               partial_pooling_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                                               partial_pooling_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                                               partial_pooling_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                                               partial_pooling_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                                               partial_pooling_bth_means[heat_cluster_idx] * outdoor_temp_h[hour] + \
                                                               partial_pooling_btc_means[cool_cluster_idx] * outdoor_temp_c[hour])

# Calculate prediction error
predictions = np.exp(partial_pooling_predictions)
mse = mean_squared_error(df.total_electricity, predictions)
rmse = sqrt(mse)
cvrmse = rmse/df.total_electricity.mean()

# PLOTS

# output to static HTML file
output_file("predictions.html")

# predictions vs real log scale
p = figure(plot_width=800, plot_height=400)

p.circle(df.index, partial_pooling_predictions, size=5, color="navy", alpha=0.5)
p.circle(df.index, log_electricity, size=5, color="orange", alpha=0.2)
show(p)

# predictions vs real

p2 = figure(plot_width=800, plot_height=400)

p2.circle(df.index, predictions, size=5, color="navy", alpha=0.5)
p2.circle(df.index, df.total_electricity, size = 5, color="orange", alpha=0.2)
show(p2)

# Temperature varying predictions vs real log scale and normal scale
p3 = figure(plot_width=800, plot_height=400)

p3.circle(df.outdoor_temp, partial_pooling_predictions, size=5, color="navy", alpha=0.5)
p3.circle(df.outdoor_temp, log_electricity, size=5, color="orange", alpha=0.2)
# show the results
show(p3)

p4 = figure(plot_width=800, plot_height=400)

p4.circle(df.outdoor_temp, predictions, size=5, color="navy", alpha=0.5)
p4.circle(df.outdoor_temp, df.total_electricity, size=5, color="orange", alpha=0.2)
# show the results
show(p4)

#Plot temperatures

p5 = figure(plot_width=800, plot_height=400)

p5.circle(df.index, df.outdoor_temp_h, size=5, color="red", alpha=0.5)
p5.circle(df.index, df.outdoor_temp_c, size=5, color="navy", alpha=0.2)
show(p5)

# Plot consumption

p6 = figure(plot_width=800, plot_height=400)
p6.line(df.index, df.total_electricity, color="orange")
show(p6)

