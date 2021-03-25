import arviz  as az
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show, save
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.metrics import mean_squared_error
from math import sqrt

RANDOM_SEED = 8924

# Data import
df = pd.read_csv("~/Github/Bayes-M&V/data/Id50_preprocessed2.csv", index_col = 0)

# Plotting data hist
df["log_v"] = log_electricity = np.log(df["total_electricity"]).values
total_electricity = df.total_electricity.values

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


# Create local variables (assign daypart, cluster values need to start from 0)
# clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
# temperature dependence (likely to modify this in the new version of the preprocessing)

df.t = pd.to_datetime(pd.Series(df.t))
df["daypart"] = np.where(df['t'].dt.hour <= 19,
                         np.where(df['t'].dt.hour <= 15,
                                  np.where(df['t'].dt.hour <= 11,
                                           np.where(df['t'].dt.hour <= 7,
                                                    np.where(df['t'].dt.hour <= 3,0,1),2),3),4),5)
df.s = df.s -1
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

# create coords for pymc3
coords = {"obs_id": np.arange(temperature.size)}
coords["profile_cluster"] = unique_clusters
coords["heat_cluster"] = unique_heat_clusters
coords["cool_cluster"] = unique_cool_clusters
coords["daypart"] = unique_dayparts

# Bayesian linear model with Intercept, Fourier series for the seasonal features,
# temperatures, pooled on profile and temperature clustering

with pm.Model(coords=coords) as partial_pooling:
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

    # Hyperpriors:
    bf = pm.Normal("bf", mu=0.0, sigma=1.0)
    sigma_bf = pm.Exponential("sigma_bf", 1.0)
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    btc = pm.Normal("btc", mu=0.0, sigma=1.0, dims=("daypart", "cool_cluster"))
    bth = pm.Normal("bth", mu=0.0, sigma=1.0, dims=("daypart", "heat_cluster"))

    # Varying intercepts
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("daypart", "profile_cluster"))

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

    # Expected value per county:
    mu = a_cluster[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[profile_cluster_idx] * fs_sin_2 + \
         bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
         bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
         bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
         bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
         btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    #Likelihood
    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

#Fitting
with partial_pooling:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    partial_pooling_trace = approx.sample(1000)
    partial_pooling_idata = az.from_pymc3(partial_pooling_trace)

# Sampling from the posterior

with partial_pooling:

    pm.set_data({"profile_cluster_idx" : clusters, "heat_temp_cluster_idx" : heat_clusters, "cool_temp_cluster_idx": cool_clusters,
                 "daypart":dayparts, "fs_sin_1" : daypart_fs_sin_1, "fs_sin_2" : daypart_fs_sin_2, "fs_sin_3" : daypart_fs_sin_3,
                 "fs_sin_4" : daypart_fs_sin_4, "fs_sin_5" : daypart_fs_sin_5, "fs_cos_1" : daypart_fs_cos_1, "fs_cos_2" : daypart_fs_cos_2,
                 "fs_cos_3" : daypart_fs_cos_3, "fs_cos_4" : daypart_fs_cos_4, "fs_cos_5" : daypart_fs_cos_5, "cooling_temp":outdoor_temp_c,
                 "heating_temp": outdoor_temp_h})

    posterior_hdi = pm.sample_posterior_predictive(partial_pooling_trace, keep_size=True)
    posterior = pm.sample_posterior_predictive(partial_pooling_trace)

    prior = pm.sample_prior_predictive(150)

    partial_pooling_idata = az.from_pymc3(partial_pooling_trace, prior = prior, posterior_predictive = posterior)

# Calculate predictions and HDI

predictions = np.exp(posterior['y'].mean(0))
hdi_data = az.hdi(posterior_hdi)
lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi = 'lower'))).flatten()
higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi = 'higher'))).flatten()

# Calculate cvrmse and coverage of the HDI
mse = mean_squared_error(df.total_electricity, predictions)
rmse = sqrt(mse)
cvrmse = rmse/df.total_electricity.mean()
coverage = sum((lower_bound <= df.total_electricity) & (df.total_electricity <= higher_bound)) * 100 / len(df)

# PLOTS

# output to static HTML file
output_file("predictions.html")

# Plot real consumption, predicted consumption, HDI
p1 = figure(plot_width=800, plot_height=400,  x_axis_type = 'datetime')
p1.line(df.t, predictions, color="navy", alpha=0.8)
p1.line(df.t, df.total_electricity, color="orange", alpha=0.6)
p1.varea(df.t, y1 = lower_bound, y2 = higher_bound, color = 'gray', alpha = 0.2)
show(p1)

# Temperature varying predictions vs real and normal scale

p2 = figure(plot_width=800, plot_height=400)

p2.circle(df.outdoor_temp, predictions, size=5, color="navy", alpha=0.5)
p2.circle(df.outdoor_temp, df.total_electricity, size=5, color="orange", alpha=0.2)
show(p2)


