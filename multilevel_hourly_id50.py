import arviz  as az
import matplotlib.pyplot as plt
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings

RANDOM_SEED = 8924

# Data import
df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/Id50_preprocessed.csv", index_col = 0)

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
n_hours = len(df.index)
df.t = pd.to_datetime(pd.Series(df.t))
dayhour = df['t'].dt.hour
temperature = df.outdoor_temp
outdoor_temp_c = df.outdoor_temp_c
outdoor_temp_h = df.outdoor_temp_h
coords = {"obs_id": np.arange(temperature.size)}
coords["Cluster"] = unique_clusters
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


#Might want to try temperature clustering

with pm.Model(coords=coords) as partial_pooling:
    cluster_idx = pm.Data("cluster_idx", clusters, dims="obs_id")
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

    # Fixed intercepts
    btc = pm.Normal("btc", mu=0.0, sigma=1.0)
    bth = pm.Normal("bth", mu=0.0, sigma=1.0)
    # Hyperpriors:
    bf = pm.Normal("bf", mu=0.0, sigma=1.0)
    sigma_bf = pm.Exponential("sigma_bf", 1.0)

    # Varying intercepts:
    bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs4 = pm.Normal("bs4", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs5 = pm.Normal("bs5", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc4 = pm.Normal("bc4", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc5 = pm.Normal("bc5", mu=bf, sigma=sigma_bf, dims="Cluster")

    # Expected value per county:
    mu = bs1[cluster_idx] * fs_sin_1 + bs2[cluster_idx] * fs_sin_2 + bs3[cluster_idx] * fs_sin_3 + \
         bs4[cluster_idx] * fs_sin_4 + bs5[cluster_idx] * fs_sin_5 + bc1[cluster_idx] * fs_cos_1 + \
         bc2[cluster_idx] * fs_cos_2 + bc3[cluster_idx] * fs_cos_3 + bc4[cluster_idx] * fs_cos_4 + \
         bc5[cluster_idx] * fs_cos_5 + btc * cooling_temp + bth * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")


# Need to install graphviz
varying_intercept_and_temp_graph = pm.model_to_graphviz(partial_pooling)
varying_intercept_and_temp_graph.render(filename='img/varying_intercept_and_temp_hourly')


with partial_pooling:
    partial_pooling_trace = pm.sample(random_seed=RANDOM_SEED, init = 'adapt_diag')
    partial_pooling_idata = az.from_pymc3(partial_pooling_trace)

az.summary(partial_pooling_idata, round_to=2)
az.plot_trace(partial_pooling_idata)
plt.show()

# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

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

partial_pooling_bth_means = np.mean(partial_pooling_trace['bth'], axis = 0)
partial_pooling_btc_means = np.mean(partial_pooling_trace['btc'], axis = 0)
# Create array with predictions
partial_pooling_predictions = []
# Create array with bounds
# varying_intercept_slope_hdi = az.hdi(varying_temp_idata)
# varying_intercept_slope_mean_lower = []
# varying_intercept_slope_mean_higher= []
# varying_intercept_slope_lower = []
# varying_intercept_slope_higher= []

for hour, row in df.iterrows():
    for cluster_idx in unique_clusters:
        if clusters[hour] == cluster_idx:
            partial_pooling_predictions.append(partial_pooling_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                               partial_pooling_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                               partial_pooling_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                               partial_pooling_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                               partial_pooling_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                               partial_pooling_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                               partial_pooling_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                               partial_pooling_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                               partial_pooling_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                               partial_pooling_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                               partial_pooling_bth_means * outdoor_temp_h[hour] + \
                                               partial_pooling_btc_means * outdoor_temp_c[hour])

# output to static HTML file
output_file("predictions.html")

p = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha
p.circle(df.index, partial_pooling_predictions, size=5, color="navy", alpha=0.5)
p.circle(df.index, log_electricity, size=5, color="orange", alpha=0.5)
# show the results
show(p)

p2 = figure(plot_width=800, plot_height=400)

p2.circle(df.index, log_electricity, size=5, color="orange", alpha=0.5)
show(p2)

#Cons vs temp
p3 = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha
p3.circle(df.outdoor_temp, partial_pooling_predictions, size=5, color="navy", alpha=0.5)
p3.circle(df.outdoor_temp, log_electricity, size=5, color="orange", alpha=0.5)
# show the results
show(p3)

actual_predictions = np.exp(partial_pooling_predictions)

p4 = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha
p4.circle(df.index, actual_predictions, size=5, color="navy", alpha=0.5)
p4.circle(df.index, df.total_electricity, size=5, color="orange", alpha=0.5)
p4.y_range.end = 30000
# show t
show(p4)

#Plot temperatures

p5 = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha

p5.circle(df.index, df.outdoor_temp_h, size=5, color="red", alpha=0.5)
p5.circle(df.index, df.outdoor_temp_c, size=5, color="navy", alpha=0.5)
show(p5)

p6 = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha

p6.circle(df.index, df.outdoor_temp, size=5, color="red", alpha=0.5)
# show t
show(p6)

#Cons vs temp
p7 = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha
p7.circle(df.outdoor_temp, df.total_electricity, size=5, color="navy", alpha=0.5)
# show the results
show(p7)

# Try a model without temperature term to understand if it's the temperature that it's screwing the model

with pm.Model(coords=coords) as partial_pooling_notemp:
    cluster_idx = pm.Data("cluster_idx", clusters, dims="obs_id")
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

    # Hyperpriors:
    bf = pm.Normal("bf", mu=0.0, sigma=1.0)
    sigma_bf = pm.Exponential("sigma_bf", 1.0)

    # Varying intercepts:
    bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs4 = pm.Normal("bs4", mu=bf, sigma=sigma_bf, dims="Cluster")
    bs5 = pm.Normal("bs5", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc4 = pm.Normal("bc4", mu=bf, sigma=sigma_bf, dims="Cluster")
    bc5 = pm.Normal("bc5", mu=bf, sigma=sigma_bf, dims="Cluster")

    # Expected value per county:
    mu = bs1[cluster_idx] * fs_sin_1 + bs2[cluster_idx] * fs_sin_2 + bs3[cluster_idx] * fs_sin_3 + \
         bs4[cluster_idx] * fs_sin_4 + bs5[cluster_idx] * fs_sin_5 + bc1[cluster_idx] * fs_cos_1 + \
         bc2[cluster_idx] * fs_cos_2 + bc3[cluster_idx] * fs_cos_3 + bc4[cluster_idx] * fs_cos_4 + \
         bc5[cluster_idx] * fs_cos_5

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")


with partial_pooling_notemp:
    partial_pooling_notemp_trace = pm.sample(random_seed=RANDOM_SEED, target_accept = 0.99)
    partial_pooling_notemp_idata = az.from_pymc3(partial_pooling_notemp_trace)

az.summary(partial_pooling_notemp_idata, round_to=2)
az.plot_trace(partial_pooling_notemp_idata)
plt.show()

partial_pooling_notemp_bs1_means = np.mean(partial_pooling_notemp_trace['bs1'], axis =0)
partial_pooling_notemp_bs2_means = np.mean(partial_pooling_notemp_trace['bs2'], axis =0)
partial_pooling_notemp_bs3_means = np.mean(partial_pooling_notemp_trace['bs3'], axis =0)
partial_pooling_notemp_bs4_means = np.mean(partial_pooling_notemp_trace['bs4'], axis =0)
partial_pooling_notemp_bs5_means = np.mean(partial_pooling_notemp_trace['bs5'], axis =0)
partial_pooling_notemp_bc1_means = np.mean(partial_pooling_notemp_trace['bc1'], axis =0)
partial_pooling_notemp_bc2_means = np.mean(partial_pooling_notemp_trace['bc2'], axis =0)
partial_pooling_notemp_bc3_means = np.mean(partial_pooling_notemp_trace['bc3'], axis =0)
partial_pooling_notemp_bc4_means = np.mean(partial_pooling_notemp_trace['bc4'], axis =0)
partial_pooling_notemp_bc5_means = np.mean(partial_pooling_notemp_trace['bc5'], axis =0)
# Create array with predictions
partial_pooling_notemp_predictions = []

for hour, row in df.iterrows():
    for cluster_idx in unique_clusters:
        if clusters[hour] == cluster_idx:
            partial_pooling_notemp_predictions.append(partial_pooling_notemp_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                                      partial_pooling_notemp_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                                      partial_pooling_notemp_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                                      partial_pooling_notemp_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                                      partial_pooling_notemp_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                                      partial_pooling_notemp_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                                      partial_pooling_notemp_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                                      partial_pooling_notemp_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                                      partial_pooling_notemp_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                                      partial_pooling_notemp_bc5_means[cluster_idx] * daypart_fs_cos_5[hour])



p = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha
p.circle(df.index, partial_pooling_notemp_predictions, size=5, color="navy", alpha=0.5)
p.circle(df.index, log_electricity, size=5, color="orange", alpha=0.5)
# show the results
show(p)


# HOUR OF THE DAY + TEMPERATURE SLOPE ( NO FOURIER )


with pm.Model(coords=coords) as partial_pooling_hour:
    cluster_idx = pm.Data("cluster_idx", clusters, dims="obs_id")
    hour = pm.Data("hour", dayhour, dims = "obs_id")

    cooling_temp = pm.Data("cooling_temp", outdoor_temp_c, dims="obs_id")
    heating_temp = pm.Data("heating_temp", outdoor_temp_h, dims="obs_id")

    # Fixed intercepts
    btc = pm.Normal("btc", mu=0.0, sigma=1.0)
    bth = pm.Normal("bth", mu=0.0, sigma=1.0)
    # Hyperpriors:
    bh_mean = pm.Normal("bh_mean", mu=0.0, sigma=1.0)
    sigma_bh = pm.Exponential("sigma_bh", 1.0)

    # Varying intercepts:
    bh = pm.Normal("bh", mu=bh_mean, sigma=sigma_bh, dims="Cluster")

    # Expected value per county:
    mu = bh[cluster_idx] * hour + btc * cooling_temp + bth * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

with partial_pooling_hour:
    partial_pooling_hour_trace = pm.sample(random_seed=RANDOM_SEED, init = 'adapt_diag',
                                           target_accept = 0.99)
    partial_pooling_hour_idata = az.from_pymc3(partial_pooling_hour_trace)


# Calculate predictions
partial_pooling_hour_bh_means = np.mean(partial_pooling_hour_trace['bh'], axis =0)
partial_pooling_hour_bth_means = np.mean(partial_pooling_hour_trace['bth'], axis = 0)
partial_pooling_hour_btc_means = np.mean(partial_pooling_hour_trace['btc'], axis = 0)
# Create array with predictions
partial_pooling_hour_predictions = []
# Create array with bounds
# varying_intercept_slope_hdi = az.hdi(varying_temp_idata)
# varying_intercept_slope_mean_lower = []
# varying_intercept_slope_mean_higher= []
# varying_intercept_slope_lower = []
# varying_intercept_slope_higher= []

for hour, row in df.iterrows():
    for cluster_idx in unique_clusters:
        if clusters[hour] == cluster_idx:
            partial_pooling_hour_predictions.append(partial_pooling_hour_bh_means[cluster_idx] * dayhour[hour]+ \
                                                    partial_pooling_hour_bth_means * outdoor_temp_h[hour] + \
                                                    partial_pooling_hour_btc_means * outdoor_temp_c[hour])

p = figure(plot_width=800, plot_height=400)

# add a circle renderer with a size, color, and alpha
p.circle(df.index, partial_pooling_hour_predictions, size=5, color="navy", alpha=0.5)
p.circle(df.index, log_electricity, size=5, color="orange", alpha=0.5)
# show the results
show(p)