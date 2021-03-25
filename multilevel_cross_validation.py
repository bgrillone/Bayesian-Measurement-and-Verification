import arviz  as az
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
from pymc3.variational.callbacks import CheckParametersConvergence

RANDOM_SEED = 8924

# Data import
df = pd.read_csv("~/Github/Bayes-M&V/data/Id50_preprocessed2.csv", index_col=0)

# Plotting data hist
df["log_v"] = log_electricity = np.log(df["total_electricity"]).values
total_electricity = df.total_electricity.values

measured = df[np.isfinite(df["total_electricity"])].total_electricity
hist, edges = np.histogram(measured, density=True, bins=50)


def make_plot(title, hist, edges, x):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)

    p.y_range.start = 0
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color = "white"
    return p


x = np.linspace(0, 30000, num=3000)
p1 = make_plot("Electricity hist", hist, edges, x)

measured_log = df[np.isfinite(df["total_electricity"])].log_v
hist_l, edges_l = np.histogram(measured_log, density=True, bins=50)
x_l = np.linspace(0, 12, num=20)
p2 = make_plot("Log Electricity Hist", hist_l, edges_l, x_l)
show(gridplot([p1, p2], ncols=2))

# Create local variables (assign daypart, cluster values need to start from 0)
# clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
# temperature dependence (likely to modify this in the new version of the preprocessing)

df.t = pd.to_datetime(pd.Series(df.t))
df["daypart"] = np.where(df['t'].dt.hour <= 19,
                         np.where(df['t'].dt.hour <= 15,
                                  np.where(df['t'].dt.hour <= 11,
                                           np.where(df['t'].dt.hour <= 7,
                                                    np.where(df['t'].dt.hour <= 3, 0, 1), 2), 3), 4), 5)
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

# create coords for pymc3
coords = {"obs_id": np.arange(temperature.size)}
coords["profile_cluster"] = unique_clusters
coords["heat_cluster"] = unique_heat_clusters
coords["cool_cluster"] = unique_cool_clusters
coords["daypart"] = unique_dayparts

# Create kfold cross-validation splits

kf = KFold(n_splits = 10)
kf.get_n_splits(df)

# Create array to save cross validation accuracy
partial_pooling_cv_accuracy = []
coverage_list = []

# Bayesian linear model with Intercept, Fourier series for the seasonal features,
# temperatures, pooled on profile and temperature clustering

for train_index, test_index in kf.split(df):

    coords = {"obs_id": np.arange(temperature[train_index].size)}
    coords["profile_cluster"] = unique_clusters
    coords["heat_cluster"] = unique_heat_clusters
    coords["cool_cluster"] = unique_cool_clusters
    coords["daypart"] = unique_dayparts

    with pm.Model(coords=coords) as partial_pooling:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters[train_index], dims="obs_id")
        heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters[train_index], dims="obs_id")
        cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters[train_index], dims="obs_id")
        daypart = pm.Data("daypart", dayparts[train_index], dims="obs_id")

        fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1[train_index], dims="obs_id")
        fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2[train_index], dims="obs_id")
        fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3[train_index], dims="obs_id")
        fs_sin_4 = pm.Data("fs_sin_4", daypart_fs_sin_4[train_index], dims="obs_id")
        fs_sin_5 = pm.Data("fs_sin_5", daypart_fs_sin_5[train_index], dims="obs_id")
        fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1[train_index], dims="obs_id")
        fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2[train_index], dims="obs_id")
        fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3[train_index], dims="obs_id")
        fs_cos_4 = pm.Data("fs_cos_4", daypart_fs_cos_4[train_index], dims="obs_id")
        fs_cos_5 = pm.Data("fs_cos_5", daypart_fs_cos_5[train_index], dims="obs_id")

        cooling_temp = pm.Data("cooling_temp", outdoor_temp_c[train_index], dims="obs_id")
        heating_temp = pm.Data("heating_temp", outdoor_temp_h[train_index], dims="obs_id")

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
        mu = a_cluster[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[
            profile_cluster_idx] * fs_sin_2 + \
             bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
             bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
             bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
             bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
             btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims='obs_id')

    # Fitting without sampling
    with partial_pooling:
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        partial_pooling_trace = approx.sample(1000)
        partial_pooling_idata = az.from_pymc3(partial_pooling_trace)

    # Sampling from the posterior setting test data to check the predictions on unseen data

    with partial_pooling:

        pm.set_data({"profile_cluster_idx": clusters[test_index], "heat_temp_cluster_idx": heat_clusters[test_index],
                     "cool_temp_cluster_idx": cool_clusters[test_index], "daypart": dayparts[test_index], "fs_sin_1": daypart_fs_sin_1[test_index],
                     "fs_sin_2": daypart_fs_sin_2[test_index], "fs_sin_3": daypart_fs_sin_3[test_index], "fs_sin_4": daypart_fs_sin_4[test_index],
                     "fs_sin_5": daypart_fs_sin_5[test_index], "fs_cos_1": daypart_fs_cos_1[test_index], "fs_cos_2": daypart_fs_cos_2[test_index],
                     "fs_cos_3": daypart_fs_cos_3[test_index], "fs_cos_4": daypart_fs_cos_4[test_index], "fs_cos_5": daypart_fs_cos_5[test_index],
                     "cooling_temp": outdoor_temp_c[test_index], "heating_temp": outdoor_temp_h[test_index]})

        posterior_hdi = pm.sample_posterior_predictive(partial_pooling_trace, keep_size=True)
        posterior = pm.sample_posterior_predictive(partial_pooling_trace)


    # Calculate predictions and HDI

    predictions = np.exp(posterior['y'].mean(0))
    hdi_data = az.hdi(posterior_hdi)
    lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate cvrmse and coverage of the HDI
    mse = mean_squared_error(df.total_electricity[test_index], predictions)
    rmse = sqrt(mse)
    cvrmse = rmse / df.total_electricity.mean()
    coverage = sum((lower_bound <= df.total_electricity[test_index]) & (df.total_electricity[test_index] <= higher_bound)) * 100 / len(test_index)

    partial_pooling_cv_accuracy.append (cvrmse)
    coverage_list.append(coverage)
    avg_cvrmse = np.mean(partial_pooling_cv_accuracy)
    avg_coverage = np.mean(coverage_list)