import arviz  as az
import matplotlib.pyplot as plt
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
import xarray as xr
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
# Optimize with 5 fold CV which variables should be pooled on and which not
# Run different models and save the accuracy to a DF, finally write the DF in a csv.
# Also save graphs for each model in the folder
# Compare cross-validation CV(RMSE) and WAIC

# ---- DATA IMPORT AND PREPROCESSING
RANDOM_SEED = 8924

# Data import
df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/Id50_preprocessed2.csv", index_col=0)

# Preprocessing
df["log_v"] = log_electricity = np.log(df["total_electricity"]).values
total_electricity = df.total_electricity.values


# Create local variables (clusters need to start from 0)

df.t = pd.to_datetime(pd.Series(df.t))
df["daypart"] = np.where(df['t'].dt.hour <= 19,
                         np.where(df['t'].dt.hour <= 15,
                                  np.where(df['t'].dt.hour <= 11,
                                           np.where(df['t'].dt.hour <= 7,
                                                    np.where(df['t'].dt.hour <= 3,0,1),2),3),4),5)
dayhours = df['t'].dt.hour
df.s = df.s -1
clusters = df.s
unique_clusters = clusters.unique()
heat_clusters = df.temp_h_cluster
cool_clusters = df.temp_c_cluster
dayparts = df.daypart
unique_heat_clusters = heat_clusters.unique()
unique_cool_clusters = cool_clusters.unique()
unique_dayparts = dayparts.unique()
unique_dayhour = dayhours.unique()
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


# CROSS VALIDATION

# Create kfold cross-validation splits

kf = KFold(n_splits = 5)
kf.get_n_splits(df)

# Create arrays to save model results
model_1_cv_accuracy = []
model_2_cv_accuracy = []
model_3_cv_accuracy = []
model_4_cv_accuracy = []
model_5_cv_accuracy = []
coverage_1_list = []
coverage_2_list = []
coverage_3_list = []
coverage_4_list = []
coverage_5_list = []
cvrmse = []
waic = []
model_name = []
coverage = []

for train_index, test_index in kf.split(df):

    # Coords
    coords = {"obs_id": np.arange(temperature[train_index].size)}
    coords["profile_cluster"] = unique_clusters
    coords["heat_cluster"] = unique_heat_clusters
    coords["cool_cluster"] = unique_cool_clusters
    coords["daypart"] = unique_dayparts
    coords["dayhour"] = unique_dayhour

    # MODEL 1) No pooling

    with pm.Model(coords=coords) as model_1:
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

        # Priors:
        a = pm.Normal("a", mu=0.0, sigma=1.0, dims=("daypart", "profile_cluster"))
        btc = pm.Normal("btc", mu=0.0, sigma=1.0, dims=("daypart", "cool_cluster"))
        bth = pm.Normal("bth", mu=0.0, sigma=1.0, dims=("daypart", "heat_cluster"))

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
        mu = a[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + \
             bs2[profile_cluster_idx] * fs_sin_2 + bs3[profile_cluster_idx] * fs_sin_3 + \
             bs4[profile_cluster_idx] * fs_sin_4 + bs5[profile_cluster_idx] * fs_sin_5 + \
             bc1[profile_cluster_idx] * fs_cos_1 + bc2[profile_cluster_idx] * fs_cos_2 + \
             bc3[profile_cluster_idx] * fs_cos_3 + bc4[profile_cluster_idx] * fs_cos_4 + \
             bc5[profile_cluster_idx] * fs_cos_5 + btc[daypart, cool_temp_cluster_idx] * cooling_temp + \
             bth[daypart, heat_temp_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims='obs_id')

    # advi fitting
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_1_trace = approx.sample(1000)
        model_1_idata = az.from_pymc3(model_1_trace)
        az.plot_trace(model_1_idata)
        plt.save('results/plots/model_1_trace.png')

    model_1_a_means = np.mean(model_1_trace['a'], axis=0)
    model_1_bs1_means = np.mean(model_1_trace['bs1'], axis=0)
    model_1_bs2_means = np.mean(model_1_trace['bs2'], axis=0)
    model_1_bs3_means = np.mean(model_1_trace['bs3'], axis=0)
    model_1_bs4_means = np.mean(model_1_trace['bs4'], axis=0)
    model_1_bs5_means = np.mean(model_1_trace['bs5'], axis=0)
    model_1_bc1_means = np.mean(model_1_trace['bc1'], axis=0)
    model_1_bc2_means = np.mean(model_1_trace['bc2'], axis=0)
    model_1_bc3_means = np.mean(model_1_trace['bc3'], axis=0)
    model_1_bc4_means = np.mean(model_1_trace['bc4'], axis=0)
    model_1_bc5_means = np.mean(model_1_trace['bc5'], axis=0)

    model_1_bth_means = np.mean(model_1_trace['bth'], axis=0)
    model_1_btc_means = np.mean(model_1_trace['btc'], axis=0)
    # Create array with predictions
    model_1_log_predictions = []
    # Create array with bounds
    model_1_hdi = az.hdi(model_1_idata)
    model_1_lower_log = []
    model_1_higher_log = []

    for hour, row in df.loc[test_index].iterrows():
        for cluster_idx in unique_clusters:
            if clusters[hour] == cluster_idx:
                for heat_cluster_idx in unique_heat_clusters:
                    if heat_clusters[hour] == heat_cluster_idx:
                        for cool_cluster_idx in unique_cool_clusters:
                            if cool_clusters[hour] == cool_cluster_idx:
                                for daypart_idx in unique_dayparts:
                                    if dayparts[hour] == daypart_idx:
                                        model_1_log_predictions.append(model_1_a_means[daypart_idx, cluster_idx] + \
                                                                      model_1_bs1_means[cluster_idx] *
                                                                      daypart_fs_sin_1[hour] + \
                                                                      model_1_bs2_means[cluster_idx] *
                                                                      daypart_fs_sin_2[hour] + \
                                                                      model_1_bs3_means[cluster_idx] *
                                                                      daypart_fs_sin_3[hour] + \
                                                                      model_1_bs4_means[cluster_idx] *
                                                                      daypart_fs_sin_4[hour] + \
                                                                      model_1_bs5_means[cluster_idx] *
                                                                      daypart_fs_sin_5[hour] + \
                                                                      model_1_bc1_means[cluster_idx] *
                                                                      daypart_fs_cos_1[hour] + \
                                                                      model_1_bc2_means[cluster_idx] *
                                                                      daypart_fs_cos_2[hour] + \
                                                                      model_1_bc3_means[cluster_idx] *
                                                                      daypart_fs_cos_3[hour] + \
                                                                      model_1_bc4_means[cluster_idx] *
                                                                      daypart_fs_cos_4[hour] + \
                                                                      model_1_bc5_means[cluster_idx] *
                                                                      daypart_fs_cos_5[hour] + \
                                                                      model_1_bth_means[
                                                                          daypart_idx, heat_cluster_idx] *
                                                                      outdoor_temp_h[hour] + \
                                                                      model_1_btc_means[
                                                                          daypart_idx, cool_cluster_idx] *
                                                                      outdoor_temp_c[hour])

                                        model_1_lower_log.append(
                                            model_1_hdi['a'][daypart_idx, cluster_idx].sel(
                                                hdi='lower').values + \
                                            model_1_hdi['bs1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_1_hdi['bs2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_1_hdi['bs3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_1_hdi['bs4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_1_hdi['bs5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_1_hdi['bc1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_1_hdi['bc2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_1_hdi['bc3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_1_hdi['bc4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_1_hdi['bc5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_1_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_h[hour] + \
                                            model_1_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_c[hour] - \
                                            model_1_hdi['sigma'].sel(hdi='higher').values)

                                        model_1_higher_log.append(
                                            model_1_hdi['a'][daypart_idx, cluster_idx].sel(
                                                hdi='higher').values + \
                                            model_1_hdi['bs1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_1_hdi['bs2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_1_hdi['bs3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_1_hdi['bs4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_1_hdi['bs5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_1_hdi['bc1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_1_hdi['bc2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_1_hdi['bc3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_1_hdi['bc4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_1_hdi['bc5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_1_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_h[hour] + \
                                            model_1_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_c[hour] + \
                                            model_1_hdi['sigma'].sel(hdi='higher').values)

    # Calculate prediction error
    model_1_predictions = np.exp(model_1_log_predictions)
    model_1_mse = mean_squared_error(df.loc[test_index].total_electricity, model_1_predictions)
    model_1_rmse = sqrt(model_1_mse)
    model_1_cvrmse = model_1_rmse / df.total_electricity.mean()
    model_1_cv_accuracy.append(model_1_cvrmse)
    print('Model 1 cv cv(rmse): ', model_1_cv_accuracy, '%')

    # Coverage calculations
    model_1_higher = np.exp(model_1_higher_log)
    model_1_lower = np.exp(model_1_lower_log)
    coverage_1 = np.where(df.total_electricity[test_index] <= model_1_higher, np.where(model_1_predictions >= model_1_lower, 1, 0), 0)
    coverage_perc_1 = sum(coverage_1) * 100 / len(coverage_1)
    coverage_1_list.append(coverage_perc_1)
    print('Model 1 coverage: ', coverage_1_list, '%')


    # ---- MODEL 2) Intercept pooled on daypart and profile; Fourier pooled on profile; Temperature unpooled on daypart and cluster

    with pm.Model(coords=coords) as model_2:
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

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_2_trace = approx.sample(1000)
        model_2_idata = az.from_pymc3(model_2_trace)
        az.plot_trace(model_2_idata)
        plt.save('results/plots/model_2_trace.png')

    # Calculate predictions
    model_2_acluster_means = np.mean(model_2_trace['a_cluster'], axis=0)
    model_2_bs1_means = np.mean(model_2_trace['bs1'], axis=0)
    model_2_bs2_means = np.mean(model_2_trace['bs2'], axis=0)
    model_2_bs3_means = np.mean(model_2_trace['bs3'], axis=0)
    model_2_bs4_means = np.mean(model_2_trace['bs4'], axis=0)
    model_2_bs5_means = np.mean(model_2_trace['bs5'], axis=0)
    model_2_bc1_means = np.mean(model_2_trace['bc1'], axis=0)
    model_2_bc2_means = np.mean(model_2_trace['bc2'], axis=0)
    model_2_bc3_means = np.mean(model_2_trace['bc3'], axis=0)
    model_2_bc4_means = np.mean(model_2_trace['bc4'], axis=0)
    model_2_bc5_means = np.mean(model_2_trace['bc5'], axis=0)

    model_2_bth_means = np.mean(model_2_trace['bth'], axis=0)
    model_2_btc_means = np.mean(model_2_trace['btc'], axis=0)
    # Create array with predictions
    model_2_log_predictions = []
    # Create array with bounds
    model_2_hdi = az.hdi(model_2_idata)
    model_2_lower_log = []
    model_2_higher_log = []

    for hour, row in df.loc[test_index].iterrows():
        for cluster_idx in unique_clusters:
            if clusters[hour] == cluster_idx:
                for heat_cluster_idx in unique_heat_clusters:
                    if heat_clusters[hour] == heat_cluster_idx:
                        for cool_cluster_idx in unique_cool_clusters:
                            if cool_clusters[hour] == cool_cluster_idx:
                                for daypart_idx in unique_dayparts:
                                    if dayparts[hour] == daypart_idx:
                                        model_2_log_predictions.append(
                                            model_2_acluster_means[daypart_idx, cluster_idx] + \
                                            model_2_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                            model_2_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                            model_2_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                            model_2_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                            model_2_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                            model_2_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                            model_2_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                            model_2_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                            model_2_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                            model_2_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                            model_2_bth_means[daypart_idx, heat_cluster_idx] * outdoor_temp_h[
                                                hour] + \
                                            model_2_btc_means[daypart_idx, cool_cluster_idx] * outdoor_temp_c[
                                                hour])

                                        model_2_lower_log.append(
                                            model_2_hdi['a_cluster'][daypart_idx, cluster_idx].sel(
                                                hdi='lower').values + \
                                            model_2_hdi['bs1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_2_hdi['bs2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_2_hdi['bs3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_2_hdi['bs4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_2_hdi['bs5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_2_hdi['bc1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_2_hdi['bc2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_2_hdi['bc3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_2_hdi['bc4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_2_hdi['bc5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_2_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_h[hour] + \
                                            model_2_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_c[hour] - \
                                            model_2_hdi['sigma'].sel(hdi='higher').values)

                                        model_2_higher_log.append(
                                            model_2_hdi['a_cluster'][daypart_idx, cluster_idx].sel(
                                                hdi='higher').values + \
                                            model_2_hdi['bs1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_2_hdi['bs2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_2_hdi['bs3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_2_hdi['bs4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_2_hdi['bs5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_2_hdi['bc1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_2_hdi['bc2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_2_hdi['bc3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_2_hdi['bc4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_2_hdi['bc5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_2_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_h[hour] + \
                                            model_2_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_c[hour] + \
                                            model_2_hdi['sigma'].sel(hdi='higher').values)

    # Calculate prediction error
    model_2_predictions = np.exp(model_2_log_predictions)
    model_2_mse = mean_squared_error(df.loc[test_index].total_electricity, model_2_predictions)
    model_2_rmse = sqrt(model_2_mse)
    model_2_cvrmse = model_2_rmse / df.total_electricity.mean()
    model_2_cv_accuracy.append(model_2_cvrmse)
    print('Model 2 cv cv(rmse): ', model_2_cv_accuracy, '%')

    # Coverage calculations
    model_2_higher = np.exp(model_2_higher_log)
    model_2_lower = np.exp(model_2_lower_log)
    coverage_2 = np.where(df.total_electricity[test_index] <= model_2_higher, np.where(model_2_predictions >= model_2_lower, 1, 0), 0)
    coverage_perc_2 = sum(coverage_2) * 100 / len(coverage_2)
    coverage_2_list.append(coverage_perc_2)
    print('Model 2 coverage: ', coverage_2_list, '%')


    # ---- MODEL 3) Intercept pooled on dayhour and profile; Fourier pooled on profile; Temperature unpooled on daypart and cluster

    with pm.Model(coords=coords) as model_3:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters[train_index], dims="obs_id")
        heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters[train_index], dims="obs_id")
        cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters[train_index], dims="obs_id")
        dayhour = pm.Data("dayhour", dayhours[train_index], dims="obs_id")
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
        a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("dayhour", "profile_cluster"))

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
        mu = a_cluster[dayhour, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[
            profile_cluster_idx] * fs_sin_2 + \
             bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
             bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
             bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
             bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
             btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_3_trace = approx.sample(1000)
        model_3_idata = az.from_pymc3(model_3_trace)
        az.plot_trace(model_3_idata)
        plt.save('results/plots/model_3_trace.png')

    # Calculate predictions
    model_3_acluster_means = np.mean(model_3_trace['a_cluster'], axis=0)
    model_3_bs1_means = np.mean(model_3_trace['bs1'], axis=0)
    model_3_bs2_means = np.mean(model_3_trace['bs2'], axis=0)
    model_3_bs3_means = np.mean(model_3_trace['bs3'], axis=0)
    model_3_bs4_means = np.mean(model_3_trace['bs4'], axis=0)
    model_3_bs5_means = np.mean(model_3_trace['bs5'], axis=0)
    model_3_bc1_means = np.mean(model_3_trace['bc1'], axis=0)
    model_3_bc2_means = np.mean(model_3_trace['bc2'], axis=0)
    model_3_bc3_means = np.mean(model_3_trace['bc3'], axis=0)
    model_3_bc4_means = np.mean(model_3_trace['bc4'], axis=0)
    model_3_bc5_means = np.mean(model_3_trace['bc5'], axis=0)

    model_3_bth_means = np.mean(model_3_trace['bth'], axis=0)
    model_3_btc_means = np.mean(model_3_trace['btc'], axis=0)
    # Create array with predictions
    model_3_log_predictions = []
    # Create array with bounds
    model_3_hdi = az.hdi(model_3_idata)
    model_3_lower_log = []
    model_3_higher_log = []

    for hour, row in df.loc[test_index].iterrows():
        for cluster_idx in unique_clusters:
            if clusters[hour] == cluster_idx:
                for heat_cluster_idx in unique_heat_clusters:
                    if heat_clusters[hour] == heat_cluster_idx:
                        for cool_cluster_idx in unique_cool_clusters:
                            if cool_clusters[hour] == cool_cluster_idx:
                                for dayhour_idx in unique_dayhour:
                                    if dayhours[hour] == dayhour_idx:
                                        for daypart_idx in unique_dayparts:
                                            if dayparts[hour] == daypart_idx:
                                                model_3_log_predictions.append(
                                                    model_3_acluster_means[dayhour_idx, cluster_idx] + \
                                                    model_3_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                                    model_3_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                                    model_3_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                                    model_3_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                                    model_3_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                                    model_3_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                                    model_3_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                                    model_3_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                                    model_3_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                                    model_3_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                                    model_3_bth_means[daypart_idx, heat_cluster_idx] * outdoor_temp_h[
                                                        hour] + \
                                                    model_3_btc_means[daypart_idx, cool_cluster_idx] * outdoor_temp_c[
                                                        hour])

                                                model_3_lower_log.append(
                                                    model_3_hdi['a_cluster'][dayhour_idx, cluster_idx].sel(
                                                        hdi='lower').values + \
                                                    model_3_hdi['bs1'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_sin_1[hour] + \
                                                    model_3_hdi['bs2'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_sin_2[hour] + \
                                                    model_3_hdi['bs3'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_sin_3[hour] + \
                                                    model_3_hdi['bs4'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_sin_4[hour] + \
                                                    model_3_hdi['bs5'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_sin_5[hour] + \
                                                    model_3_hdi['bc1'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_cos_1[hour] + \
                                                    model_3_hdi['bc2'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_cos_2[hour] + \
                                                    model_3_hdi['bc3'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_cos_3[hour] + \
                                                    model_3_hdi['bc4'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_cos_4[hour] + \
                                                    model_3_hdi['bc5'][cluster_idx].sel(hdi='lower').values *
                                                    daypart_fs_cos_5[hour] + \
                                                    model_3_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                        hdi='lower').values * outdoor_temp_h[hour] + \
                                                    model_3_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                        hdi='lower').values * outdoor_temp_c[hour] - \
                                                    model_3_hdi['sigma'].sel(hdi='higher').values)

                                                model_3_higher_log.append(
                                                    model_3_hdi['a_cluster'][dayhour_idx, cluster_idx].sel(
                                                        hdi='higher').values + \
                                                    model_3_hdi['bs1'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_sin_1[hour] + \
                                                    model_3_hdi['bs2'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_sin_2[hour] + \
                                                    model_3_hdi['bs3'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_sin_3[hour] + \
                                                    model_3_hdi['bs4'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_sin_4[hour] + \
                                                    model_3_hdi['bs5'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_sin_5[hour] + \
                                                    model_3_hdi['bc1'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_cos_1[hour] + \
                                                    model_3_hdi['bc2'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_cos_2[hour] + \
                                                    model_3_hdi['bc3'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_cos_3[hour] + \
                                                    model_3_hdi['bc4'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_cos_4[hour] + \
                                                    model_3_hdi['bc5'][cluster_idx].sel(hdi='higher').values *
                                                    daypart_fs_cos_5[hour] + \
                                                    model_3_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                        hdi='higher').values * outdoor_temp_h[hour] + \
                                                    model_3_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                        hdi='higher').values * outdoor_temp_c[hour] + \
                                                    model_3_hdi['sigma'].sel(hdi='higher').values)

    # Calculate prediction error
    model_3_predictions = np.exp(model_3_log_predictions)
    model_3_mse = mean_squared_error(df.loc[test_index].total_electricity, model_3_predictions)
    model_3_rmse = sqrt(model_3_mse)
    model_3_cvrmse = model_3_rmse / df.total_electricity.mean()
    model_3_cv_accuracy.append(model_3_cvrmse)
    print('Model 3 cv cv(rmse): ', model_3_cv_accuracy, '%')

    # Coverage calculations
    model_3_higher = np.exp(model_3_higher_log)
    model_3_lower = np.exp(model_3_lower_log)
    coverage_3 = np.where(df.total_electricity[test_index] <= model_3_higher, np.where(model_3_predictions >= model_3_lower, 1, 0), 0)
    coverage_perc_3 = sum(coverage_3) * 100 / len(coverage_3)
    coverage_3_list.append(coverage_perc_3)
    print('Model 3 coverage: ', coverage_3_list, '%')


# ---- Model 4) Intercept pooled on profile; Fourier pooled on profile; Temperature unpooled on daypart and cluster

    with pm.Model(coords=coords) as model_4:
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
        a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("profile_cluster"))

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
        mu = a_cluster[profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[
            profile_cluster_idx] * fs_sin_2 + \
             bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
             bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
             bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
             bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
             btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_4_trace = approx.sample(1000)
        model_4_idata = az.from_pymc3(model_4_trace)
        az.plot_trace(model_4_idata)
        plt.save('results/plots/model_4_trace.png')

    # Calculate predictions
    model_4_acluster_means = np.mean(model_4_trace['a_cluster'], axis=0)
    model_4_bs1_means = np.mean(model_4_trace['bs1'], axis=0)
    model_4_bs2_means = np.mean(model_4_trace['bs2'], axis=0)
    model_4_bs3_means = np.mean(model_4_trace['bs3'], axis=0)
    model_4_bs4_means = np.mean(model_4_trace['bs4'], axis=0)
    model_4_bs5_means = np.mean(model_4_trace['bs5'], axis=0)
    model_4_bc1_means = np.mean(model_4_trace['bc1'], axis=0)
    model_4_bc2_means = np.mean(model_4_trace['bc2'], axis=0)
    model_4_bc3_means = np.mean(model_4_trace['bc3'], axis=0)
    model_4_bc4_means = np.mean(model_4_trace['bc4'], axis=0)
    model_4_bc5_means = np.mean(model_4_trace['bc5'], axis=0)

    model_4_bth_means = np.mean(model_4_trace['bth'], axis=0)
    model_4_btc_means = np.mean(model_4_trace['btc'], axis=0)
    # Create array with predictions
    model_4_log_predictions = []
    # Create array with bounds
    model_4_hdi = az.hdi(model_4_idata)
    model_4_lower_log = []
    model_4_higher_log = []

    for hour, row in df.loc[test_index].iterrows():
        for cluster_idx in unique_clusters:
            if clusters[hour] == cluster_idx:
                for heat_cluster_idx in unique_heat_clusters:
                    if heat_clusters[hour] == heat_cluster_idx:
                        for cool_cluster_idx in unique_cool_clusters:
                            if cool_clusters[hour] == cool_cluster_idx:
                                for daypart_idx in unique_dayparts:
                                    if dayparts[hour] == daypart_idx:
                                        model_4_log_predictions.append(
                                            model_4_acluster_means[cluster_idx] + \
                                            model_4_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                            model_4_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                            model_4_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                            model_4_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                            model_4_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                            model_4_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                            model_4_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                            model_4_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                            model_4_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                            model_4_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                            model_4_bth_means[daypart_idx, heat_cluster_idx] * outdoor_temp_h[
                                                hour] + \
                                            model_4_btc_means[daypart_idx, cool_cluster_idx] * outdoor_temp_c[
                                                hour])

                                        model_4_lower_log.append(
                                            model_4_hdi['a_cluster'][cluster_idx].sel(
                                                hdi='lower').values + \
                                            model_4_hdi['bs1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_4_hdi['bs2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_4_hdi['bs3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_4_hdi['bs4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_4_hdi['bs5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_4_hdi['bc1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_4_hdi['bc2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_4_hdi['bc3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_4_hdi['bc4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_4_hdi['bc5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_4_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_h[hour] + \
                                            model_4_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_c[hour] - \
                                            model_4_hdi['sigma'].sel(hdi='higher').values)

                                        model_4_higher_log.append(
                                            model_4_hdi['a_cluster'][cluster_idx].sel(
                                                hdi='higher').values + \
                                            model_4_hdi['bs1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_4_hdi['bs2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_4_hdi['bs3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_4_hdi['bs4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_4_hdi['bs5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_4_hdi['bc1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_4_hdi['bc2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_4_hdi['bc3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_4_hdi['bc4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_4_hdi['bc5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_4_hdi['bth'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_h[hour] + \
                                            model_4_hdi['btc'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_c[hour] + \
                                            model_4_hdi['sigma'].sel(hdi='higher').values)

    # Calculate prediction error
    model_4_predictions = np.exp(model_4_log_predictions)
    model_4_mse = mean_squared_error(df.loc[test_index].total_electricity, model_4_predictions)
    model_4_rmse = sqrt(model_4_mse)
    model_4_cvrmse = model_4_rmse / df.total_electricity.mean()
    model_4_cv_accuracy.append(model_4_cvrmse)
    print('Model 4 cv cvrmse: ', model_4_cv_accuracy, '%')

    # Coverage calculations
    model_4_higher = np.exp(model_4_higher_log)
    model_4_lower = np.exp(model_4_lower_log)
    coverage_4 = np.where(df.total_electricity[test_index] <= model_4_higher, np.where(model_4_predictions >= model_4_lower, 1, 0), 0)
    coverage_perc_4 = sum(coverage_4) * 100 / len(coverage_4)
    coverage_4_list.append(coverage_perc_4)
    print('Model 4 coverage: ', coverage_4_list, '%')


# ---- Model 5) Intercept pooled on daypart and profile; Fourier pooled on profile; Temperature pooled on daypart and cluster

    with pm.Model(coords=coords) as model_5:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters[train_index], dims="obs_id")
        heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters[train_index], dims="obs_id")
        cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters[train_index], dims="obs_id")
        daypart_idx = pm.Data("daypart", dayparts[train_index], dims="obs_id")

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
        bth = pm.Normal("bth", mu=0.0, sigma=1.0)
        sigma_bth = pm.Exponential("sigma_bth", 1.0)
        btc = pm.Normal("btc", mu=0.0, sigma=1.0)
        sigma_btc = pm.Exponential("sigma_btc", 1.0)

        btc_cluster = pm.Normal("btc_cluster", mu=btc, sigma=sigma_btc, dims=("daypart", "cool_cluster"))
        bth_cluster = pm.Normal("bth_cluster", mu=btc, sigma=sigma_bth, dims=("daypart", "heat_cluster"))

        # Varying intercepts
        a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("daypart","profile_cluster"))

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
        mu = a_cluster[daypart_idx, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[
            profile_cluster_idx] * fs_sin_2 + \
             bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
             bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
             bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
             bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
             btc_cluster[daypart_idx, cool_temp_cluster_idx] * cooling_temp + bth_cluster[daypart_idx, heat_temp_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_5_trace = approx.sample(1000)
        model_5_idata = az.from_pymc3(model_5_trace)
        az.plot_trace(model_5_idata)
        plt.save('results/plots/model_5_trace.png')

    # Calculate predictions
    model_5_acluster_means = np.mean(model_5_trace['a_cluster'], axis=0)
    model_5_bs1_means = np.mean(model_5_trace['bs1'], axis=0)
    model_5_bs2_means = np.mean(model_5_trace['bs2'], axis=0)
    model_5_bs3_means = np.mean(model_5_trace['bs3'], axis=0)
    model_5_bs4_means = np.mean(model_5_trace['bs4'], axis=0)
    model_5_bs5_means = np.mean(model_5_trace['bs5'], axis=0)
    model_5_bc1_means = np.mean(model_5_trace['bc1'], axis=0)
    model_5_bc2_means = np.mean(model_5_trace['bc2'], axis=0)
    model_5_bc3_means = np.mean(model_5_trace['bc3'], axis=0)
    model_5_bc4_means = np.mean(model_5_trace['bc4'], axis=0)
    model_5_bc5_means = np.mean(model_5_trace['bc5'], axis=0)

    model_5_bth_cluster_means = np.mean(model_5_trace['bth_cluster'], axis=0)
    model_5_btc_cluster_means = np.mean(model_5_trace['btc_cluster'], axis=0)
    # Create array with predictions
    model_5_log_predictions = []
    # Create array with bounds
    model_5_hdi = az.hdi(model_5_idata)
    model_5_lower_log = []
    model_5_higher_log = []

    for hour, row in df.loc[test_index].iterrows():
        for cluster_idx in unique_clusters:
            if clusters[hour] == cluster_idx:
                for heat_cluster_idx in unique_heat_clusters:
                    if heat_clusters[hour] == heat_cluster_idx:
                        for cool_cluster_idx in unique_cool_clusters:
                            if cool_clusters[hour] == cool_cluster_idx:
                                for daypart_idx in unique_dayparts:
                                    if dayparts[hour] == daypart_idx:
                                        model_5_log_predictions.append(
                                            model_5_acluster_means[daypart_idx, cluster_idx] + \
                                            model_5_bs1_means[cluster_idx] * daypart_fs_sin_1[hour] + \
                                            model_5_bs2_means[cluster_idx] * daypart_fs_sin_2[hour] + \
                                            model_5_bs3_means[cluster_idx] * daypart_fs_sin_3[hour] + \
                                            model_5_bs4_means[cluster_idx] * daypart_fs_sin_4[hour] + \
                                            model_5_bs5_means[cluster_idx] * daypart_fs_sin_5[hour] + \
                                            model_5_bc1_means[cluster_idx] * daypart_fs_cos_1[hour] + \
                                            model_5_bc2_means[cluster_idx] * daypart_fs_cos_2[hour] + \
                                            model_5_bc3_means[cluster_idx] * daypart_fs_cos_3[hour] + \
                                            model_5_bc4_means[cluster_idx] * daypart_fs_cos_4[hour] + \
                                            model_5_bc5_means[cluster_idx] * daypart_fs_cos_5[hour] + \
                                            model_5_bth_cluster_means[daypart_idx, heat_cluster_idx] * outdoor_temp_h[
                                                hour] + \
                                            model_5_btc_cluster_means[daypart_idx, cool_cluster_idx] * outdoor_temp_c[
                                                hour])

                                        model_5_lower_log.append(
                                            model_5_hdi['a_cluster'][daypart_idx, cluster_idx].sel(
                                                hdi='lower').values + \
                                            model_5_hdi['bs1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_5_hdi['bs2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_5_hdi['bs3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_5_hdi['bs4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_5_hdi['bs5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_5_hdi['bc1'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_5_hdi['bc2'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_5_hdi['bc3'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_5_hdi['bc4'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_5_hdi['bc5'][cluster_idx].sel(hdi='lower').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_5_hdi['bth_cluster'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_h[hour] + \
                                            model_5_hdi['btc_cluster'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='lower').values * outdoor_temp_c[hour] - \
                                            model_5_hdi['sigma'].sel(hdi='higher').values)

                                        model_5_higher_log.append(
                                            model_5_hdi['a_cluster'][daypart_idx, cluster_idx].sel(
                                                hdi='higher').values + \
                                            model_5_hdi['bs1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_1[hour] + \
                                            model_5_hdi['bs2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_2[hour] + \
                                            model_5_hdi['bs3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_3[hour] + \
                                            model_5_hdi['bs4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_4[hour] + \
                                            model_5_hdi['bs5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_sin_5[hour] + \
                                            model_5_hdi['bc1'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_1[hour] + \
                                            model_5_hdi['bc2'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_2[hour] + \
                                            model_5_hdi['bc3'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_3[hour] + \
                                            model_5_hdi['bc4'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_4[hour] + \
                                            model_5_hdi['bc5'][cluster_idx].sel(hdi='higher').values *
                                            daypart_fs_cos_5[hour] + \
                                            model_5_hdi['bth_cluster'][daypart_idx, heat_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_h[hour] + \
                                            model_5_hdi['btc_cluster'][daypart_idx, cool_cluster_idx].sel(
                                                hdi='higher').values * outdoor_temp_c[hour] + \
                                            model_5_hdi['sigma'].sel(hdi='higher').values)

    # Calculate prediction error
    model_5_predictions = np.exp(model_5_log_predictions)
    model_5_mse = mean_squared_error(df.loc[test_index].total_electricity, model_5_predictions)
    model_5_rmse = sqrt(model_5_mse)
    model_5_cvrmse = model_5_rmse / df.total_electricity.mean()
    model_5_cv_accuracy.append(model_5_cvrmse)
    print('Model 5 cv cv(rmse): ', model_5_cv_accuracy, '%')

    # Coverage calculations
    model_5_higher = np.exp(model_5_higher_log)
    model_5_lower = np.exp(model_5_lower_log)
    coverage_5 = np.where(df.total_electricity[test_index] <= model_5_higher, np.where(model_5_predictions >= model_5_lower, 1, 0), 0)
    coverage_perc_5 = sum(coverage_5) * 100 / len(coverage_5)
    coverage_5_list.append(coverage_perc_5)
    print('Model 5 coverage: ', coverage_5_list, '%')

#Export Results
cvrmse.append(np.mean(model_1_cv_accuracy), np.mean(model_2_cv_accuracy), np.mean(model_3_cv_accuracy), np.mean(model_4_cv_accuracy), np.mean(model_5_cv_accuracy))
coverage.append(np.mean(coverage_1_list), np.mean(coverage_2_list), np.mean(coverage_3_list), np.mean(coverage_4_list), np.mean(coverage_5_list))
models = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']

export_dict = {'cvrmse' : cvrmse, 'coverage' : coverage, 'models' : models}

df = pd.DataFrame(data = export_dict)

df.to_csv("optimization.csv")

# Export plots
output_file("results/plots/predictions_1.html")
p1 = figure(plot_width=800, plot_height=400, x_axis_type = 'datetime')

p1.line(df.t[test_index], model_1_predictions, color="navy", alpha=0.8)
p1.line(df.t[test_index], df.total_electricity[test_index], color="orange", alpha=0.6)
p1.varea(x = df.t[test_index], y1 = model_1_lower, y2 = model_1_higher, color = 'gray', alpha = 0.2)
save(p1)

output_file("results/plots/predictions_2.html")
p2 = figure(plot_width=800, plot_height=400, x_axis_type = 'datetime')

p2.line(df.t[test_index], model_2_predictions, color="navy", alpha=0.8)
p2.line(df.t[test_index], df.total_electricity[test_index], color="orange", alpha=0.6)
p2.varea(x = df.t[test_index], y1 = model_2_lower, y2 = model_2_higher, color = 'gray', alpha = 0.2)
save(p2)

output_file("results/plots/predictions_3.html")
p3 = figure(plot_width=800, plot_height=400, x_axis_type = 'datetime')

p3.line(df.t[test_index], model_3_predictions, color="navy", alpha=0.8)
p3.line(df.t[test_index], df.total_electricity[test_index], color="orange", alpha=0.6)
p3.varea(x = df.t[test_index], y1 = model_3_lower, y2 = model_3_higher, color = 'gray', alpha = 0.2)
save(p3)

output_file("results/plots/predictions_4.html")
p4 = figure(plot_width=800, plot_height=400, x_axis_type = 'datetime')

p4.line(df.t[test_index], model_4_predictions, color="navy", alpha=0.8)
p4.line(df.t[test_index], df.total_electricity[test_index], color="orange", alpha=0.6)
p4.varea(x = df.t[test_index], y1 = model_4_lower, y2 = model_4_higher, color = 'gray', alpha = 0.2)
save(p4)

output_file("results/plots/predictions_5.html")
p5 = figure(plot_width=800, plot_height=400, x_axis_type = 'datetime')

p5.line(df.t[test_index], model_5_predictions, color="navy", alpha=0.8)
p5.line(df.t[test_index], df.total_electricity[test_index], color="orange", alpha=0.6)
p5.varea(x = df.t[test_index], y1 = model_5_lower, y2 = model_5_higher, color = 'gray', alpha = 0.2)
save(p5)
