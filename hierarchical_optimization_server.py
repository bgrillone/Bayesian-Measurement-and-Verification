import arviz  as az
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt

# Optimize with 5 fold CV to compare partial pooling with other methods.
# Compare cross-validation CV(RMSE) and WAIC

# ---- DATA IMPORT AND PREPROCESSING
RANDOM_SEED = 8924

# Data import
df = pd.read_csv("/root/benedetto/data/Id50_preprocessed2.csv", index_col = 0)

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

# 1 - CV(RMSE) and coverage through cross-validation
# First run a cross-validation to calculate cv(rmse) and coverage on unseen data for the three pooling techniques

# Create kfold cross-validation splits

kf = KFold(n_splits = 5)
kf.get_n_splits(df)

# Create arrays to save model results
partial_pool_cvrmse_list = []
nopool_cvrmse_list = []
complete_pool_cvrmse_list = []

partial_pool_coverage_list = []
nopool_coverage_list = []
complete_pool_coverage_list = []

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

        partial_pool_posterior_hdi = pm.sample_posterior_predictive(partial_pooling_trace, keep_size=True)
        partial_pool_posterior = pm.sample_posterior_predictive(partial_pooling_trace)


    # Calculate predictions and HDI

    partial_pool_predictions = np.exp(partial_pool_posterior['y'].mean(0))
    partial_pool_hdi_data = az.hdi(partial_pool_posterior_hdi)
    partial_pool_lower_bound = np.array(np.exp(partial_pool_hdi_data.to_array().sel(hdi='lower'))).flatten()
    partial_pool_higher_bound = np.array(np.exp(partial_pool_hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate cvrmse and coverage of the HDI
    partial_pool_mse = mean_squared_error(df.total_electricity[test_index], partial_pool_predictions)
    partial_pool_rmse = sqrt(partial_pool_mse)
    partial_pool_cvrmse = partial_pool_rmse / df.total_electricity.mean()
    partial_pool_coverage = sum((partial_pool_lower_bound <= df.total_electricity[test_index]) & (df.total_electricity[test_index] <= partial_pool_higher_bound)) * 100 / len(test_index)

    partial_pool_cvrmse_list.append (partial_pool_cvrmse)
    partial_pool_coverage_list.append(partial_pool_coverage)


    with pm.Model(coords=coords) as no_pooling:
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
        a_cluster = pm.Normal("a", mu=0.0, sigma=1.0, dims=("daypart", "profile_cluster"))
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
        mu = a_cluster[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[profile_cluster_idx] * fs_sin_2 + \
             bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
             bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
             bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
             bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
             btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims='obs_id')

    # Fitting without sampling
    with no_pooling:
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        no_pooling_trace = approx.sample(1000)
        no_pooling_idata = az.from_pymc3(no_pooling_trace)

    # Sampling from the posterior setting test data to check the predictions on unseen data

    with no_pooling:

        pm.set_data({"profile_cluster_idx": clusters[test_index], "heat_temp_cluster_idx": heat_clusters[test_index],
                     "cool_temp_cluster_idx": cool_clusters[test_index], "daypart": dayparts[test_index], "fs_sin_1": daypart_fs_sin_1[test_index],
                     "fs_sin_2": daypart_fs_sin_2[test_index], "fs_sin_3": daypart_fs_sin_3[test_index], "fs_sin_4": daypart_fs_sin_4[test_index],
                     "fs_sin_5": daypart_fs_sin_5[test_index], "fs_cos_1": daypart_fs_cos_1[test_index], "fs_cos_2": daypart_fs_cos_2[test_index],
                     "fs_cos_3": daypart_fs_cos_3[test_index], "fs_cos_4": daypart_fs_cos_4[test_index], "fs_cos_5": daypart_fs_cos_5[test_index],
                     "cooling_temp": outdoor_temp_c[test_index], "heating_temp": outdoor_temp_h[test_index]})

        nopool_posterior_hdi = pm.sample_posterior_predictive(no_pooling_trace, keep_size=True)
        nopool_posterior = pm.sample_posterior_predictive(no_pooling_trace)


    # Calculate predictions and HDI

    nopool_predictions = np.exp(nopool_posterior['y'].mean(0))
    nopool_hdi_data = az.hdi(nopool_posterior_hdi)
    nopool_lower_bound = np.array(np.exp(nopool_hdi_data.to_array().sel(hdi='lower'))).flatten()
    nopool_higher_bound = np.array(np.exp(nopool_hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate cvrmse and coverage of the HDI
    nopool_mse = mean_squared_error(df.total_electricity[test_index], nopool_predictions)
    nopool_rmse = sqrt(nopool_mse)
    nopool_cvrmse = nopool_rmse / df.total_electricity.mean()
    nopool_coverage = sum((nopool_lower_bound <= df.total_electricity[test_index]) & (df.total_electricity[test_index] <= nopool_higher_bound)) * 100 / len(test_index)

    nopool_cvrmse_list.append (nopool_cvrmse)
    nopool_coverage_list.append(nopool_coverage)


    with pm.Model(coords=coords) as complete_pooling:

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
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        btc = pm.Normal("btc", mu=0.0, sigma=1.0)
        bth = pm.Normal("bth", mu=0.0, sigma=1.0)

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

        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims='obs_id')

    # Fitting without sampling
    with complete_pooling:
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        complete_pooling_trace = approx.sample(1000)
        complete_pooling_idata = az.from_pymc3(complete_pooling_trace)

    # Sampling from the posterior setting test data to check the predictions on unseen data

    with complete_pooling:

        pm.set_data({"fs_sin_1": daypart_fs_sin_1[test_index], "fs_sin_2": daypart_fs_sin_2[test_index], "fs_sin_3": daypart_fs_sin_3[test_index],
                     "fs_sin_4": daypart_fs_sin_4[test_index], "fs_sin_5": daypart_fs_sin_5[test_index], "fs_cos_1": daypart_fs_cos_1[test_index],
                     "fs_cos_2": daypart_fs_cos_2[test_index], "fs_cos_3": daypart_fs_cos_3[test_index], "fs_cos_4": daypart_fs_cos_4[test_index],
                     "fs_cos_5": daypart_fs_cos_5[test_index], "cooling_temp": outdoor_temp_c[test_index], "heating_temp": outdoor_temp_h[test_index]})

        complete_pool_posterior_hdi = pm.sample_posterior_predictive(complete_pooling_trace, keep_size=True)
        complete_pool_posterior = pm.sample_posterior_predictive(complete_pooling_trace)


    # Calculate predictions and HDI

    complete_pool_predictions = np.exp(complete_pool_posterior['y'].mean(0))
    complete_pool_hdi_data = az.hdi(complete_pool_posterior_hdi)
    complete_pool_lower_bound = np.array(np.exp(complete_pool_hdi_data.to_array().sel(hdi='lower'))).flatten()
    complete_pool_higher_bound = np.array(np.exp(complete_pool_hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate cvrmse and coverage of the HDI
    complete_pool_mse = mean_squared_error(df.total_electricity[test_index], complete_pool_predictions)
    complete_pool_rmse = sqrt(complete_pool_mse)
    complete_pool_cvrmse = complete_pool_rmse / df.total_electricity.mean()
    complete_pool_coverage = sum((complete_pool_lower_bound <= df.total_electricity[test_index]) & (df.total_electricity[test_index] <= complete_pool_higher_bound)) * 100 / len(test_index)

    complete_pool_cvrmse_list.append (complete_pool_cvrmse)
    complete_pool_coverage_list.append(complete_pool_coverage)

# 2 - LOO/WAIC for the three pooling techniques evaluated without cross-validation

# create coords for pymc3
coords_2 = {"obs_id": np.arange(temperature.size)}
coords_2["profile_cluster"] = unique_clusters
coords_2["heat_cluster"] = unique_heat_clusters
coords_2["cool_cluster"] = unique_cool_clusters
coords_2["daypart"] = unique_dayparts

# Bayesian linear model with Intercept, Fourier series for the seasonal features,
# temperatures, pooled on profile and temperature clustering

with pm.Model(coords=coords_2) as partial_pooling_2:
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
with partial_pooling_2:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])

partial_pooled_loo = az.loo(partial_pooling_trace, partial_pooling_2)
partial_pooled_waic = az.waic(partial_pooling_trace, partial_pooling_2)

with pm.Model(coords=coords_2) as no_pooling_2:
    profile_cluster_idx = pm.Data("profile_cluster_idx", clusters, dims="obs_id")
    heat_temp_cluster_idx = pm.Data("heat_temp_cluster_idx", heat_clusters, dims="obs_id")
    cool_temp_cluster_idx = pm.Data("cool_temp_cluster_idx", cool_clusters, dims="obs_id")
    daypart = pm.Data("daypart", dayparts, dims="obs_id")

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

    # Priors:
    a_cluster = pm.Normal("a", mu=0.0, sigma=1.0, dims=("daypart", "profile_cluster"))
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
    mu = a_cluster[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + bs2[
        profile_cluster_idx] * fs_sin_2 + \
         bs3[profile_cluster_idx] * fs_sin_3 + bs4[profile_cluster_idx] * fs_sin_4 + \
         bs5[profile_cluster_idx] * fs_sin_5 + bc1[profile_cluster_idx] * fs_cos_1 + \
         bc2[profile_cluster_idx] * fs_cos_2 + bc3[profile_cluster_idx] * fs_cos_3 + \
         bc4[profile_cluster_idx] * fs_cos_4 + bc5[profile_cluster_idx] * fs_cos_5 + \
         btc[daypart, cool_temp_cluster_idx] * cooling_temp + bth[daypart, heat_temp_cluster_idx] * heating_temp

    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims='obs_id')

# Fitting without sampling
with no_pooling_2:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    no_pooling_trace = approx.sample(1000)

# Sampling from the posterior setting test data to check the predictions on unseen data

with pm.Model(coords=coords_2) as complete_pooling_2:
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

    # Priors:
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    btc = pm.Normal("btc", mu=0.0, sigma=1.0)
    bth = pm.Normal("bth", mu=0.0, sigma=1.0)

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

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims='obs_id')

# Fitting without sampling
with complete_pooling_2:
    approx = pm.fit(n=50000,
                    method='fullrank_advi',
                    callbacks=[CheckParametersConvergence(tolerance=0.01)])
    complete_pooling_trace = approx.sample(1000)

#Compare the LOO of the 3 complete models

df_comp_loo = az.compare({'partial_pooling': partial_pooling_trace, 'no_pooling': no_pooling_trace, 'complete_pooling': complete_pooling_trace})

#Export Results
cvrmse_list = [np.mean(partial_pool_cvrmse_list), np.mean(complete_pool_cvrmse_list), np.mean(nopool_cvrmse_list)]
coverage_list = [np.mean(partial_pool_coverage_list), np.mean(complete_pool_coverage_list), np.mean(nopool_coverage_list)]
models = ['partial_pooling', 'complete_pooling', 'no_pooling']
export_dict = {'cvrmse' : cvrmse_list, 'coverage' : coverage_list}
export_df = pd.DataFrame(data = export_dict, index=models)
export_df = export_df.join(df_comp_loo)

export_df.to_csv("/root/benedetto/results/optimization.csv")