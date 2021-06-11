import arviz  as az
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import subprocess
import os


def bayesian_model_comparison (df):
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values
    total_electricity = df.total_electricity.values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1
    df.weekday = df.weekday - 1
    clusters = df.s
    unique_clusters = clusters.unique()
    dayparts = df.daypart
    weekdays = df.weekday
    unique_dayparts = dayparts.unique()
    unique_weekdays = weekdays.unique()
    n_hours = len(df.index)
    outdoor_temp_c = df.outdoor_temp_c
    outdoor_temp_h = df.outdoor_temp_h
    outdoor_temp_lp_c = df.outdoor_temp_lp_c
    outdoor_temp_lp_h = df.outdoor_temp_lp_h
    daypart_fs_sin_1 = df.daypart_fs_sin_1
    daypart_fs_sin_2 = df.daypart_fs_sin_2
    daypart_fs_sin_3 = df.daypart_fs_sin_3
    daypart_fs_cos_1 = df.daypart_fs_cos_1
    daypart_fs_cos_2 = df.daypart_fs_cos_2
    daypart_fs_cos_3 = df.daypart_fs_cos_3

    # create coords for pymc3
    coords = {"obs_id": np.arange(total_electricity.size)}
    coords["profile_cluster"] = unique_clusters
    coords["daypart"] = unique_dayparts
    coords["weekday"] = unique_weekdays

    # Create kfold cross-validation splits

    kf = KFold(n_splits=5)
    kf.get_n_splits(df)

    # Create arrays to save model results
    partial_pool_cvrmse_list = []
    no_pool_cvrmse_list = []
    complete_pool_cvrmse_list = []

    partial_pool_coverage_list = []
    no_pool_coverage_list = []
    complete_pool_coverage_list = []

    partial_pool_confidence_length_list = []
    no_pool_confidence_length_list = []
    complete_pool_confidence_length_list = []

    for train_index, test_index in kf.split(df):
        coords = {"obs_id": np.arange(total_electricity[train_index].size)}
        coords["profile_cluster"] = unique_clusters
        coords["daypart"] = unique_dayparts
        coords["weekday"] = unique_weekdays

        # Partial Pooling

        with pm.Model(coords=coords) as partial_pooling:
            profile_cluster_idx = pm.Data("profile_cluster_idx", clusters[train_index], dims="obs_id")
            daypart = pm.Data("daypart", dayparts[train_index], dims="obs_id")
            weekday = pm.Data("weekday", weekdays[train_index], dims="obs_id")

            fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1[train_index], dims="obs_id")
            fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2[train_index], dims="obs_id")
            fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3[train_index], dims="obs_id")

            fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1[train_index], dims="obs_id")
            fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2[train_index], dims="obs_id")
            fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3[train_index], dims="obs_id")

            # cooling_temp = pm.Data("cooling_temp", outdoor_temp_c[train_index], dims="obs_id")
            # heating_temp = pm.Data("heating_temp", outdoor_temp_h[train_index], dims="obs_id")
            cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c[train_index], dims="obs_id")
            heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h[train_index], dims="obs_id")

            # Hyperpriors:
            bf = pm.Normal("bf", mu=0.0, sigma=1.0)
            sigma_bf = pm.Exponential("sigma_bf", 1.0)
            a = pm.Normal("a", mu=0.0, sigma=1.0)
            sigma_a = pm.Exponential("sigma_a", 1.0)

            # btc = pm.Normal("btc", mu=0.0, sigma=1.0, dims="daypart")
            # bth = pm.Normal("bth", mu=0.0, sigma=1.0, dims="daypart")

            btclp = pm.Normal("btclp", mu=0.0, sigma=1.0, dims="daypart")
            bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0, dims="daypart")

            # Varying intercepts
            a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("daypart", "profile_cluster"))

            # Varying slopes:
            bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
            bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
            bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

            bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
            bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
            bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

            # Expected value per county:
            mu = a_cluster[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + \
                 bs2[profile_cluster_idx] * fs_sin_2 + bs3[profile_cluster_idx] * fs_sin_3 + \
                 bc1[profile_cluster_idx] * fs_cos_1 + bc2[profile_cluster_idx] * fs_cos_2 + \
                 bc3[profile_cluster_idx] * fs_cos_3 + \
                 btclp[daypart] * cooling_temp_lp + \
                 bthlp[daypart] * heating_temp_lp
            # btc[daypart] * cooling_temp + bth[daypart] * heating_temp + \

            # Model error:
            sigma = pm.Exponential("sigma", 1.0)

            # Likelihood
            y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        # Fitting
        with partial_pooling:
            approx = pm.fit(n=50000,
                            method='fullrank_advi',
                            callbacks=[CheckParametersConvergence(tolerance=0.01)])
            partial_pooling_trace = approx.sample(1000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

        with partial_pooling:
            pm.set_data({"profile_cluster_idx": clusters[test_index], "daypart": dayparts[test_index],  # "weekday":weekdays,
                         "fs_sin_1": daypart_fs_sin_1[test_index], "fs_sin_2": daypart_fs_sin_2[test_index], "fs_sin_3": daypart_fs_sin_3[test_index],
                         "fs_cos_1": daypart_fs_cos_1[test_index], "fs_cos_2": daypart_fs_cos_2[test_index], "fs_cos_3": daypart_fs_cos_3[test_index],
                         # "cooling_temp":outdoor_temp_c, "heating_temp": outdoor_temp_h,
                         "cooling_temp_lp": outdoor_temp_lp_c[test_index],
                         "heating_temp_lp": outdoor_temp_lp_h[test_index]
                         })

            partial_pool_posterior_hdi = pm.sample_posterior_predictive(partial_pooling_trace, keep_size=True)
            partial_pool_posterior = pm.sample_posterior_predictive(partial_pooling_trace)
            partial_pool_prior = pm.sample_prior_predictive(150)


        # Calculate predictions and HDI

        partial_pool_predictions = np.exp(partial_pool_posterior['y'].mean(0))
        hdi_data = az.hdi(partial_pool_posterior_hdi)
        partial_pool_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
        partial_pool_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

        # Calculate cvrmse and coverage of the HDI
        partial_pool_mse = mean_squared_error(df.total_electricity[test_index], partial_pool_predictions)
        partial_pool_rmse = sqrt(partial_pool_mse)
        partial_pool_cvrmse = partial_pool_rmse / df.total_electricity.mean()
        partial_pool_coverage = sum((partial_pool_lower_bound <= df.total_electricity[test_index]) & (
                    df.total_electricity[test_index] <= partial_pool_higher_bound)) * 100 / len(test_index)
        partial_pool_confidence_length = sum(partial_pool_higher_bound) - sum(partial_pool_lower_bound)

        partial_pool_cvrmse_list.append(partial_pool_cvrmse)
        partial_pool_coverage_list.append(partial_pool_coverage)
        partial_pool_confidence_length_list.append(partial_pool_confidence_length)

        # No Pooling

        with pm.Model(coords=coords) as no_pooling:
            profile_cluster_idx = pm.Data("profile_cluster_idx", clusters[train_index], dims="obs_id")
            daypart = pm.Data("daypart", dayparts[train_index], dims="obs_id")
            weekday = pm.Data("weekday", weekdays[train_index], dims="obs_id")

            fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1[train_index], dims="obs_id")
            fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2[train_index], dims="obs_id")
            fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3[train_index], dims="obs_id")

            fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1[train_index], dims="obs_id")
            fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2[train_index], dims="obs_id")
            fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3[train_index], dims="obs_id")

            # cooling_temp = pm.Data("cooling_temp", outdoor_temp_c[train_index], dims="obs_id")
            # heating_temp = pm.Data("heating_temp", outdoor_temp_h[train_index], dims="obs_id")
            cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c[train_index], dims="obs_id")
            heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h[train_index], dims="obs_id")

            # Priors:
            a_cluster = pm.Normal("a_cluster", mu=0.0, sigma=1.0, dims=("daypart", "profile_cluster"))
            btclp = pm.Normal("btclp", mu=0.0, sigma=1.0, dims="daypart")
            bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0, dims="daypart")

            bs1 = pm.Normal("bs1", mu=0.0, sigma=1.0, dims="profile_cluster")
            bs2 = pm.Normal("bs2", mu=0.0, sigma=1.0, dims="profile_cluster")
            bs3 = pm.Normal("bs3", mu=0.0, sigma=1.0, dims="profile_cluster")
            bc1 = pm.Normal("bc1", mu=0.0, sigma=1.0, dims="profile_cluster")
            bc2 = pm.Normal("bc2", mu=0.0, sigma=1.0, dims="profile_cluster")
            bc3 = pm.Normal("bc3", mu=0.0, sigma=1.0, dims="profile_cluster")

            # Expected value per county:
            mu = a_cluster[daypart, profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + \
                 bs2[profile_cluster_idx] * fs_sin_2 + bs3[profile_cluster_idx] * fs_sin_3 + \
                 bc1[profile_cluster_idx] * fs_cos_1 + bc2[profile_cluster_idx] * fs_cos_2 + \
                 bc3[profile_cluster_idx] * fs_cos_3 + \
                 btclp[daypart] * cooling_temp_lp + \
                 bthlp[daypart] * heating_temp_lp
            # btc[daypart] * cooling_temp + bth[daypart] * heating_temp + \

            # Model error:
            sigma = pm.Exponential("sigma", 1.0)

            # Likelihood
            y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        # Fitting

        with no_pooling:
            approx = pm.fit(n=50000,
                            method='fullrank_advi',
                            callbacks=[CheckParametersConvergence(tolerance=0.01)])
            no_pooling_trace = approx.sample(1000)

            # Sampling from the posterior setting test data to check the predictions on unseen data

        with no_pooling:
            pm.set_data(
                {"profile_cluster_idx": clusters[test_index], "daypart": dayparts[test_index],  # "weekday":weekdays,
                 "fs_sin_1": daypart_fs_sin_1[test_index], "fs_sin_2": daypart_fs_sin_2[test_index],
                 "fs_sin_3": daypart_fs_sin_3[test_index],
                 "fs_cos_1": daypart_fs_cos_1[test_index], "fs_cos_2": daypart_fs_cos_2[test_index],
                 "fs_cos_3": daypart_fs_cos_3[test_index],
                 # "cooling_temp":outdoor_temp_c, "heating_temp": outdoor_temp_h,
                 "cooling_temp_lp": outdoor_temp_lp_c[test_index],
                 "heating_temp_lp": outdoor_temp_lp_h[test_index]
                 })

            no_pool_posterior_hdi = pm.sample_posterior_predictive(no_pooling_trace, keep_size=True)
            no_pool_posterior = pm.sample_posterior_predictive(no_pooling_trace)

            no_pool_prior = pm.sample_prior_predictive(150)

            # Calculate predictions and HDI

        no_pool_predictions = np.exp(no_pool_posterior['y'].mean(0))
        no_pool_hdi_data = az.hdi(no_pool_posterior_hdi)
        no_pool_lower_bound = np.array(np.exp(no_pool_hdi_data.to_array().sel(hdi='lower'))).flatten()
        no_pool_higher_bound = np.array(np.exp(no_pool_hdi_data.to_array().sel(hdi='higher'))).flatten()

        # Calculate cvrmse and coverage of the HDI
        no_pool_mse = mean_squared_error(df.total_electricity[test_index], no_pool_predictions)
        no_pool_rmse = sqrt(no_pool_mse)
        no_pool_cvrmse = no_pool_rmse / df.total_electricity.mean()
        no_pool_coverage = sum((no_pool_lower_bound <= df.total_electricity[test_index]) & (
                df.total_electricity[test_index] <= no_pool_higher_bound)) * 100 / len(test_index)
        no_pool_confidence_length = sum(no_pool_higher_bound) - sum(no_pool_lower_bound)

        no_pool_cvrmse_list.append(no_pool_cvrmse)
        no_pool_coverage_list.append(no_pool_coverage)
        no_pool_confidence_length_list.append(no_pool_confidence_length)

        # Complete pooling

        with pm.Model(coords=coords) as complete_pooling:

            fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1[train_index], dims="obs_id")
            fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2[train_index], dims="obs_id")
            fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3[train_index], dims="obs_id")

            fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1[train_index], dims="obs_id")
            fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2[train_index], dims="obs_id")
            fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3[train_index], dims="obs_id")

            # cooling_temp = pm.Data("cooling_temp", outdoor_temp_c[train_index], dims="obs_id")
            # heating_temp = pm.Data("heating_temp", outdoor_temp_h[train_index], dims="obs_id")
            cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c[train_index], dims="obs_id")
            heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h[train_index], dims="obs_id")

            # Priors:
            a = pm.Normal("a", mu=0.0, sigma=1.0)
            btclp = pm.Normal("btclp", mu=0.0, sigma=1.0)
            bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0)

            bs1 = pm.Normal("bs1", mu=0.0, sigma=1.0)
            bs2 = pm.Normal("bs2", mu=0.0, sigma=1.0)
            bs3 = pm.Normal("bs3", mu=0.0, sigma=1.0)
            bc1 = pm.Normal("bc1", mu=0.0, sigma=1.0)
            bc2 = pm.Normal("bc2", mu=0.0, sigma=1.0)
            bc3 = pm.Normal("bc3", mu=0.0, sigma=1.0)

            # Expected value per county:
            mu = a + bs1 * fs_sin_1 + bs2 * fs_sin_2 + bs3 * fs_sin_3 + bc1 * fs_cos_1 + bc2 * fs_cos_2 + \
                 bc3 * fs_cos_3 + btclp * cooling_temp_lp + bthlp * heating_temp_lp
            # btc[daypart] * cooling_temp + bth[daypart] * heating_temp + \

            # Model error:
            sigma = pm.Exponential("sigma", 1.0)

            # Likelihood
            y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity[train_index], dims="obs_id")

        # Fitting

        with complete_pooling:
            approx = pm.fit(n=50000,
                            method='fullrank_advi',
                            callbacks=[CheckParametersConvergence(tolerance=0.01)])
            complete_pooling_trace = approx.sample(1000)

            # Sampling from the posterior setting test data to check the predictions on unseen data

        with complete_pooling:
            pm.set_data(
                {"fs_sin_1": daypart_fs_sin_1[test_index], "fs_sin_2": daypart_fs_sin_2[test_index],
                 "fs_sin_3": daypart_fs_sin_3[test_index],
                 "fs_cos_1": daypart_fs_cos_1[test_index], "fs_cos_2": daypart_fs_cos_2[test_index],
                 "fs_cos_3": daypart_fs_cos_3[test_index],
                 # "cooling_temp":outdoor_temp_c, "heating_temp": outdoor_temp_h,
                 "cooling_temp_lp": outdoor_temp_lp_c[test_index],
                 "heating_temp_lp": outdoor_temp_lp_h[test_index]
                 })

            complete_pool_posterior_hdi = pm.sample_posterior_predictive(complete_pooling_trace, keep_size=True)
            complete_pool_posterior = pm.sample_posterior_predictive(complete_pooling_trace)

            complete_pool_prior = pm.sample_prior_predictive(150)

            # Calculate predictions and HDI

        complete_pool_predictions = np.exp(complete_pool_posterior['y'].mean(0))
        complete_pool_hdi_data = az.hdi(complete_pool_posterior_hdi)
        complete_pool_lower_bound = np.array(np.exp(complete_pool_hdi_data.to_array().sel(hdi='lower'))).flatten()
        complete_pool_higher_bound = np.array(np.exp(complete_pool_hdi_data.to_array().sel(hdi='higher'))).flatten()

        # Calculate cvrmse and coverage of the HDI
        complete_pool_mse = mean_squared_error(df.total_electricity[test_index], complete_pool_predictions)
        complete_pool_rmse = sqrt(complete_pool_mse)
        complete_pool_cvrmse = complete_pool_rmse / df.total_electricity.mean()
        complete_pool_coverage = sum((complete_pool_lower_bound <= df.total_electricity[test_index]) & (
                df.total_electricity[test_index] <= complete_pool_higher_bound)) * 100 / len(test_index)
        complete_pool_confidence_length = sum(complete_pool_higher_bound) - sum(complete_pool_lower_bound)

        complete_pool_cvrmse_list.append(complete_pool_cvrmse)
        complete_pool_coverage_list.append(complete_pool_coverage)
        complete_pool_confidence_length_list.append(complete_pool_confidence_length)

    # Export Results
    np_cvrmse = np.mean(no_pool_cvrmse_list)
    cp_cvrmse = np.mean(complete_pool_cvrmse_list)
    pp_cvrmse = np.mean(partial_pool_cvrmse_list)

    np_coverage = np.mean(no_pool_coverage_list)
    cp_coverage =  np.mean(complete_pool_coverage_list)
    pp_coverage = np.mean(partial_pool_coverage_list)

    np_length = np.mean(no_pool_confidence_length_list)
    cp_length = np.mean(complete_pool_confidence_length_list)
    pp_length = np.mean(partial_pool_confidence_length_list)

    export_data = {'partial_pooling_cvrmse': [pp_cvrmse], 'no_pooling_cvrmse': [np_cvrmse], 'complete_pooling_cvrmse': [cp_cvrmse],
                   'partial_pooling_coverage': [pp_coverage], 'no_pooling_coverage': [np_coverage],
                   'complete_pooling_coverage': [cp_coverage], 'partial_pooling_length':[pp_length],
                   'no_pooling_length': [np_length], 'complete_pooling_length': [cp_length]}
    export_df = pd.DataFrame(data=export_data)
    return export_df

def bayesian_model_comparison_whole_year (df):
    #df = pd.read_csv("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Panther_lodging_Janice_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1
    df.weekday = df.weekday - 1

    # Create training and test set (for the ashrae data training is 2016, test is 2017)
    train_df = df.loc[df["t"] <= pd.to_datetime("2017-01-01")]
    test_df = df.loc[df["t"] > pd.to_datetime("2017-01-01")]

    # Define Training variables
    total_electricity_train = train_df.total_electricity.values
    log_electricity_train = train_df['log_v']
    clusters_train = train_df.s
    unique_clusters = clusters_train.unique()
    dayparts_train = train_df.daypart
    weekdays_train = train_df.weekday
    unique_dayparts = dayparts_train.unique()
    unique_weekdays = weekdays_train.unique()
    outdoor_temp_c_train = train_df.outdoor_temp_c
    outdoor_temp_h_train = train_df.outdoor_temp_h
    outdoor_temp_lp_c_train = train_df.outdoor_temp_lp_c
    outdoor_temp_lp_h_train = train_df.outdoor_temp_lp_h
    daypart_fs_sin_1_train = train_df.daypart_fs_sin_1
    daypart_fs_sin_2_train = train_df.daypart_fs_sin_2
    daypart_fs_sin_3_train = train_df.daypart_fs_sin_3
    daypart_fs_cos_1_train = train_df.daypart_fs_cos_1
    daypart_fs_cos_2_train = train_df.daypart_fs_cos_2
    daypart_fs_cos_3_train = train_df.daypart_fs_cos_3

    # Define test variables
    clusters_test = test_df.s
    dayparts_test = test_df.daypart
    weekdays_test = test_df.weekday
    outdoor_temp_c_test = test_df.outdoor_temp_c
    outdoor_temp_h_test = test_df.outdoor_temp_h
    outdoor_temp_lp_c_test = test_df.outdoor_temp_lp_c
    outdoor_temp_lp_h_test = test_df.outdoor_temp_lp_h
    daypart_fs_sin_1_test = test_df.daypart_fs_sin_1
    daypart_fs_sin_2_test = test_df.daypart_fs_sin_2
    daypart_fs_sin_3_test = test_df.daypart_fs_sin_3
    daypart_fs_cos_1_test = test_df.daypart_fs_cos_1
    daypart_fs_cos_2_test = test_df.daypart_fs_cos_2
    daypart_fs_cos_3_test = test_df.daypart_fs_cos_3

    # create coords for pymc3
    coords = {"obs_id": np.arange(total_electricity_train.size)}
    coords["profile_cluster"] = unique_clusters
    coords["daypart"] = unique_dayparts
    coords["weekday"] = unique_weekdays

    # Partial Pooling

    with pm.Model(coords=coords) as partial_pooling:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")
        weekday = pm.Data("weekday", weekdays_train, dims="obs_id")

        fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        cooling_temp = pm.Data("cooling_temp", outdoor_temp_c_train, dims="obs_id")
        heating_temp = pm.Data("heating_temp", outdoor_temp_h_train, dims="obs_id")
        cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c_train, dims="obs_id")
        heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h_train, dims="obs_id")

        # Hyperpriors:
        bf = pm.Normal("bf", mu=0.0, sigma=1.0)
        sigma_bf = pm.Exponential("sigma_bf", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)

        btc = pm.Normal("btc", mu=0.0, sigma=1.0, dims="daypart")
        bth = pm.Normal("bth", mu=0.0, sigma=1.0, dims="daypart")

        btclp = pm.Normal("btclp", mu=0.0, sigma=1.0, dims="daypart")
        bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0, dims="daypart")

        # Varying intercepts
        a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("profile_cluster"))

        # Varying slopes:
        bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + \
             bs2[profile_cluster_idx] * fs_sin_2 + bs3[profile_cluster_idx] * fs_sin_3 + \
             bc1[profile_cluster_idx] * fs_cos_1 + bc2[profile_cluster_idx] * fs_cos_2 + \
             bc3[profile_cluster_idx] * fs_cos_3 + \
             btclp[daypart] * cooling_temp_lp + \
             bthlp[daypart] * heating_temp_lp + btc[daypart] * cooling_temp + \
             bth[daypart] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    # Fitting
    with partial_pooling:
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        partial_pooling_trace = approx.sample(1000)

    # Sampling from the posterior setting test data to check the predictions on unseen data

    with partial_pooling:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     # "weekday":weekdays_test,
                     "fs_sin_1": daypart_fs_sin_1_test,
                     "fs_sin_2": daypart_fs_sin_2_test,
                     "fs_sin_3": daypart_fs_sin_3_test,
                     "fs_cos_1": daypart_fs_cos_1_test,
                     "fs_cos_2": daypart_fs_cos_2_test,
                     "fs_cos_3": daypart_fs_cos_3_test,
                     "cooling_temp":outdoor_temp_c_test,
                     "heating_temp": outdoor_temp_h_test,
                     "cooling_temp_lp": outdoor_temp_lp_c_test,
                     "heating_temp_lp": outdoor_temp_lp_h_test
                     })

        partial_pool_posterior_hdi = pm.sample_posterior_predictive(partial_pooling_trace, keep_size=True)
        partial_pool_posterior = pm.sample_posterior_predictive(partial_pooling_trace)
        partial_pool_prior = pm.sample_prior_predictive(150)


    # Calculate predictions and HDI

    partial_pool_predictions = np.exp(partial_pool_posterior['y'].mean(0))
    hdi_data = az.hdi(partial_pool_posterior_hdi)
    partial_pool_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    partial_pool_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    partial_pool_gamma =  np.where(test_df.total_electricity > partial_pool_higher_bound,
                      1 + (np.log(test_df.total_electricity - partial_pool_higher_bound)/np.log(test_df.total_electricity)),
                      np.where(test_df.total_electricity < partial_pool_lower_bound,
                               1 + (np.log(partial_pool_lower_bound - test_df.total_electricity)/np.log(test_df.total_electricity)),
                               1))

    partial_pool_adjusted_coverage = 1/len(test_df) * np.sum(partial_pool_gamma * np.log(partial_pool_higher_bound) /np.log(partial_pool_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    partial_pool_mse = mean_squared_error(test_df.total_electricity, partial_pool_predictions)
    partial_pool_rmse = sqrt(partial_pool_mse)
    partial_pool_cvrmse = partial_pool_rmse / test_df.total_electricity.mean()
    partial_pool_coverage = sum((partial_pool_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= partial_pool_higher_bound)) * 100 / len(test_df)
    partial_pool_confidence_length = sum(partial_pool_higher_bound) - sum(partial_pool_lower_bound)

    # Calculate NMBE
    partial_pool_nmbe = np.sum(test_df.total_electricity - partial_pool_predictions) * 100 / len(test_df) / test_df.total_electricity.mean()

    # No Pooling

    with pm.Model(coords=coords) as no_pooling:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")
        weekday = pm.Data("weekday", weekdays_train, dims="obs_id")

        fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        cooling_temp = pm.Data("cooling_temp", outdoor_temp_c_train, dims="obs_id")
        heating_temp = pm.Data("heating_temp", outdoor_temp_h_train, dims="obs_id")
        cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c_train, dims="obs_id")
        heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h_train, dims="obs_id")

        # Priors:
        a_cluster = pm.Normal("a_cluster", mu=0.0, sigma=1.0, dims=("profile_cluster"))
        btclp = pm.Normal("btclp", mu=0.0, sigma=1.0, dims="daypart")
        bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0, dims="daypart")

        bs1 = pm.Normal("bs1", mu=0.0, sigma=1.0, dims="profile_cluster")
        bs2 = pm.Normal("bs2", mu=0.0, sigma=1.0, dims="profile_cluster")
        bs3 = pm.Normal("bs3", mu=0.0, sigma=1.0, dims="profile_cluster")
        bc1 = pm.Normal("bc1", mu=0.0, sigma=1.0, dims="profile_cluster")
        bc2 = pm.Normal("bc2", mu=0.0, sigma=1.0, dims="profile_cluster")
        bc3 = pm.Normal("bc3", mu=0.0, sigma=1.0, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + \
             bs2[profile_cluster_idx] * fs_sin_2 + bs3[profile_cluster_idx] * fs_sin_3 + \
             bc1[profile_cluster_idx] * fs_cos_1 + bc2[profile_cluster_idx] * fs_cos_2 + \
             bc3[profile_cluster_idx] * fs_cos_3 + \
             btclp[daypart] * cooling_temp_lp + \
             bthlp[daypart] * heating_temp_lp + \
             btc[daypart] * cooling_temp + \
             bth[daypart] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    # Fitting

    with no_pooling:
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        no_pooling_trace = approx.sample(1000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with no_pooling:
        pm.set_data(
            {"profile_cluster_idx": clusters_test,
             "daypart": dayparts_test,
             # "weekday":weekdays,
             "fs_sin_1": daypart_fs_sin_1_test,
             "fs_sin_2": daypart_fs_sin_2_test,
             "fs_sin_3": daypart_fs_sin_3_test,
             "fs_cos_1": daypart_fs_cos_1_test,
             "fs_cos_2": daypart_fs_cos_2_test,
             "fs_cos_3": daypart_fs_cos_3_test,
             "cooling_temp":outdoor_temp_c_test,
             "heating_temp": outdoor_temp_h_test,
             "cooling_temp_lp": outdoor_temp_lp_c_test,
             "heating_temp_lp": outdoor_temp_lp_h_test
             })

        no_pool_posterior_hdi = pm.sample_posterior_predictive(no_pooling_trace, keep_size=True)
        no_pool_posterior = pm.sample_posterior_predictive(no_pooling_trace)

        no_pool_prior = pm.sample_prior_predictive(150)

        # Calculate predictions and HDI

    no_pool_predictions = np.exp(no_pool_posterior['y'].mean(0))
    no_pool_hdi_data = az.hdi(no_pool_posterior_hdi)
    no_pool_lower_bound = np.array(np.exp(no_pool_hdi_data.to_array().sel(hdi='lower'))).flatten()
    no_pool_higher_bound = np.array(np.exp(no_pool_hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate wideness and completeness

    no_pool_gamma =  np.where(test_df.total_electricity > no_pool_higher_bound,
                      1 + (np.log(test_df.total_electricity - no_pool_higher_bound)/np.log(test_df.total_electricity)),
                      np.where(test_df.total_electricity < no_pool_lower_bound,
                               1 + (np.log(no_pool_lower_bound - test_df.total_electricity)/np.log(test_df.total_electricity)),
                               1))

    no_pool_adjusted_coverage = 1/len(test_df) * np.sum(no_pool_gamma * np.log(no_pool_higher_bound) /np.log(no_pool_lower_bound))


    # Calculate cvrmse and coverage of the HDI
    no_pool_mse = mean_squared_error(test_df.total_electricity, no_pool_predictions)
    no_pool_rmse = sqrt(no_pool_mse)
    no_pool_cvrmse = no_pool_rmse / test_df.total_electricity.mean()
    no_pool_coverage = sum((no_pool_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= no_pool_higher_bound)) * 100 / len(test_df)
    no_pool_confidence_length = sum(no_pool_higher_bound) - sum(no_pool_lower_bound)

    # Calculate NMBE
    no_pool_nmbe = np.sum(test_df.total_electricity - no_pool_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Complete pooling

    with pm.Model(coords=coords) as complete_pooling:

        fs_sin_1 = pm.Data("fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        fs_sin_2 = pm.Data("fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        fs_sin_3 = pm.Data("fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        fs_cos_1 = pm.Data("fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        fs_cos_2 = pm.Data("fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        fs_cos_3 = pm.Data("fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        cooling_temp = pm.Data("cooling_temp", outdoor_temp_c_train, dims="obs_id")
        heating_temp = pm.Data("heating_temp", outdoor_temp_h_train, dims="obs_id")
        cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c_train, dims="obs_id")
        heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h_train, dims="obs_id")

        # Priors:
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        btclp = pm.Normal("btclp", mu=0.0, sigma=1.0)
        bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0)

        bs1 = pm.Normal("bs1", mu=0.0, sigma=1.0)
        bs2 = pm.Normal("bs2", mu=0.0, sigma=1.0)
        bs3 = pm.Normal("bs3", mu=0.0, sigma=1.0)
        bc1 = pm.Normal("bc1", mu=0.0, sigma=1.0)
        bc2 = pm.Normal("bc2", mu=0.0, sigma=1.0)
        bc3 = pm.Normal("bc3", mu=0.0, sigma=1.0)

        # Expected value per county:
        mu = a + bs1 * fs_sin_1 + bs2 * fs_sin_2 + bs3 * fs_sin_3 + \
             bc1 * fs_cos_1 + bc2 * fs_cos_2 + \
             bc3 * fs_cos_3 + btclp * cooling_temp_lp + bthlp * heating_temp_lp + \
             btc * cooling_temp + bth * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    # Fitting

    with complete_pooling:
        approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        complete_pooling_trace = approx.sample(1000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with complete_pooling:
        pm.set_data(
            {"fs_sin_1": daypart_fs_sin_1_test,
             "fs_sin_2": daypart_fs_sin_2_test,
             "fs_sin_3": daypart_fs_sin_3_test,
             "fs_cos_1": daypart_fs_cos_1_test,
             "fs_cos_2": daypart_fs_cos_2_test,
             "fs_cos_3": daypart_fs_cos_3_test,
             "cooling_temp":outdoor_temp_c_test,
             "heating_temp": outdoor_temp_h_test,
             "cooling_temp_lp": outdoor_temp_lp_c_test,
             "heating_temp_lp": outdoor_temp_lp_h_test
             })

        complete_pool_posterior_hdi = pm.sample_posterior_predictive(complete_pooling_trace, keep_size=True)
        complete_pool_posterior = pm.sample_posterior_predictive(complete_pooling_trace)

        complete_pool_prior = pm.sample_prior_predictive(150)

        # Calculate predictions and HDI

    complete_pool_predictions = np.exp(complete_pool_posterior['y'].mean(0))
    complete_pool_hdi_data = az.hdi(complete_pool_posterior_hdi)
    complete_pool_lower_bound = np.array(np.exp(complete_pool_hdi_data.to_array().sel(hdi='lower'))).flatten()
    complete_pool_higher_bound = np.array(np.exp(complete_pool_hdi_data.to_array().sel(hdi='higher'))).flatten()

    complete_pool_gamma =  np.where(test_df.total_electricity > complete_pool_higher_bound,
                      1 + (np.log(test_df.total_electricity - complete_pool_higher_bound)/np.log(test_df.total_electricity)),
                      np.where(test_df.total_electricity < complete_pool_lower_bound,
                               1 + (np.log(complete_pool_lower_bound - test_df.total_electricity)/np.log(test_df.total_electricity)),
                               1))

    complete_pool_adjusted_coverage = 1/len(test_df) * np.sum(complete_pool_gamma * np.log(complete_pool_higher_bound) /np.log(complete_pool_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    complete_pool_mse = mean_squared_error(test_df.total_electricity, complete_pool_predictions)
    complete_pool_rmse = sqrt(complete_pool_mse)
    complete_pool_cvrmse = complete_pool_rmse / test_df.total_electricity.mean()
    complete_pool_coverage = sum((complete_pool_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= complete_pool_higher_bound)) * 100 / len(test_df)
    complete_pool_confidence_length = sum(complete_pool_higher_bound) - sum(complete_pool_lower_bound)

    complete_pool_nmbe = np.sum(test_df.total_electricity - complete_pool_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()


    export_data = {'partial_pooling_cvrmse': [partial_pool_cvrmse], 'no_pooling_cvrmse': [no_pool_cvrmse],
                   'complete_pooling_cvrmse': [complete_pool_cvrmse], 'partial_pooling_coverage': [partial_pool_coverage],
                   'no_pooling_coverage': [no_pool_coverage], 'complete_pooling_coverage': [complete_pool_coverage],
                   'partial_pooling_length':[partial_pool_confidence_length], 'no_pooling_length': [no_pool_confidence_length],
                   'complete_pooling_length': [complete_pool_confidence_length], 'partial_pooling_adj_coverage': [partial_pool_adjusted_coverage],
                   'no_pooling_adj_coverage':[no_pool_adjusted_coverage], 'complete_pooling_adj_coverage':[complete_pool_adjusted_coverage],
                   'partial_pooling_nmbe':[partial_pool_nmbe], 'no_pooling_nmbe':[no_pool_nmbe], 'complete_pooling_nmbe':[complete_pool_nmbe]}

    export_df = pd.DataFrame(data=export_data)
    return export_df


def multiprocessing_bayesian_comparison(df):

    building_id = df.columns[1]
    df.columns = ['t', 'total_electricity', 'outdoor_temp']
    edif = str(building_id)

    # Print consumption and temperature df if it doesn't exist already

    if os.path.isfile("/root/benedetto/results/buildings/" + edif + ".csv"):
        print(edif + ' data was retreived successfully')
    else:
        df.to_csv("/root/benedetto/results/buildings/" + edif + ".csv", index=False)

    # Try to read preprocessed file, otherwise run preprocessing
    try:
        df_preprocessed = pd.read_csv("/root/benedetto/results/buildings/" + edif + "_preprocess.csv")
        print(str(edif) + ' preprocessed data was retrieved succesfully')
    except:
        print('Running preprocessing for ' + str(edif))
        subprocess.run(["Rscript", "ashrae_preprocess_server.R", edif, building_id])
        try:
            df_preprocessed = pd.read_csv("/root/benedetto/results/buildings/" + edif + "_preprocess.csv")
        except:
            print("Preprocessing failed for " + str(edif) + '. Skipping to next building.')

    try:
        dat = pd.read_csv("/root/benedetto/results/bayes_results.csv")
        results_exist = True
    except:
        results_exist = False

    # If model results already exist for the selected building, skip to next
    if results_exist == True & (building_id in dat.values):
        print('Results for ' + building_id + ' are already calculated. Skipping to next building')

    elif results_exist == True & (building_id not in dat.values):
        try:
            model_results = bayesian_model_comparison_whole_year(df_preprocessed)
            model_results['id'] = building_id
            final_export = dat.append(model_results)
            final_export.to_csv("/root/benedetto/results/bayes_results.csv", index=False)
        except:
            print('Modeling error for ' + str(edif) + '. Skipping to the next building')

    else:
        try:
            model_results = bayesian_model_comparison_whole_year(df_preprocessed)
            model_results['id'] = building_id
            final_export = model_results
            final_export.to_csv("/root/benedetto/results/bayes_results.csv", index=False)
        except:
            print('Modeling error for ' + str(edif) + '. Skipping to the next building')


    
