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
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show

def bayesian_model_comparison_test_1 (df, building_id):
    #df = pd.read_csv("/root/benedetto/results/buildings/Crow_education_Keisha_preprocess.csv")
    #df = pd.read_csv("/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Rat_lodging_Marguerite_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1

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
    outdoor_temp_train = train_df.outdoor_temp
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
    outdoor_temp_test = test_df.outdoor_temp
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

    # Model 1 ADVI dep

    with pm.Model(coords=coords) as model_1:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")
        weekday = pm.Data("weekday", weekdays_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Temperature coefficients:

        btc = pm.Uniform("btc", lower = -5, upper = 5, dims="daypart")
        bth = pm.Uniform("bth", lower = -5, upper = 5, dims="daypart")

        # Intercept
        a_cluster = pm.Uniform("a_cluster", lower = -100, upper = 100, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Uniform("bsd1", lower = -5, upper = 5, dims=("profile_cluster"))
        bsd2 = pm.Uniform("bsd2", lower = -5, upper = 5, dims=("profile_cluster"))
        bsd3 = pm.Uniform("bsd3", lower = -5, upper = 5, dims=("profile_cluster"))

        bcd1 = pm.Uniform("bcd1", lower = -5, upper = 5, dims=("profile_cluster"))
        bcd2 = pm.Uniform("bcd2", lower = -5, upper = 5, dims=("profile_cluster"))
        bcd3 = pm.Uniform("bcd3", lower = -5, upper = 5, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = -50, upper = 50)
        tbal_c = pm.Uniform("tbal_c", lower = -50, upper = 50)

        # Dependence
        dep_h = pm.Uniform("dep_h", lower=0, upper=1, dims="profile_cluster")
        dep_c = pm.Uniform("dep_c", lower=0, upper=1, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx] > 0.5) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx] > 0.5)

        # Model error:
        sigma = pm.Uniform("sigma", lower = -5, upper = 5)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

        # Fitting
    with model_1:
        approx_dep = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_1_trace = approx_dep.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_1:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_1_posterior_hdi = pm.sample_posterior_predictive(model_1_trace, keep_size=True)
        model_1_posterior = pm.sample_posterior_predictive(model_1_trace)
        model_1_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_1_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h.png')
    az.plot_trace(model_1_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c.png')
    az.plot_trace(model_1_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth.png')
    az.plot_trace(model_1_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc.png')
    az.plot_trace(model_1_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h.png')
    az.plot_trace(model_1_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c.png')

    # Calculate predictions and HDI

    model_1_predictions = np.exp(model_1_posterior['y'].mean(0))
    hdi_data = az.hdi(model_1_posterior_hdi)
    model_1_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_1_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_1_gamma = np.where(test_df.total_electricity > model_1_higher_bound,
                              1 + ((test_df.total_electricity - model_1_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_1_lower_bound,
                                       1 + ((model_1_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_1_adjusted_coverage = np.nanmean(
        model_1_gamma * (model_1_higher_bound) / (model_1_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_1_mse = mean_squared_error(test_df.total_electricity, model_1_predictions)
    model_1_rmse = sqrt(model_1_mse)
    model_1_cvrmse = model_1_rmse / test_df.total_electricity.mean()
    model_1_coverage = sum((model_1_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_1_higher_bound)) * 100 / len(test_df)
    model_1_confidence_length = sum(model_1_higher_bound) - sum(model_1_lower_bound)

    # Calculate NMBE
    model_1_nmbe = np.sum(test_df.total_electricity - model_1_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_1_data = {'t': test_df['t'],
               'prediction': model_1_predictions,
               'lower_bound': model_1_lower_bound,
               'higher_bound': model_1_higher_bound}

    mod_1_results = pd.DataFrame(data=mod_1_data)
    mod_1_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_1.csv", index=False)

    export_data = {'mod_1_cvrmse': [model_1_cvrmse],
                   'mod_1_adjusted_coverage': [model_1_adjusted_coverage],
                   'mod_1_nmbe': [model_1_nmbe],
                   'mod_1_cov':[model_1_coverage],
                   'id': building_id}

    export_df = pd.DataFrame(data=export_data)
    return export_df

def bayesian_model_comparison_test_2 (df, building_id):
    #df = pd.read_csv("/root/benedetto/results/buildings/Crow_education_Keisha_preprocess.csv")
    #df = pd.read_csv("/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Rat_lodging_Marguerite_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1

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
    outdoor_temp_train = train_df.outdoor_temp
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
    outdoor_temp_test = test_df.outdoor_temp
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

    # Model 1 ADVI dep

    with pm.Model(coords=coords) as model_2:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = 1, dims="daypart")
        bth = pm.HalfNormal("bth", sigma = 1, dims="daypart")

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = 0, sigma = 1, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Uniform("dep_h", lower=0, upper=1, dims="profile_cluster")
        dep_c = pm.Uniform("dep_c", lower=0, upper=1, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx] > 0.5) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx] > 0.5)

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

        # Fitting
    with model_2:
        approx_dep = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_2_trace = approx_dep.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_2:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_2_posterior_hdi = pm.sample_posterior_predictive(model_2_trace, keep_size=True)
        model_2_posterior = pm.sample_posterior_predictive(model_2_trace)
        model_2_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_2_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h.png')
    az.plot_trace(model_2_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c.png')
    az.plot_trace(model_2_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth.png')
    az.plot_trace(model_2_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc.png')
    az.plot_trace(model_2_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h.png')
    az.plot_trace(model_2_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c.png')

    # Calculate predictions and HDI

    # Bokeh plots to compare NUTS and ADVI
    p1 = figure(plot_width=800, plot_height=400, x_axis_type='datetime')
    p1.line(test_df['t'], model_2_predictions, color="navy", alpha=0.8)
    p1.line(test_df['t'], test_df['total_electricity'], color="orange", alpha=0.6)
    p1.varea(x=test_df['t'], y1=model_2_lower_bound, y2=model_2_higher_bound, color='gray', alpha=0.2)
    show(p1)

    model_2_predictions = np.exp(model_2_posterior['y'].mean(0))
    hdi_data = az.hdi(model_2_posterior_hdi)
    model_2_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_2_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_2_gamma = np.where(test_df.total_electricity > model_2_higher_bound,
                              1 + ((test_df.total_electricity - model_2_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_2_lower_bound,
                                       1 + ((model_2_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_2_adjusted_coverage = np.nanmean(
        model_2_gamma * (model_2_higher_bound) / (model_2_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_2_mse = mean_squared_error(test_df.total_electricity, model_2_predictions)
    model_2_rmse = sqrt(model_2_mse)
    model_2_cvrmse = model_2_rmse / test_df.total_electricity.mean()
    model_2_coverage = sum((model_2_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_2_higher_bound)) * 100 / len(test_df)
    model_2_confidence_length = sum(model_2_higher_bound) - sum(model_2_lower_bound)

    # Calculate NMBE
    model_2_nmbe = np.sum(test_df.total_electricity - model_2_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_2_data = {'t': test_df['t'],
               'prediction': model_2_predictions,
               'lower_bound': model_2_lower_bound,
               'higher_bound': model_2_higher_bound}

    mod_2_results = pd.DataFrame(data=mod_2_data)
    mod_2_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_2.csv", index=False)

    export_data = {'mod_2_cvrmse': [model_2_cvrmse],
                   'mod_2_adjusted_coverage': [model_2_adjusted_coverage],
                   'mod_2_nmbe': [model_2_nmbe],
                   'id': building_id}

    export_df = pd.DataFrame(data=export_data)
    return export_df

def bayesian_model_comparison_test_3 (df, building_id):
    #df = pd.read_csv("/root/benedetto/results/buildings/Crow_education_Keisha_preprocess.csv")
    #df = pd.read_csv("/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Fox_education_Ollie_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1

    # Create training and test set (for the ashrae data training is 2016, test is 2017)
    train_df = df.loc[df["t"] <= pd.to_datetime("2017-01-01")]
    test_df = df.loc[df["t"] > pd.to_datetime("2017-01-01")]

    # Define Training variables
    total_electricity_train = train_df.total_electricity.values
    log_electricity_train = train_df['log_v']
    clusters_train = train_df.s
    unique_clusters = clusters_train.unique()
    dayparts_train = train_df.daypart
    unique_dayparts = dayparts_train.unique()
    outdoor_temp_train = train_df.outdoor_temp
    daypart_fs_sin_1_train = train_df.daypart_fs_sin_1
    daypart_fs_sin_2_train = train_df.daypart_fs_sin_2
    daypart_fs_sin_3_train = train_df.daypart_fs_sin_3
    daypart_fs_cos_1_train = train_df.daypart_fs_cos_1
    daypart_fs_cos_2_train = train_df.daypart_fs_cos_2
    daypart_fs_cos_3_train = train_df.daypart_fs_cos_3
    # yearpart_fs_sin_1_train = train_df.yearpart_fs_sin_1
    # yearpart_fs_sin_2_train = train_df.yearpart_fs_sin_2
    # yearpart_fs_sin_3_train = train_df.yearpart_fs_sin_3
    # yearpart_fs_cos_1_train = train_df.yearpart_fs_cos_1
    # yearpart_fs_cos_2_train = train_df.yearpart_fs_cos_2
    # yearpart_fs_cos_3_train = train_df.yearpart_fs_cos_3
    windspeed_train = train_df.windSpeed

    # Define test variables
    clusters_test = test_df.s
    dayparts_test = test_df.daypart
    outdoor_temp_test = test_df.outdoor_temp

    daypart_fs_sin_1_test = test_df.daypart_fs_sin_1
    daypart_fs_sin_2_test = test_df.daypart_fs_sin_2
    daypart_fs_sin_3_test = test_df.daypart_fs_sin_3
    daypart_fs_cos_1_test = test_df.daypart_fs_cos_1
    daypart_fs_cos_2_test = test_df.daypart_fs_cos_2
    daypart_fs_cos_3_test = test_df.daypart_fs_cos_3
    # yearpart_fs_sin_1_test = test_df.yearpart_fs_sin_1
    # yearpart_fs_sin_2_test = test_df.yearpart_fs_sin_2
    # yearpart_fs_sin_3_test = test_df.yearpart_fs_sin_3
    # yearpart_fs_cos_1_test = test_df.yearpart_fs_cos_1
    # yearpart_fs_cos_2_test = test_df.yearpart_fs_cos_2
    # yearpart_fs_cos_3_test = test_df.yearpart_fs_cos_3
    windspeed_test = test_df.windSpeed

    # create coords for pymc3
    coords = {"obs_id": np.arange(total_electricity_train.size)}
    coords["profile_cluster"] = unique_clusters
    coords["daypart"] = unique_dayparts

    # Model 1 ADVI dep

    with pm.Model(coords=coords) as model_3:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        # yp_fs_sin_1 = pm.Data("yp_fs_sin_1", yearpart_fs_sin_1_train, dims="obs_id")
        # yp_fs_sin_2 = pm.Data("yp_fs_sin_2", yearpart_fs_sin_2_train, dims="obs_id")
        # yp_fs_sin_3 = pm.Data("yp_fs_sin_3", yearpart_fs_sin_3_train, dims="obs_id")
        #
        # yp_fs_cos_1 = pm.Data("yp_fs_cos_1", yearpart_fs_cos_1_train, dims="obs_id")
        # yp_fs_cos_2 = pm.Data("yp_fs_cos_2", yearpart_fs_cos_2_train, dims="obs_id")
        # yp_fs_cos_3 = pm.Data("yp_fs_cos_3", yearpart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")
        windspeed = pm.Data("windspeed", windspeed_train, dims = "obs_id")
        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = 1, dims="daypart")
        bth = pm.HalfNormal("bth", sigma = 1, dims="daypart")
        bws = pm.HalfNormal("bws", sigma = 1, dims = "daypart")

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = 0, sigma = 1, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        # bsy1 = pm.Normal("bsy1", mu=0, sigma=1, dims=("profile_cluster"))
        # bsy2 = pm.Normal("bsy2", mu=0, sigma=1, dims=("profile_cluster"))
        # bsy3 = pm.Normal("bsy3", mu=0, sigma=1, dims=("profile_cluster"))
        #
        # bcy1 = pm.Normal("bcy1", mu=0, sigma=1, dims=("profile_cluster"))
        # bcy2 = pm.Normal("bcy2", mu=0, sigma=1, dims=("profile_cluster"))
        # bcy3 = pm.Normal("bcy3", mu=0, sigma=1, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Uniform("dep_h", lower=0, upper=1, dims="profile_cluster")
        dep_c = pm.Uniform("dep_c", lower=0, upper=1, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx] > 0.5) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx] > 0.5) + bws[daypart] * windspeed

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

        # Fitting
    with model_3:
        approx_dep = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_3_trace = approx_dep.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_3:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     # "yp_fs_sin_1": yearpart_fs_sin_1_test,
                     # "yp_fs_sin_2": yearpart_fs_sin_2_test,
                     # "yp_fs_sin_3": yearpart_fs_sin_3_test,
                     # "yp_fs_cos_1": yearpart_fs_cos_1_test,
                     # "yp_fs_cos_2": yearpart_fs_cos_2_test,
                     # "yp_fs_cos_3": yearpart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test,
                     "windspeed": windspeed_test
                     })

        model_3_posterior_hdi = pm.sample_posterior_predictive(model_3_trace, keep_size=True)
        model_3_posterior = pm.sample_posterior_predictive(model_3_trace)
        model_3_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_3_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h.png')
    az.plot_trace(model_3_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c.png')
    az.plot_trace(model_3_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth.png')
    az.plot_trace(model_3_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc.png')
    az.plot_trace(model_3_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h.png')
    az.plot_trace(model_3_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c.png')

    # Calculate predictions and HDI

    model_3_predictions = np.exp(model_3_posterior['y'].mean(0))
    hdi_data = az.hdi(model_3_posterior_hdi)
    model_3_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_3_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_3_gamma = np.where(test_df.total_electricity > model_3_higher_bound,
                              1 + ((test_df.total_electricity - model_3_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_3_lower_bound,
                                       1 + ((model_3_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_3_adjusted_coverage = np.nanmean(
        model_3_gamma * (model_3_higher_bound) / (model_3_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_3_mse = mean_squared_error(test_df.total_electricity, model_3_predictions)
    model_3_rmse = sqrt(model_3_mse)
    model_3_cvrmse = model_3_rmse / test_df.total_electricity.mean()
    model_3_coverage = sum((model_3_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_3_higher_bound)) * 100 / len(test_df)
    model_3_confidence_length = sum(model_3_higher_bound) - sum(model_3_lower_bound)

    # Calculate NMBE
    model_3_nmbe = np.sum(test_df.total_electricity - model_3_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_3_data = {'t': test_df['t'],
               'prediction': model_3_predictions,
               'lower_bound': model_3_lower_bound,
               'higher_bound': model_3_higher_bound,
               'total_electricity':test_df['total_electricity']}

    mod_3_results = pd.DataFrame(data=mod_3_data)
    mod_3_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_3.csv", index=False)

    export_data = {'mod_3_cvrmse': [model_3_cvrmse],
                   'mod_3_adjusted_coverage': [model_3_adjusted_coverage],
                   'mod_3_nmbe': [model_3_nmbe],
                   'mod_3_coverage': [model_3_coverage],
                   'id': building_id}

    export_df = pd.DataFrame(data=export_data)
    return export_df

def bayesian_model_comparison_test_4 (df, building_id):
    #df = pd.read_csv("/root/benedetto/results/buildings/Crow_education_Keisha_preprocess.csv")
    #df = pd.read_csv("/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Fox_education_Ollie_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1

    # Create training and test set (for the ashrae data training is 2016, test is 2017)
    train_df = df.loc[df["t"] <= pd.to_datetime("2017-01-01")]
    test_df = df.loc[df["t"] > pd.to_datetime("2017-01-01")]

    # Define Training variables
    total_electricity_train = train_df.total_electricity.values
    log_electricity_train = train_df['log_v']
    clusters_train = train_df.s
    unique_clusters = clusters_train.unique()
    dayparts_train = train_df.daypart
    unique_dayparts = dayparts_train.unique()
    outdoor_temp_train = train_df.outdoor_temp
    daypart_fs_sin_1_train = train_df.daypart_fs_sin_1
    daypart_fs_sin_2_train = train_df.daypart_fs_sin_2
    daypart_fs_sin_3_train = train_df.daypart_fs_sin_3
    daypart_fs_cos_1_train = train_df.daypart_fs_cos_1
    daypart_fs_cos_2_train = train_df.daypart_fs_cos_2
    daypart_fs_cos_3_train = train_df.daypart_fs_cos_3

    # Define test variables
    clusters_test = test_df.s
    dayparts_test = test_df.daypart
    outdoor_temp_test = test_df.outdoor_temp

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

    with pm.Model(coords=coords) as model_4_np_advi:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = 1, dims="daypart")
        bth = pm.HalfNormal("bth", sigma = 1, dims="daypart")

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = 0, sigma = 1, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Uniform("dep_h", lower=0, upper=1, dims="profile_cluster")
        dep_c = pm.Uniform("dep_c", lower=0, upper=1, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx] > 0.5) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx] > 0.5)

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    with model_4_np_advi:
        model_4_np_advi_approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_4_np_advi_trace = model_4_np_advi_approx.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_4_np_advi:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_4_np_advi_posterior_hdi = pm.sample_posterior_predictive(model_4_np_advi_trace, keep_size=True)
        model_4_np_advi_posterior = pm.sample_posterior_predictive(model_4_np_advi_trace)
        # model_4_np_advi_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_4_np_advi_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_np_advi.png')
    az.plot_trace(model_4_np_advi_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_np_advi.png')
    az.plot_trace(model_4_np_advi_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_np_advi.png')
    az.plot_trace(model_4_np_advi_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_np_advi.png')
    az.plot_trace(model_4_np_advi_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_np_advi.png')
    az.plot_trace(model_4_np_advi_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_np_advi.png')

    # Calculate predictions and HDI

    model_4_np_advi_predictions = np.exp(model_4_np_advi_posterior['y'].mean(0))
    hdi_data = az.hdi(model_4_np_advi_posterior_hdi)
    model_4_np_advi_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_4_np_advi_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_4_np_advi_gamma = np.where(test_df.total_electricity > model_4_np_advi_higher_bound,
                              1 + ((test_df.total_electricity - model_4_np_advi_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_4_np_advi_lower_bound,
                                       1 + ((model_4_np_advi_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_4_np_advi_adjusted_coverage = np.nanmean(
        model_4_np_advi_gamma * (model_4_np_advi_higher_bound) / (model_4_np_advi_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_4_np_advi_mse = mean_squared_error(test_df.total_electricity, model_4_np_advi_predictions)
    model_4_np_advi_rmse = sqrt(model_4_np_advi_mse)
    model_4_np_advi_cvrmse = model_4_np_advi_rmse / test_df.total_electricity.mean()
    model_4_np_advi_coverage = sum((model_4_np_advi_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_4_np_advi_higher_bound)) * 100 / len(test_df)
    model_4_np_advi_confidence_length = sum(model_4_np_advi_higher_bound) - sum(model_4_np_advi_lower_bound)

    # Calculate NMBE
    model_4_np_advi_nmbe = np.sum(test_df.total_electricity - model_4_np_advi_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_4_np_advi_data = {'t': test_df['t'],
               'prediction': model_4_np_advi_predictions,
               'lower_bound': model_4_np_advi_lower_bound,
               'higher_bound': model_4_np_advi_higher_bound}

    mod_4_np_advi_results = pd.DataFrame(data=mod_4_np_advi_data)
    mod_4_np_advi_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_4_np_advi.csv", index=False)

    with pm.Model(coords=coords) as model_4_pp_advi:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Hyperpriors

        bf = pm.Normal("bf", mu=0.0, sigma=1.0)
        sigma_bf = pm.Exponential("sigma_bf", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)
        sigma_btc = pm.Exponential("sigma_btc", 1.0)
        sigma_bth = pm.Exponential("sigma_bth", 1.0)

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = sigma_btc, dims="daypart")
        bth = pm.HalfNormal("bth", sigma = sigma_bth, dims="daypart")

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = a, sigma = sigma_a, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Uniform("dep_h", lower=0, upper=1, dims="profile_cluster")
        dep_c = pm.Uniform("dep_c", lower=0, upper=1, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx] > 0.5) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx] > 0.5)

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    with model_4_pp_advi:
        model_4_pp_advi_approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_4_pp_advi_trace = model_4_pp_advi_approx.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_4_pp_advi:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_4_pp_advi_posterior_hdi = pm.sample_posterior_predictive(model_4_pp_advi_trace, keep_size=True)
        model_4_pp_advi_posterior = pm.sample_posterior_predictive(model_4_pp_advi_trace)
        # model_4_pp_advi_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_4_pp_advi_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_pp_advi.png')
    az.plot_trace(model_4_pp_advi_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_pp_advi.png')
    az.plot_trace(model_4_pp_advi_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_pp_advi.png')
    az.plot_trace(model_4_pp_advi_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_pp_advi.png')
    az.plot_trace(model_4_pp_advi_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_pp_advi.png')
    az.plot_trace(model_4_pp_advi_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_pp_advi.png')

    # Calculate predictions and HDI

    model_4_pp_advi_predictions = np.exp(model_4_pp_advi_posterior['y'].mean(0))
    hdi_data = az.hdi(model_4_pp_advi_posterior_hdi)
    model_4_pp_advi_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_4_pp_advi_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_4_pp_advi_gamma = np.where(test_df.total_electricity > model_4_pp_advi_higher_bound,
                              1 + ((test_df.total_electricity - model_4_pp_advi_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_4_pp_advi_lower_bound,
                                       1 + ((model_4_pp_advi_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_4_pp_advi_adjusted_coverage = np.nanmean(
        model_4_pp_advi_gamma * (model_4_pp_advi_higher_bound) / (model_4_pp_advi_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_4_pp_advi_mse = mean_squared_error(test_df.total_electricity, model_4_pp_advi_predictions)
    model_4_pp_advi_rmse = sqrt(model_4_pp_advi_mse)
    model_4_pp_advi_cvrmse = model_4_pp_advi_rmse / test_df.total_electricity.mean()
    model_4_pp_advi_coverage = sum((model_4_pp_advi_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_4_pp_advi_higher_bound)) * 100 / len(test_df)
    model_4_pp_advi_confidence_length = sum(model_4_pp_advi_higher_bound) - sum(model_4_pp_advi_lower_bound)

    # Calculate NMBE
    model_4_pp_advi_nmbe = np.sum(test_df.total_electricity - model_4_pp_advi_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_4_pp_advi_data = {'t': test_df['t'],
               'prediction': model_4_pp_advi_predictions,
               'lower_bound': model_4_pp_advi_lower_bound,
               'higher_bound': model_4_pp_advi_higher_bound}

    mod_4_pp_advi_results = pd.DataFrame(data=mod_4_pp_advi_data)
    mod_4_pp_advi_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_4_pp_advi.csv", index=False)

    with pm.Model(coords=coords) as model_4_cp_advi:

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = 1)
        bth = pm.HalfNormal("bth", sigma = 1)

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = 0, sigma = 1)

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = 0, sigma = 1)
        bsd2 = pm.Normal("bsd2", mu = 0, sigma = 1)
        bsd3 = pm.Normal("bsd3",  mu = 0, sigma = 1)

        bcd1 = pm.Normal("bcd1",  mu = 0, sigma = 1)
        bcd2 = pm.Normal("bcd2",  mu = 0, sigma = 1)
        bcd3 = pm.Normal("bcd3",  mu = 0, sigma = 1)

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Uniform("dep_h", lower=0, upper=1)
        dep_c = pm.Uniform("dep_c", lower=0, upper=1)

        # Expected value per county:
        mu = a_cluster + bsd1 * dp_fs_sin_1 + \
             bsd2 * dp_fs_sin_2 + bsd3 * dp_fs_sin_3 + \
             bcd1 * dp_fs_cos_1 + bcd2 * dp_fs_cos_2 + \
             bcd3 * dp_fs_cos_3 + \
             btc * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c > 0.5) + \
             bth * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h > 0.5)

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    with model_4_cp_advi:
        model_4_cp_advi_approx = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        model_4_cp_advi_trace = model_4_cp_advi_approx.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_4_cp_advi:
        pm.set_data({"dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_4_cp_advi_posterior_hdi = pm.sample_posterior_predictive(model_4_cp_advi_trace, keep_size=True)
        model_4_cp_advi_posterior = pm.sample_posterior_predictive(model_4_cp_advi_trace)
        # model_4_cp_advi_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_4_cp_advi_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_cp_advi.png')
    az.plot_trace(model_4_cp_advi_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_cp_advi.png')
    az.plot_trace(model_4_cp_advi_trace['bth'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_cp_advi.png')
    az.plot_trace(model_4_cp_advi_trace['btc'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_cp_advi.png')
    az.plot_trace(model_4_cp_advi_trace['dep_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_cp_advi.png')
    az.plot_trace(model_4_cp_advi_trace['dep_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_cp_advi.png')

    # Calculate predictions and HDI

    model_4_cp_advi_predictions = np.exp(model_4_cp_advi_posterior['y'].mean(0))
    hdi_data = az.hdi(model_4_cp_advi_posterior_hdi)
    model_4_cp_advi_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_4_cp_advi_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_4_cp_advi_gamma = np.where(test_df.total_electricity > model_4_cp_advi_higher_bound,
                              1 + ((test_df.total_electricity - model_4_cp_advi_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_4_cp_advi_lower_bound,
                                       1 + ((model_4_cp_advi_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_4_cp_advi_adjusted_coverage = np.nanmean(
        model_4_cp_advi_gamma * (model_4_cp_advi_higher_bound) / (model_4_cp_advi_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_4_cp_advi_mse = mean_squared_error(test_df.total_electricity, model_4_cp_advi_predictions)
    model_4_cp_advi_rmse = sqrt(model_4_cp_advi_mse)
    model_4_cp_advi_cvrmse = model_4_cp_advi_rmse / test_df.total_electricity.mean()
    model_4_cp_advi_coverage = sum((model_4_cp_advi_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_4_cp_advi_higher_bound)) * 100 / len(test_df)
    model_4_cp_advi_confidence_length = sum(model_4_cp_advi_higher_bound) - sum(model_4_cp_advi_lower_bound)

    # Calculate NMBE
    model_4_cp_advi_nmbe = np.sum(test_df.total_electricity - model_4_cp_advi_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_4_cp_advi_data = {'t': test_df['t'],
               'prediction': model_4_cp_advi_predictions,
               'lower_bound': model_4_cp_advi_lower_bound,
               'higher_bound': model_4_cp_advi_higher_bound}

    mod_4_cp_advi_results = pd.DataFrame(data=mod_4_cp_advi_data)
    mod_4_cp_advi_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_4_cp_advi.csv", index=False)

    with pm.Model(coords=coords) as model_4_np_nuts:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = 1, dims="daypart")
        bth = pm.HalfNormal("bth", sigma = 1, dims="daypart")

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = 0, sigma = 1, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu = 0, sigma = 1, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2",  mu = 0, sigma = 1, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3",  mu = 0, sigma = 1, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Bernoulli("dep_h", p=0.5, dims="profile_cluster")
        dep_c = pm.Bernoulli("dep_c", p=0.5, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx]) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx])

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    with model_4_np_nuts:
        model_4_np_nuts_trace = pm.sample(5000, tune = 2000, chains = 4, cores = 1, nuts={'target_accept':0.95})

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_4_np_nuts:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_4_np_nuts_posterior_hdi = pm.sample_posterior_predictive(model_4_np_nuts_trace, keep_size=True)
        model_4_np_nuts_posterior = pm.sample_posterior_predictive(model_4_np_nuts_trace)
        # model_4_np_nuts_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_4_np_nuts_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_np_nuts.png')
    az.plot_trace(model_4_np_nuts_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_np_nuts.png')
    az.plot_trace(model_4_np_nuts_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_np_nuts.png')
    az.plot_trace(model_4_np_nuts_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_np_nuts.png')
    az.plot_trace(model_4_np_nuts_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_np_nuts.png')
    az.plot_trace(model_4_np_nuts_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_np_nuts.png')

    # Calculate predictions and HDI

    model_4_np_nuts_predictions = np.exp(model_4_np_nuts_posterior['y'].mean(0))
    hdi_data = az.hdi(model_4_np_nuts_posterior_hdi)
    model_4_np_nuts_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_4_np_nuts_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_4_np_nuts_gamma = np.where(test_df.total_electricity > model_4_np_nuts_higher_bound,
                              1 + ((test_df.total_electricity - model_4_np_nuts_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_4_np_nuts_lower_bound,
                                       1 + ((model_4_np_nuts_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_4_np_nuts_adjusted_coverage = np.nanmean(
        model_4_np_nuts_gamma * (model_4_np_nuts_higher_bound) / (model_4_np_nuts_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_4_np_nuts_mse = mean_squared_error(test_df.total_electricity, model_4_np_nuts_predictions)
    model_4_np_nuts_rmse = sqrt(model_4_np_nuts_mse)
    model_4_np_nuts_cvrmse = model_4_np_nuts_rmse / test_df.total_electricity.mean()
    model_4_np_nuts_coverage = sum((model_4_np_nuts_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_4_np_nuts_higher_bound)) * 100 / len(test_df)
    model_4_np_nuts_confidence_length = sum(model_4_np_nuts_higher_bound) - sum(model_4_np_nuts_lower_bound)

    # Calculate NMBE
    model_4_np_nuts_nmbe = np.sum(test_df.total_electricity - model_4_np_nuts_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_4_np_nuts_data = {'t': test_df['t'],
               'prediction': model_4_np_nuts_predictions,
               'lower_bound': model_4_np_nuts_lower_bound,
               'higher_bound': model_4_np_nuts_higher_bound}

    mod_4_np_nuts_results = pd.DataFrame(data=mod_4_np_nuts_data)
    mod_4_np_nuts_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_4_np_nuts.csv", index=False)


    with pm.Model(coords=coords) as model_4_pp_nuts:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Hyperpriors

        bf = pm.Normal("bf", mu=0.0, sigma=1.0)
        sigma_bf = pm.Exponential("sigma_bf", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)
        sigma_btc = pm.Exponential("sigma_btc", 1.0)
        sigma_bth = pm.Exponential("sigma_bth", 1.0)

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = sigma_btc, dims="daypart")
        bth = pm.HalfNormal("bth", sigma = sigma_bth, dims="daypart")

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = a, sigma = sigma_a, dims=("profile_cluster"))

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3",  mu = bf, sigma = sigma_bf, dims=("profile_cluster"))

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Bernoulli("dep_h", p=0.5, dims="profile_cluster")
        dep_c = pm.Bernoulli("dep_c", p=0.5, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx]) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx])

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    with model_4_pp_nuts:
        model_4_pp_nuts_trace = pm.sample(5000, tune = 2000, chains = 4, cores = 1, nuts={'target_accept':0.95})

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_4_pp_nuts:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_4_pp_nuts_posterior_hdi = pm.sample_posterior_predictive(model_4_pp_nuts_trace, keep_size=True)
        model_4_pp_nuts_posterior = pm.sample_posterior_predictive(model_4_pp_nuts_trace)
        # model_4_pp_nuts_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_4_pp_nuts_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_pp_nuts.png')
    az.plot_trace(model_4_pp_nuts_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_nuts.png')
    az.plot_trace(model_4_pp_nuts_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_pp_nuts.png')
    az.plot_trace(model_4_pp_nuts_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_pp_nuts.png')
    az.plot_trace(model_4_pp_nuts_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_pp_nuts.png')
    az.plot_trace(model_4_pp_nuts_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_pp_nuts.png')

    # Calculate predictions and HDI

    model_4_pp_nuts_predictions = np.exp(model_4_pp_nuts_posterior['y'].mean(0))
    hdi_data = az.hdi(model_4_pp_nuts_posterior_hdi)
    model_4_pp_nuts_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_4_pp_nuts_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_4_pp_nuts_gamma = np.where(test_df.total_electricity > model_4_pp_nuts_higher_bound,
                              1 + ((test_df.total_electricity - model_4_pp_nuts_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_4_pp_nuts_lower_bound,
                                       1 + ((model_4_pp_nuts_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_4_pp_nuts_adjusted_coverage = np.nanmean(
        model_4_pp_nuts_gamma * (model_4_pp_nuts_higher_bound) / (model_4_pp_nuts_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_4_pp_nuts_mse = mean_squared_error(test_df.total_electricity, model_4_pp_nuts_predictions)
    model_4_pp_nuts_rmse = sqrt(model_4_pp_nuts_mse)
    model_4_pp_nuts_cvrmse = model_4_pp_nuts_rmse / test_df.total_electricity.mean()
    model_4_pp_nuts_coverage = sum((model_4_pp_nuts_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_4_pp_nuts_higher_bound)) * 100 / len(test_df)
    model_4_pp_nuts_confidence_length = sum(model_4_pp_nuts_higher_bound) - sum(model_4_pp_nuts_lower_bound)

    # Calculate NMBE
    model_4_pp_nuts_nmbe = np.sum(test_df.total_electricity - model_4_pp_nuts_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_4_pp_nuts_data = {'t': test_df['t'],
               'prediction': model_4_pp_nuts_predictions,
               'lower_bound': model_4_pp_nuts_lower_bound,
               'higher_bound': model_4_pp_nuts_higher_bound}

    mod_4_pp_nuts_results = pd.DataFrame(data=mod_4_pp_nuts_data)
    mod_4_pp_nuts_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_4_pp_nuts.csv", index=False)

    with pm.Model(coords=coords) as model_4_cp_nuts:

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Temperature coefficients:

        btc = pm.HalfNormal("btc", sigma = 1)
        bth = pm.HalfNormal("bth", sigma = 1)

        # Intercept
        a_cluster = pm.Normal("a_cluster", mu = 0, sigma = 1)

        # Fourier slopes:
        bsd1 = pm.Normal("bsd1", mu = 0, sigma = 1)
        bsd2 = pm.Normal("bsd2", mu = 0, sigma = 1)
        bsd3 = pm.Normal("bsd3",  mu = 0, sigma = 1)

        bcd1 = pm.Normal("bcd1",  mu = 0, sigma = 1)
        bcd2 = pm.Normal("bcd2",  mu = 0, sigma = 1)
        bcd3 = pm.Normal("bcd3",  mu = 0, sigma = 1)

        # Balance temperatures
        tbal_h = pm.Uniform("tbal_h", lower = 8, upper = 30)
        tbal_c = pm.Uniform("tbal_c", lower = 8, upper = 30)

        # Dependence
        dep_h = pm.Bernoulli("dep_h", p=0.5)
        dep_c = pm.Bernoulli("dep_c", p=0.5)

        # Expected value per county:
        mu = a_cluster + bsd1 * dp_fs_sin_1 + \
             bsd2 * dp_fs_sin_2 + bsd3 * dp_fs_sin_3 + \
             bcd1 * dp_fs_cos_1 + bcd2 * dp_fs_cos_2 + \
             bcd3 * dp_fs_cos_3 + \
             btc * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (dep_c) + \
             bth * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (dep_h)

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    with model_4_cp_nuts:
        model_4_cp_nuts_trace = pm.sample(5000, tune = 2000, chains = 4, cores = 1, nuts={'target_accept':0.95})

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with model_4_cp_nuts:
        pm.set_data({"dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        model_4_cp_nuts_posterior_hdi = pm.sample_posterior_predictive(model_4_cp_nuts_trace, keep_size=True)
        model_4_cp_nuts_posterior = pm.sample_posterior_predictive(model_4_cp_nuts_trace)
        # model_4_cp_nuts_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(model_4_cp_nuts_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_cp_nuts.png')
    az.plot_trace(model_4_cp_nuts_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_cp_nuts.png')
    az.plot_trace(model_4_cp_nuts_trace['bth'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_cp_nuts.png')
    az.plot_trace(model_4_cp_nuts_trace['btc'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_cp_nuts.png')
    az.plot_trace(model_4_cp_nuts_trace['dep_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_cp_nuts.png')
    az.plot_trace(model_4_cp_nuts_trace['dep_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_cp_advi.png')

    # Calculate predictions and HDI

    model_4_cp_nuts_predictions = np.exp(model_4_cp_nuts_posterior['y'].mean(0))
    hdi_data = az.hdi(model_4_cp_nuts_posterior_hdi)
    model_4_cp_nuts_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    model_4_cp_nuts_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    model_4_cp_nuts_gamma = np.where(test_df.total_electricity > model_4_cp_nuts_higher_bound,
                              1 + ((test_df.total_electricity - model_4_cp_nuts_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < model_4_cp_nuts_lower_bound,
                                       1 + ((model_4_cp_nuts_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    model_4_cp_nuts_adjusted_coverage = np.nanmean(
        model_4_cp_nuts_gamma * (model_4_cp_nuts_higher_bound) / (model_4_cp_nuts_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    model_4_cp_nuts_mse = mean_squared_error(test_df.total_electricity, model_4_cp_nuts_predictions)
    model_4_cp_nuts_rmse = sqrt(model_4_cp_nuts_mse)
    model_4_cp_nuts_cvrmse = model_4_cp_nuts_rmse / test_df.total_electricity.mean()
    model_4_cp_nuts_coverage = sum((model_4_cp_nuts_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= model_4_cp_nuts_higher_bound)) * 100 / len(test_df)
    model_4_cp_nuts_confidence_length = sum(model_4_cp_nuts_higher_bound) - sum(model_4_cp_nuts_lower_bound)

    # Calculate NMBE
    model_4_cp_nuts_nmbe = np.sum(test_df.total_electricity - model_4_cp_nuts_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print df
    mod_4_cp_nuts_data = {'t': test_df['t'],
               'prediction': model_4_cp_nuts_predictions,
               'lower_bound': model_4_cp_nuts_lower_bound,
               'higher_bound': model_4_cp_nuts_higher_bound}

    mod_4_cp_nuts_results = pd.DataFrame(data=mod_4_cp_nuts_data)
    mod_4_cp_nuts_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_mod_4_cp_nuts.csv", index=False)

    export_data = {'mod_4_np_advi_cvrmse': [model_4_np_advi_cvrmse],
                   'mod_4_pp_advi_cvrmse': [model_4_pp_advi_cvrmse],
                   'mod_4_cp_advi_cvrmse': [model_4_cp_advi_cvrmse],
                   'mod_4_np_nuts_cvrmse': [model_4_np_nuts_cvrmse],
                   'mod_4_pp_nuts_cvrmse': [model_4_pp_nuts_cvrmse],
                   'mod_4_cp_nuts_cvrmse': [model_4_cp_nuts_cvrmse],
                   'mod_4_np_advi_adjusted_coverage': [model_4_np_advi_adjusted_coverage],
                   'mod_4_pp_advi_adjusted_coverage': [model_4_pp_advi_adjusted_coverage],
                   'mod_4_cp_advi_adjusted_coverage': [model_4_cp_advi_adjusted_coverage],
                   'mod_4_np_nuts_adjusted_coverage': [model_4_np_nuts_adjusted_coverage],
                   'mod_4_pp_nuts_adjusted_coverage': [model_4_pp_nuts_adjusted_coverage],
                   'mod_4_cp_nuts_adjusted_coverage': [model_4_cp_nuts_adjusted_coverage],
                   'mod_4_np_advi_nmbe': [model_4_np_advi_nmbe],
                   'mod_4_pp_advi_nmbe': [model_4_pp_advi_nmbe],
                   'mod_4_cp_advi_nmbe': [model_4_cp_advi_nmbe],
                   'mod_4_np_nuts_nmbe': [model_4_np_nuts_nmbe],
                   'mod_4_pp_nuts_nmbe': [model_4_pp_nuts_nmbe],
                   'mod_4_cp_nuts_nmbe': [model_4_cp_nuts_nmbe],
                   'id': building_id}

    export_df = pd.DataFrame(data=export_data)
    return export_df

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

def bayesian_model_comparison_whole_year (df, building_id):
    #df = pd.read_csv("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Fox_education_Gloria_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1
    #df.s = df['s'].astype(object)
    #df.daypart = df['daypart'].astype(object)
    df.weekday = df.weekday - 1
    #df.weekday = df['weekday'].astype(object)

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
    outdoor_temp_train = train_df.outdoor_temp
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
    yearpart_fs_sin_1_train = train_df.yearpart_fs_sin_1
    yearpart_fs_sin_2_train = train_df.yearpart_fs_sin_2
    yearpart_fs_sin_3_train = train_df.yearpart_fs_sin_3
    yearpart_fs_cos_1_train = train_df.yearpart_fs_cos_1
    yearpart_fs_cos_2_train = train_df.yearpart_fs_cos_2
    yearpart_fs_cos_3_train = train_df.yearpart_fs_cos_3

    # Define test variables
    clusters_test = test_df.s
    dayparts_test = test_df.daypart
    weekdays_test = test_df.weekday
    outdoor_temp_test = test_df.outdoor_temp
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
    yearpart_fs_sin_1_test = test_df.yearpart_fs_sin_1
    yearpart_fs_sin_2_test = test_df.yearpart_fs_sin_2
    yearpart_fs_sin_3_test = test_df.yearpart_fs_sin_3
    yearpart_fs_cos_1_test = test_df.yearpart_fs_cos_1
    yearpart_fs_cos_2_test = test_df.yearpart_fs_cos_2
    yearpart_fs_cos_3_test = test_df.yearpart_fs_cos_3

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
        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims = "obs_id")

        # Hyperpriors:
        BoundNormal = pm.Bound(pm.Normal, lower=0.0, upper = 1.0)
        mu_btc = pm.Normal("mu_btc", mu=0.0, sigma=1.0)
        sigma_btc = pm.Exponential("sigma_btc", 1.0)
        mu_bth = pm.Normal("mu_bth", mu=0.0, sigma=1.0)
        sigma_bth = pm.Exponential("sigma_bth", 1.0)
        mu_btclp = pm.Normal("mu_btclp", mu=0.0, sigma=1.0)
        sigma_btclp = pm.Exponential("sigma_btclp", 1.0)
        mu_bthlp = pm.Normal("mu_bthlp", mu=0.0, sigma=1.0)
        sigma_bthlp = pm.Exponential("sigma_bthlp", 1.0)
        bf = pm.Normal("bf", mu=0.0, sigma=1.0)
        sigma_bf = pm.Exponential("sigma_bf", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)
        # k = pm.Gamma('k', alpha=15, beta=15)
        # lambda_ = pm.Gamma('lambda_', alpha=15, beta=15)
        # bounded_laplacian = pm.Bound(pm.Laplace, lower=0, upper=5)

        # btc = pm.Uniform("btc", lower=0, upper = 5, dims="daypart")
        # bth = pm.Uniform("bth", lower=0, upper = 5, dims="daypart")
        # btclp = pm.Uniform("btclp", lower=0, upper = 5, dims="daypart")
        # bthlp = pm.Uniform("bthlp", lower=0, upper = 5, dims="daypart")

        btc = pm.Normal("btc", mu=mu_btc, sigma=sigma_btc, dims="profile_cluster")
        bth = pm.Normal("bth", mu=mu_bth, sigma=sigma_bth, dims="profile_cluster")
        btclp = pm.Normal("btclp", mu=mu_btclp, sigma=sigma_btclp, dims="profile_cluster")
        bthlp = pm.Normal("bthlp", mu=mu_bthlp, sigma=sigma_bthlp, dims="profile_cluster")

        # btc = pm.Weibull('btc', alpha=k, beta=lambda_, dims="daypart")
        # bth = pm.Weibull('bth', alpha=k, beta=lambda_, dims="daypart")
        # btclp = pm.Weibull('btclp', alpha=k, beta=lambda_, dims="daypart")
        # bthlp = pm.Weibull('bthlp', alpha=k, beta=lambda_, dims="daypart")

        # btc = bounded_laplacian('btc', mu=0, b=4, dims = "daypart")
        # bth = bounded_laplacian('bth', mu=0, b=4, dims="daypart")
        # btclp = bounded_laplacian('btclp', mu=0, b=4, dims = "daypart")
        # bthlp = bounded_laplacian('bthlp', mu=0, b=4, dims="daypart")

        # Varying intercepts
        a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("profile_cluster"))

        # Varying slopes:
        bs1 = pm.Normal("bs1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bs2 = pm.Normal("bs2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bs3 = pm.Normal("bs3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        bc1 = pm.Normal("bc1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bc2 = pm.Normal("bc2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bc3 = pm.Normal("bc3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        # Balance temp
        tbal_h = pm.Normal("tbal_h", mu = 18.0, sigma = 1.5, dims="daypart")
        tbal_c = pm.Normal("tbal_c", mu=18.0, sigma=1.5, dims="daypart")

        # Dependence
        dep_h = pm.Bernoulli("dep_h", p = 0.5 , dims="daypart")
        dep_c = pm.Bernoulli("dep_c", p = 0.5, dims="daypart")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bs1[profile_cluster_idx] * fs_sin_1 + \
             bs2[profile_cluster_idx] * fs_sin_2 + bs3[profile_cluster_idx] * fs_sin_3 + \
             bc1[profile_cluster_idx] * fs_cos_1 + bc2[profile_cluster_idx] * fs_cos_2 + \
             bc3[profile_cluster_idx] * fs_cos_3 + \
             btc[profile_cluster_idx] * (outdoor_temp - tbal_c[daypart]) * ((outdoor_temp - tbal_c[daypart])>0) * (dep_c[daypart]>0.5) + \
             bth[profile_cluster_idx] * (tbal_h[daypart] - outdoor_temp) * ((tbal_h[daypart] - outdoor_temp)>0) * (dep_h[daypart]>0.5)
             # btclp[profile_cluster_idx] * cooling_temp_lp + \
             # bthlp[profile_cluster_idx] * heating_temp_lp + \
             # btc[profile_cluster_idx] * cooling_temp + \
             # bth[profile_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

    # Sample fitting

    with partial_pooling:
        partial_pooling_trace = pm.sample(2000)

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
                     "heating_temp_lp": outdoor_temp_lp_h_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        partial_pool_posterior_hdi = pm.sample_posterior_predictive(partial_pooling_trace, keep_size=True)
        partial_pool_posterior = pm.sample_posterior_predictive(partial_pooling_trace)
        partial_pool_prior = pm.sample_prior_predictive(150)

    # Debugging ADVI to understand if the estimation is converging
    # az.plot_trace(partial_pooling_trace['tbal_h'][None, :, :])
    # az.plot_trace(partial_pooling_trace['tbal_c'][None, :, :])
    # az.plot_trace(partial_pooling_trace['bth'][None, :, :])
    # az.plot_trace(partial_pooling_trace['dep_h'][None, :, :])
    # az.plot_trace(partial_pooling_trace['dep_c'][None, :, :])
    # plt.show()
    # plt.plot(approx.hist)
    #
    # advi_elbo = pd.DataFrame(
    #     {'log-ELBO': -np.log(approx.hist),
    #      'n': np.arange(approx.hist.shape[0])})
    #
    # plt.plot(advi_elbo['n'], advi_elbo['log-ELBO'])
    # plt.show()

    # Calculate predictions and HDI

    partial_pool_predictions = np.exp(partial_pool_posterior['y'].mean(0))
    hdi_data = az.hdi(partial_pool_posterior_hdi)
    partial_pool_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    partial_pool_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    partial_pool_gamma =  np.where(test_df.total_electricity > partial_pool_higher_bound,
                      1 + ((test_df.total_electricity - partial_pool_higher_bound)/(test_df.total_electricity)),
                      np.where(test_df.total_electricity < partial_pool_lower_bound,
                               1 + ((partial_pool_lower_bound - test_df.total_electricity)/(test_df.total_electricity)),
                               1))

    partial_pool_adjusted_coverage = np.nanmean(partial_pool_gamma * (partial_pool_higher_bound) /(partial_pool_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    partial_pool_mse = mean_squared_error(test_df.total_electricity, partial_pool_predictions)
    partial_pool_rmse = sqrt(partial_pool_mse)
    partial_pool_cvrmse = partial_pool_rmse / test_df.total_electricity.mean()
    partial_pool_coverage = sum((partial_pool_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= partial_pool_higher_bound)) * 100 / len(test_df)
    partial_pool_confidence_length = sum(partial_pool_higher_bound) - sum(partial_pool_lower_bound)

    # Calculate NMBE
    partial_pool_nmbe = np.sum(test_df.total_electricity - partial_pool_predictions) * 100 / len(test_df) / test_df.total_electricity.mean()

    # Print df
    pp_data = {'t': test_df['t'],
               'prediction': partial_pool_predictions,
               'lower_bound': partial_pool_lower_bound,
               'higher_bound': partial_pool_higher_bound}

    pp_results = pd.DataFrame(data=pp_data)
    pp_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_pp.csv", index=False)
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
        a_cluster = pm.Normal("a_cluster", mu=0.0, sigma=1.0, dims="profile_cluster")
        btclp = pm.Normal("btclp", mu=0.0, sigma=1.0, dims="profile_cluster")
        bthlp = pm.Normal("bthlp", mu=0.0, sigma=1.0, dims="profile_cluster")
        btc = pm.Normal("btc", mu=0.0, sigma=1.0, dims="profile_cluster")
        bth = pm.Normal("bth", mu=0.0, sigma=1.0, dims="profile_cluster")

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
             btclp[profile_cluster_idx] * cooling_temp_lp + \
             bthlp[profile_cluster_idx] * heating_temp_lp + \
             btc[profile_cluster_idx] * cooling_temp + \
             bth[profile_cluster_idx] * heating_temp

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
                      1 + ((test_df.total_electricity - no_pool_higher_bound)/(test_df.total_electricity)),
                      np.where(test_df.total_electricity < no_pool_lower_bound,
                               1 + ((no_pool_lower_bound - test_df.total_electricity)/(test_df.total_electricity)),
                               1))

    no_pool_adjusted_coverage = np.nanmean(no_pool_gamma * (no_pool_higher_bound) /(no_pool_lower_bound))


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

    # Print predictions df
    np_data = {'t': test_df['t'],
               'prediction': no_pool_predictions,
               'lower_bound': no_pool_lower_bound,
               'higher_bound': no_pool_higher_bound}

    np_results = pd.DataFrame(data=np_data)
    np_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_np.csv", index=False)

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
        btc = pm.Normal("btc", mu=0.0, sigma=1.0)
        bth = pm.Normal("bth", mu=0.0, sigma=1.0)

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
                      1 + ((test_df.total_electricity - complete_pool_higher_bound)/(test_df.total_electricity)),
                      np.where(test_df.total_electricity < complete_pool_lower_bound,
                               1 + ((complete_pool_lower_bound - test_df.total_electricity)/(test_df.total_electricity)),
                               1))

    complete_pool_adjusted_coverage = np.nanmean(complete_pool_gamma * (complete_pool_higher_bound) /(complete_pool_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    complete_pool_mse = mean_squared_error(test_df.total_electricity, complete_pool_predictions)
    complete_pool_rmse = sqrt(complete_pool_mse)
    complete_pool_cvrmse = complete_pool_rmse / test_df.total_electricity.mean()
    complete_pool_coverage = sum((complete_pool_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= complete_pool_higher_bound)) * 100 / len(test_df)
    complete_pool_confidence_length = sum(complete_pool_higher_bound) - sum(complete_pool_lower_bound)

    complete_pool_nmbe = np.sum(test_df.total_electricity - complete_pool_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Print predictions df
    cp_data = {'t': test_df['t'],
               'prediction': complete_pool_predictions,
               'lower_bound': complete_pool_lower_bound,
               'higher_bound': complete_pool_higher_bound}

    cp_results = pd.DataFrame(data=cp_data)
    cp_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_cp.csv", index=False)

    export_data = {'partial_pooling_cvrmse': [partial_pool_cvrmse],
                   'no_pooling_cvrmse': [no_pool_cvrmse],
                   'complete_pooling_cvrmse': [complete_pool_cvrmse],
                   'partial_pooling_coverage': [partial_pool_coverage],
                   'no_pooling_coverage': [no_pool_coverage],
                   'complete_pooling_coverage': [complete_pool_coverage],
                   'partial_pooling_length':[partial_pool_confidence_length],
                   'no_pooling_length': [no_pool_confidence_length],
                   'complete_pooling_length': [complete_pool_confidence_length],
                   'partial_pooling_adj_coverage': [partial_pool_adjusted_coverage],
                   'no_pooling_adj_coverage':[no_pool_adjusted_coverage],
                   'complete_pooling_adj_coverage':[complete_pool_adjusted_coverage],
                   'partial_pooling_nmbe':[partial_pool_nmbe],
                   'no_pooling_nmbe':[no_pool_nmbe],
                   'complete_pooling_nmbe':[complete_pool_nmbe],
                   'id':building_id}

    export_df = pd.DataFrame(data=export_data)
    return export_df

def bayesian_model_comparison_model_spec (df, building_id):
    #df = pd.read_csv("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Fox_education_Ollie_preprocess.csv")
    #df = pd.read_csv("/root/benedetto/results/buildings/Crow_education_Keisha_preprocess.csv")
    #df = pd.read_csv("/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Fox_education_Ollie_preprocess.csv")
    # Preprocess
    df["log_v"] = log_electricity = np.log(df["total_electricity"]).values

    # Create local variables (assign daypart, cluster and weekday values need to start from 0)
    # clusters are use profile categories, heat_clusters and cool_clusters indicate days having similar
    # temperature dependence (likely to modify this in the new version of the preprocessing)

    df.t = pd.to_datetime(pd.Series(df.t))
    df.s = df.s - 1
    #df.s = df['s'].astype(object)
    #df.daypart = df['daypart'].astype(object)
    df.weekday = df.weekday - 1
    #df.weekday = df['weekday'].astype(object)

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
    outdoor_temp_train = train_df.outdoor_temp
    #outdoor_temp_c_train = train_df.outdoor_temp_c
    #outdoor_temp_h_train = train_df.outdoor_temp_h
    #outdoor_temp_lp_c_train = train_df.outdoor_temp_lp_c
    #outdoor_temp_lp_h_train = train_df.outdoor_temp_lp_h
    daypart_fs_sin_1_train = train_df.daypart_fs_sin_1
    daypart_fs_sin_2_train = train_df.daypart_fs_sin_2
    daypart_fs_sin_3_train = train_df.daypart_fs_sin_3
    daypart_fs_cos_1_train = train_df.daypart_fs_cos_1
    daypart_fs_cos_2_train = train_df.daypart_fs_cos_2
    daypart_fs_cos_3_train = train_df.daypart_fs_cos_3
    # yearpart_fs_sin_1_train = train_df.yearpart_fs_sin_1
    # yearpart_fs_sin_2_train = train_df.yearpart_fs_sin_2
    # yearpart_fs_sin_3_train = train_df.yearpart_fs_sin_3
    # yearpart_fs_cos_1_train = train_df.yearpart_fs_cos_1
    # yearpart_fs_cos_2_train = train_df.yearpart_fs_cos_2
    # yearpart_fs_cos_3_train = train_df.yearpart_fs_cos_3

    # Define test variables
    clusters_test = test_df.s
    dayparts_test = test_df.daypart
    weekdays_test = test_df.weekday
    outdoor_temp_test = test_df.outdoor_temp
    #outdoor_temp_c_test = test_df.outdoor_temp_c
    #outdoor_temp_h_test = test_df.outdoor_temp_h
    #outdoor_temp_lp_c_test = test_df.outdoor_temp_lp_c
    #outdoor_temp_lp_h_test = test_df.outdoor_temp_lp_h
    daypart_fs_sin_1_test = test_df.daypart_fs_sin_1
    daypart_fs_sin_2_test = test_df.daypart_fs_sin_2
    daypart_fs_sin_3_test = test_df.daypart_fs_sin_3
    daypart_fs_cos_1_test = test_df.daypart_fs_cos_1
    daypart_fs_cos_2_test = test_df.daypart_fs_cos_2
    daypart_fs_cos_3_test = test_df.daypart_fs_cos_3
    # yearpart_fs_sin_1_test = test_df.yearpart_fs_sin_1
    # yearpart_fs_sin_2_test = test_df.yearpart_fs_sin_2
    # yearpart_fs_sin_3_test = test_df.yearpart_fs_sin_3
    # yearpart_fs_cos_1_test = test_df.yearpart_fs_cos_1
    # yearpart_fs_cos_2_test = test_df.yearpart_fs_cos_2
    # yearpart_fs_cos_3_test = test_df.yearpart_fs_cos_3

    # create coords for pymc3
    coords = {"obs_id": np.arange(total_electricity_train.size)}
    coords["profile_cluster"] = unique_clusters
    coords["daypart"] = unique_dayparts
    coords["weekday"] = unique_weekdays

    # Model 1 ADVI dep

    with pm.Model(coords=coords) as advi_dep:
        profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
        daypart = pm.Data("daypart", dayparts_train, dims="obs_id")
        weekday = pm.Data("weekday", weekdays_train, dims="obs_id")

        dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
        dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
        dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")

        dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
        dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
        dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")

        # yp_fs_sin_1 = pm.Data("yp_fs_sin_1", yearpart_fs_sin_1_train, dims="obs_id")
        # yp_fs_sin_2 = pm.Data("yp_fs_sin_2", yearpart_fs_sin_2_train, dims="obs_id")
        # yp_fs_sin_3 = pm.Data("yp_fs_sin_3", yearpart_fs_sin_3_train, dims="obs_id")
        #
        # yp_fs_cos_1 = pm.Data("yp_fs_cos_1", yearpart_fs_cos_1_train, dims="obs_id")
        # yp_fs_cos_2 = pm.Data("yp_fs_cos_2", yearpart_fs_cos_2_train, dims="obs_id")
        # yp_fs_cos_3 = pm.Data("yp_fs_cos_3", yearpart_fs_cos_3_train, dims="obs_id")

        #cooling_temp = pm.Data("cooling_temp", outdoor_temp_c_train, dims="obs_id")
        #heating_temp = pm.Data("heating_temp", outdoor_temp_h_train, dims="obs_id")
        #cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c_train, dims="obs_id")
        #heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h_train, dims="obs_id")
        outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")

        # Hyperpriors:
        BoundNormal = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
        #mu_btc = pm.Normal("mu_btc", mu=0.0, sigma=1.0)
        sigma_btc = pm.Exponential("sigma_btc", 1.0)
        #mu_bth = pm.Normal("mu_bth", mu=0.0, sigma=1.0)
        sigma_bth = pm.Exponential("sigma_bth", 1.0)
        mu_btclp = pm.Normal("mu_btclp", mu=0.0, sigma=1.0)
        sigma_btclp = pm.Exponential("sigma_btclp", 1.0)
        mu_bthlp = pm.Normal("mu_bthlp", mu=0.0, sigma=1.0)
        sigma_bthlp = pm.Exponential("sigma_bthlp", 1.0)
        bf = pm.Normal("bf", mu=0.0, sigma=1.0)
        sigma_bf = pm.Exponential("sigma_bf", 1.0)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        sigma_a = pm.Exponential("sigma_a", 1.0)

        btc = pm.HalfNormal("btc", sigma=sigma_btc, dims="daypart")
        bth = pm.HalfNormal("bth", sigma=sigma_bth, dims="daypart")
        btclp = pm.Normal("btclp", mu=mu_btclp, sigma=sigma_btclp, dims="profile_cluster")
        bthlp = pm.Normal("bthlp", mu=mu_bthlp, sigma=sigma_bthlp, dims="profile_cluster")

        # Varying intercepts
        a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("profile_cluster"))

        # Varying slopes:
        bsd1 = pm.Normal("bsd1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bsd2 = pm.Normal("bsd2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bsd3 = pm.Normal("bsd3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        bcd1 = pm.Normal("bcd1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bcd2 = pm.Normal("bcd2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bcd3 = pm.Normal("bcd3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        bsy1 = pm.Normal("bsy1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bsy2 = pm.Normal("bsy2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bsy3 = pm.Normal("bsy3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        bcy1 = pm.Normal("bcy1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bcy2 = pm.Normal("bcy2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
        bcy3 = pm.Normal("bcy3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))

        # Balance temp
        tbal_h = pm.Uniform("tbal_h", lower = 10, upper = 25)
        tbal_c = pm.Uniform("tbal_c", lower = 10, upper = 25)

        # Dependence
        # dep_h = BoundNormal("dep_h", mu=0.5, sigma=0.5, dims="profile_cluster")
        # dep_c = BoundNormal("dep_c", mu=0.5, sigma=0.5, dims="profile_cluster")
        dep_h = pm.Uniform("dep_h", lower=0, upper=1, dims="profile_cluster")
        dep_c = pm.Uniform("dep_c", lower=0, upper=1, dims="profile_cluster")

        # Expected value per county:
        mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
             bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
             bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
             bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
             btc[daypart] * (outdoor_temp - tbal_c) * ((outdoor_temp - tbal_c) > 0) * (
                     dep_c[profile_cluster_idx] > 0.5) + \
             bth[daypart] * (tbal_h - outdoor_temp) * ((tbal_h - outdoor_temp) > 0) * (
                     dep_h[profile_cluster_idx] > 0.5)
        # btclp[profile_cluster_idx] * cooling_temp_lp + \
        # bthlp[profile_cluster_idx] * heating_temp_lp + \
        # btc[profile_cluster_idx] * cooling_temp + \
        # bth[profile_cluster_idx] * heating_temp

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        # Likelihood
        y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")

        # Fitting
    with advi_dep:
        approx_dep = pm.fit(n=50000,
                        method='fullrank_advi',
                        callbacks=[CheckParametersConvergence(tolerance=0.01)])
        advi_dep_trace = approx_dep.sample(5000)

        # Sampling from the posterior setting test data to check the predictions on unseen data

    with advi_dep:
        pm.set_data({"profile_cluster_idx": clusters_test,
                     "daypart": dayparts_test,
                     # "weekday":weekdays_test,
                     "dp_fs_sin_1": daypart_fs_sin_1_test,
                     "dp_fs_sin_2": daypart_fs_sin_2_test,
                     "dp_fs_sin_3": daypart_fs_sin_3_test,
                     "dp_fs_cos_1": daypart_fs_cos_1_test,
                     "dp_fs_cos_2": daypart_fs_cos_2_test,
                     "dp_fs_cos_3": daypart_fs_cos_3_test,
                     # "yp_fs_sin_1": yearpart_fs_sin_1_test,
                     # "yp_fs_sin_2": yearpart_fs_sin_2_test,
                     # "yp_fs_sin_3": yearpart_fs_sin_3_test,
                     # "yp_fs_cos_1": yearpart_fs_cos_1_test,
                     # "yp_fs_cos_2": yearpart_fs_cos_2_test,
                     # "yp_fs_cos_3": yearpart_fs_cos_3_test,
                     #"cooling_temp": outdoor_temp_c_test,
                     #"heating_temp": outdoor_temp_h_test,
                     #"cooling_temp_lp": outdoor_temp_lp_c_test,
                     #"heating_temp_lp": outdoor_temp_lp_h_test,
                     "outdoor_temp": outdoor_temp_test
                     })

        advi_dep_posterior_hdi = pm.sample_posterior_predictive(advi_dep_trace, keep_size=True)
        advi_dep_posterior = pm.sample_posterior_predictive(advi_dep_trace)
        advi_dep_prior = pm.sample_prior_predictive(150)

        # save traceplots here

        # Debugging ADVI to understand if the estimation is converging

    az.plot_trace(advi_dep_trace['tbal_h'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_ad.png')
    az.plot_trace(advi_dep_trace['tbal_c'][None, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_ad.png')
    az.plot_trace(advi_dep_trace['bth'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_ad.png')
    az.plot_trace(advi_dep_trace['btc'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_ad.png')
    az.plot_trace(advi_dep_trace['dep_h'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_ad.png')
    az.plot_trace(advi_dep_trace['dep_c'][None, :, :])
    plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_ad.png')

    plt.show()
    # advi_d_elbo = pd.DataFrame(
    #     {'log-ELBO': -np.log(approx_dep.hist),
    #      'n': np.arange(approx_dep.hist.shape[0])})
    #
    # plt.plot(advi_d_elbo['n'], advi_d_elbo['log-ELBO'])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_elbo_ad.png')

    # Calculate predictions and HDI

    advi_dep_predictions = np.exp(advi_dep_posterior['y'].mean(0))
    hdi_data = az.hdi(advi_dep_posterior_hdi)
    advi_dep_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    advi_dep_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()

    # Calculate adjusted coverage
    advi_dep_gamma = np.where(test_df.total_electricity > advi_dep_higher_bound,
                              1 + ((test_df.total_electricity - advi_dep_higher_bound) / (
                                  test_df.total_electricity)),
                              np.where(test_df.total_electricity < advi_dep_lower_bound,
                                       1 + ((advi_dep_lower_bound - test_df.total_electricity) / (
                                           test_df.total_electricity)),
                                       1))

    advi_dep_adjusted_coverage = np.nanmean(
        advi_dep_gamma * (advi_dep_higher_bound) / (advi_dep_lower_bound))

    # Calculate cvrmse and coverage of the HDI
    advi_dep_mse = mean_squared_error(test_df.total_electricity, advi_dep_predictions)
    advi_dep_rmse = sqrt(advi_dep_mse)
    advi_dep_cvrmse = advi_dep_rmse / test_df.total_electricity.mean()
    advi_dep_coverage = sum((advi_dep_lower_bound <= test_df.total_electricity) & (
            test_df.total_electricity <= advi_dep_higher_bound)) * 100 / len(test_df)
    advi_dep_confidence_length = sum(advi_dep_higher_bound) - sum(advi_dep_lower_bound)

    # Calculate NMBE
    advi_dep_nmbe = np.sum(test_df.total_electricity - advi_dep_predictions) * 100 / len(
        test_df) / test_df.total_electricity.mean()

    # Bokeh plots to compare NUTS and ADVI
    # p1 = figure(plot_width=800, plot_height=400, x_axis_type='datetime')
    # p1.line(test_df['t'], advi_dep_predictions, color="navy", alpha=0.8)
    # p1.line(test_df['t'], test_df['total_electricity'], color="orange", alpha=0.6)
    # p1.varea(x=test_df['t'], y1=advi_dep_lower_bound, y2=advi_dep_higher_bound, color='gray', alpha=0.2)
    # show(p1)

    # p2 = figure(plot_width=800, plot_height=400, x_axis_type='datetime')
    # p2.line(test_df['t'], nuts_binomial_predictions, color="navy", alpha=0.8)
    # p2.line(test_df['t'], test_df['total_electricity'], color="orange", alpha=0.6)
    # p2.varea(x=test_df['t'], y1=nuts_binomial_lower_bound, y2=nuts_binomial_higher_bound, color='gray', alpha=0.2)
    # show(p2)

    # Print df
    ad_data = {'t': test_df['t'],
               'prediction': advi_dep_predictions,
               'lower_bound': advi_dep_lower_bound,
               'higher_bound': advi_dep_higher_bound}

    ad_results = pd.DataFrame(data=ad_data)
    ad_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_ad.csv", index=False)

    # Model 2: NUTS binomial

    # with pm.Model(coords=coords) as nuts_binomial:
    #     profile_cluster_idx = pm.Data("profile_cluster_idx", clusters_train, dims="obs_id")
    #     daypart = pm.Data("daypart", dayparts_train, dims="obs_id")
    #     #weekday = pm.Data("weekday", weekdays_train, dims="obs_id")
    #
    #     dp_fs_sin_1 = pm.Data("dp_fs_sin_1", daypart_fs_sin_1_train, dims="obs_id")
    #     dp_fs_sin_2 = pm.Data("dp_fs_sin_2", daypart_fs_sin_2_train, dims="obs_id")
    #     dp_fs_sin_3 = pm.Data("dp_fs_sin_3", daypart_fs_sin_3_train, dims="obs_id")
    #
    #     dp_fs_cos_1 = pm.Data("dp_fs_cos_1", daypart_fs_cos_1_train, dims="obs_id")
    #     dp_fs_cos_2 = pm.Data("dp_fs_cos_2", daypart_fs_cos_2_train, dims="obs_id")
    #     dp_fs_cos_3 = pm.Data("dp_fs_cos_3", daypart_fs_cos_3_train, dims="obs_id")
    #
    #     #yp_fs_sin_1 = pm.Data("yp_fs_sin_1", yearpart_fs_sin_1_train, dims="obs_id")
    #     #yp_fs_sin_2 = pm.Data("yp_fs_sin_2", yearpart_fs_sin_2_train, dims="obs_id")
    #     #yp_fs_sin_3 = pm.Data("yp_fs_sin_3", yearpart_fs_sin_3_train, dims="obs_id")
    #
    #     #yp_fs_cos_1 = pm.Data("yp_fs_cos_1", yearpart_fs_cos_1_train, dims="obs_id")
    #     #yp_fs_cos_2 = pm.Data("yp_fs_cos_2", yearpart_fs_cos_2_train, dims="obs_id")
    #     #yp_fs_cos_3 = pm.Data("yp_fs_cos_3", yearpart_fs_cos_3_train, dims="obs_id")
    #
    #     #cooling_temp = pm.Data("cooling_temp", outdoor_temp_c_train, dims="obs_id")
    #     #heating_temp = pm.Data("heating_temp", outdoor_temp_h_train, dims="obs_id")
    #     #cooling_temp_lp = pm.Data("cooling_temp_lp", outdoor_temp_lp_c_train, dims="obs_id")
    #     #heating_temp_lp = pm.Data("heating_temp_lp", outdoor_temp_lp_h_train, dims="obs_id")
    #     outdoor_temp = pm.Data("outdoor_temp", outdoor_temp_train, dims="obs_id")
    #
    #     # Hyperpriors:
    #     #BoundNormal = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
    #     #mu_btc = pm.Normal("mu_btc", mu=0.0, sigma=1.0)
    #     sigma_btc = pm.Exponential("sigma_btc", 1.0)
    #     #mu_bth = pm.Normal("mu_bth", mu=0.0, sigma=1.0)
    #     sigma_bth = pm.Exponential("sigma_bth", 1.0)
    #     #mu_btclp = pm.Normal("mu_btclp", mu=0.0, sigma=1.0)
    #     #sigma_btclp = pm.Exponential("sigma_btclp", 1.0)
    #     #mu_bthlp = pm.Normal("mu_bthlp", mu=0.0, sigma=1.0)
    #     #sigma_bthlp = pm.Exponential("sigma_bthlp", 1.0)
    #     bf = pm.Normal("bf", mu=0.0, sigma=1.0)
    #     sigma_bf = pm.Exponential("sigma_bf", 1.0)
    #     a = pm.Normal("a", mu=0.0, sigma=1.0)
    #     sigma_a = pm.Exponential("sigma_a", 1.0)
    #     # k = pm.Gamma('k', alpha=15, beta=15)
    #     # lambda_ = pm.Gamma('lambda_', alpha=15, beta=15)
    #     # bounded_laplacian = pm.Bound(pm.Laplace, lower=0, upper=5)
    #
    #     btc = pm.HalfNormal("btc", sigma=sigma_btc, dims="daypart")
    #     bth = pm.HalfNormal("bth", sigma=sigma_bth, dims="daypart")
    #     #btclp = pm.Normal("btclp", mu=mu_btclp, sigma=sigma_btclp, dims="profile_cluster")
    #     #bthlp = pm.Normal("bthlp", mu=mu_bthlp, sigma=sigma_bthlp, dims="profile_cluster")
    #
    #     # Varying intercepts
    #     a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims=("profile_cluster"))
    #
    #     # Varying slopes:
    #     bsd1 = pm.Normal("bsd1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     bsd2 = pm.Normal("bsd2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     bsd3 = pm.Normal("bsd3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #
    #     bcd1 = pm.Normal("bcd1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     bcd2 = pm.Normal("bcd2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     bcd3 = pm.Normal("bcd3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #
    #     #bsy1 = pm.Normal("bsy1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     #bsy2 = pm.Normal("bsy2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     #bsy3 = pm.Normal("bsy3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #
    #     #bcy1 = pm.Normal("bcy1", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     #bcy2 = pm.Normal("bcy2", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #     #bcy3 = pm.Normal("bcy3", mu=bf, sigma=sigma_bf, dims=("profile_cluster"))
    #
    #     # Balance temp
    #     tbal_h = pm.Normal("tbal_h", mu=18.0, sigma=1.5, dims = "daypart")
    #     tbal_c = pm.Normal("tbal_c", mu=18.0, sigma=1.5, dims = "daypart")
    #
    #     # Dependence
    #     #dep_h = pm.Bernoulli("dep_h", p=0.5, dims="profile_cluster")
    #     #dep_c = pm.Bernoulli("dep_c", p=0.5, dims="profile_cluster")
    #
    #     # Expected value per county:
    #     mu = a_cluster[profile_cluster_idx] + bsd1[profile_cluster_idx] * dp_fs_sin_1 + \
    #          bsd2[profile_cluster_idx] * dp_fs_sin_2 + bsd3[profile_cluster_idx] * dp_fs_sin_3 + \
    #          bcd1[profile_cluster_idx] * dp_fs_cos_1 + bcd2[profile_cluster_idx] * dp_fs_cos_2 + \
    #          bcd3[profile_cluster_idx] * dp_fs_cos_3 + \
    #          btc[daypart] * (outdoor_temp - tbal_c[daypart]) * ((outdoor_temp - tbal_c[daypart]) > 0) * (
    #                  btc[daypart] > 0.001) + \
    #          bth[daypart] * (tbal_h[daypart] - outdoor_temp) * ((tbal_h[daypart] - outdoor_temp) > 0) * (
    #                  bth[daypart] > 0.001)
    #
    #     # Model error:
    #     sigma = pm.Exponential("sigma", 1.0)
    #
    #     # Likelihood
    #     y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity_train, dims="obs_id")
    #
    # # Sample fitting
    #
    # with nuts_binomial:
    #     nuts_binomial_trace = pm.sample(8000, tune = 2000, chains = 4, cores = 1, target_accept=0.9)
    #
    # # Sampling from the posterior setting test data to check the predictions on unseen data
    # with nuts_binomial:
    #     pm.set_data({"profile_cluster_idx": clusters_test,
    #                  "daypart": dayparts_test,
    #                  # "weekday":weekdays_test,
    #                  "dp_fs_sin_1": daypart_fs_sin_1_test,
    #                  "dp_fs_sin_2": daypart_fs_sin_2_test,
    #                  "dp_fs_sin_3": daypart_fs_sin_3_test,
    #                  "dp_fs_cos_1": daypart_fs_cos_1_test,
    #                  "dp_fs_cos_2": daypart_fs_cos_2_test,
    #                  "dp_fs_cos_3": daypart_fs_cos_3_test,
    #                  "yp_fs_sin_1": yearpart_fs_sin_1_test,
    #                  "yp_fs_sin_2": yearpart_fs_sin_2_test,
    #                  "yp_fs_sin_3": yearpart_fs_sin_3_test,
    #                  "yp_fs_cos_1": yearpart_fs_cos_1_test,
    #                  "yp_fs_cos_2": yearpart_fs_cos_2_test,
    #                  "yp_fs_cos_3": yearpart_fs_cos_3_test,
    #                  #"cooling_temp": outdoor_temp_c_test,
    #                  #"heating_temp": outdoor_temp_h_test,
    #                  #"cooling_temp_lp": outdoor_temp_lp_c_test,
    #                  #"heating_temp_lp": outdoor_temp_lp_h_test,
    #                  "outdoor_temp": outdoor_temp_test
    #                  })
    #
    #     nuts_binomial_posterior_hdi = pm.sample_posterior_predictive(nuts_binomial_trace, keep_size=True)
    #     nuts_binomial_posterior = pm.sample_posterior_predictive(nuts_binomial_trace)
    #
    # # Save traceplots for temperatures, dep and elbo here
    #
    # # Debugging ADVI to understand if the estimation is converging
    # az.plot_trace(nuts_binomial_trace['tbal_h'][None, :])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_h_nb.png')
    # az.plot_trace(nuts_binomial_trace['tbal_c'][None, :])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_tbal_c_nb.png')
    # az.plot_trace(nuts_binomial_trace['bth'][None, :, :])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_bth_nb.png')
    # az.plot_trace(nuts_binomial_trace['btc'][None, :, :])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_btc_nb.png')
    # az.plot_trace(nuts_binomial_trace['dep_h'][None, :, :])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_h_nb.png')
    # az.plot_trace(nuts_binomial_trace['dep_c'][None, :, :])
    # plt.savefig('/root/benedetto/results/plots/' + building_id + '_dep_c_nb.png')
    #
    # plt.show()
    # # Calculate predictions and HDI
    #
    # nuts_binomial_predictions = np.exp(nuts_binomial_posterior['y'].mean(0))
    # hdi_data = az.hdi(nuts_binomial_posterior_hdi)
    # nuts_binomial_lower_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='lower'))).flatten()
    # nuts_binomial_higher_bound = np.array(np.exp(hdi_data.to_array().sel(hdi='higher'))).flatten()
    #
    # # Calculate adjusted coverage
    # nuts_binomial_gamma = np.where(test_df.total_electricity > nuts_binomial_higher_bound,
    #                               1 + ((test_df.total_electricity - nuts_binomial_higher_bound) / (
    #                                   test_df.total_electricity)),
    #                               np.where(test_df.total_electricity < nuts_binomial_lower_bound,
    #                                        1 + ((nuts_binomial_lower_bound - test_df.total_electricity) / (
    #                                            test_df.total_electricity)),
    #                                        1))
    #
    # nuts_binomial_adjusted_coverage = np.nanmean(
    #     nuts_binomial_gamma * (nuts_binomial_higher_bound) / (nuts_binomial_lower_bound))
    #
    # # Calculate cvrmse and coverage of the HDI
    # nuts_binomial_mse = mean_squared_error(test_df.total_electricity, nuts_binomial_predictions)
    # nuts_binomial_rmse = sqrt(nuts_binomial_mse)
    # nuts_binomial_cvrmse = nuts_binomial_rmse / test_df.total_electricity.mean()
    # nuts_binomial_coverage = sum((nuts_binomial_lower_bound <= test_df.total_electricity) & (
    #         test_df.total_electricity <= nuts_binomial_higher_bound)) * 100 / len(test_df)
    # nuts_binomial_confidence_length = sum(nuts_binomial_higher_bound) - sum(nuts_binomial_lower_bound)
    #
    # # Calculate NMBE
    # nuts_binomial_nmbe = np.sum(test_df.total_electricity - nuts_binomial_predictions) * 100 / len(
    #     test_df) / test_df.total_electricity.mean()
    #
    # p1 = figure(plot_width=800, plot_height=400, x_axis_type='datetime')
    # p1.line(test_df['t'], nuts_binomial_predictions, color="navy", alpha=0.8)
    # p1.line(test_df['t'], test_df['total_electricity'], color="orange", alpha=0.6)
    # p1.varea(x=test_df['t'], y1=nuts_binomial_lower_bound, y2=nuts_binomial_higher_bound, color='gray', alpha=0.2)
    # show(p1)
    #
    # # Print df
    # nb_data = {'t': test_df['t'],
    #            'prediction': nuts_binomial_predictions,
    #            'lower_bound': nuts_binomial_lower_bound,
    #            'higher_bound': nuts_binomial_higher_bound}
    #
    # nb_results = pd.DataFrame(data=nb_data)
    # nb_results.to_csv("/root/benedetto/results/predictions/" + building_id + "_nb.csv", index=False)


    export_data = {'advi_dep_cvrmse': [advi_dep_cvrmse],
                   'advi_dep_adjusted_coverage': [advi_dep_adjusted_coverage],
                   'advi_dep_nmbe': [advi_dep_nmbe],
                   'id': building_id}

    export_df = pd.DataFrame(data=export_data)
    return export_df


def multiprocessing_bayesian_comparison(df):

    building_id = df.columns[1]
    df.columns = ['t', 'total_electricity', 'outdoor_temp']

    # Print consumption and temperature df if it doesn't exist already

    if os.path.isfile("/root/benedetto/results/buildings/" + building_id + ".csv"):
        print(building_id + ' data was retreived successfully')
    else:
        df.to_csv("/root/benedetto/results/buildings/" + building_id + ".csv", index=False)

    # Try to read preprocessed file, otherwise run preprocessing
    try:
        df_preprocessed = pd.read_csv("/root/benedetto/results/buildings_windspeed/" + building_id + "_preprocess.csv")
        print(building_id + ' preprocessed data was retrieved succesfully')
    except Exception as e:
        print(e)
        print('Running preprocessing for ' + building_id)
        subprocess.run(["Rscript", "ashrae_preprocess_server.R", building_id, building_id])
        try:
            df_preprocessed = pd.read_csv("/root/benedetto/results/buildings_windspeed/" + building_id + "_preprocess.csv")
        except Exception as e:
            print(e)
            print("Preprocessing failed for " + building_id + '. Skipping to next building.')

    try:
        dat = pd.read_csv("/root/benedetto/results/bayes_results.csv")
        results_exist = True
    except:
        results_exist = False

    # If model results already exist for the selected building, skip to next
    if results_exist == True:
        if building_id in dat.values:
            print('Results for ' + building_id + ' are already calculated. Skipping to next building')
        else:
            try:
                model_results = bayesian_model_comparison_test_3(df_preprocessed, building_id)
                res = pd.read_csv("/root/benedetto/results/bayes_results.csv")
                final_export = res.append(model_results)
                final_export.to_csv("/root/benedetto/results/bayes_results.csv", index=False)
                print('Successfully added ' + building_id + ' to results file')
            except Exception as e:
                print(e)
                print('Modeling error for ' + building_id + '. Skipping to the next building')
    else:
        try:
            model_results = bayesian_model_comparison_test_3(df_preprocessed, building_id)
            final_export = model_results
            final_export.to_csv("/root/benedetto/results/bayes_results.csv", index=False)
        except Exception as e:
            print(e)
            print('Modeling error for ' + building_id + '. Skipping to the next building')
