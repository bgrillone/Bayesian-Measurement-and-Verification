import arviz  as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings

df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/hourly_multilevel_office.csv")

# Let's follow the radon notebook and implement step by step the different models

# Finally, create local copies of variables.

RANDOM_SEED = 8924
#Select training data subset
df = df[df.m == 0]
# It might be that the model can not work properly if the cluster index variable does not start from zero
df.cluster = df.cluster -1
df.cluster_hour = df.cluster_hour -1
cluster = df.cluster
cluster_hour = df.cluster_hour
unique_clusters = cluster.unique()
unique_cluster_hour = cluster_hour.unique()

# Create local copies of variables.
electricity = df.total_electricity
df["log_electricity"] = log_electricity = np.log(electricity).values
df['temperature_deviations'] = np.where(df['temp_dep_h'] > 0, df['temp_dep_h'], df['temp_dep_c'])
n_hours = len(df.index)
temperature = df.temperature
temperature_deviation = df.temperature_deviations
temp_dep_c = df.temp_dep_c
temp_dep_h = df.temp_dep_h
GHI = df.GHI_dep
coords = {"obs_id": np.arange(temperature.size)}
coords["Cluster_hour"] = unique_cluster_hour

electricity = df.total_electricity
df["log_electricity"] = log_electricity = np.log(electricity + 0.1).values

# Distribution of electricity levels (linear and log scale):
df.total_electricity.hist(bins = 25)
plt.show()

# Let's exclude the flat cluster to see if we have something closer to a normal distribution of value
# The problem is that even if we exclude the flat clusters, we still have all the night hours of the non-flat clusters
# If we want the data to look normal we should exclude those as well (totally hardcoded for now)
df_noflat = df[(df.cluster != 6) & (df.total_electricity >10)]
df_noflat.total_electricity.hist(bins = 25)
plt.show()

plt.scatter(df_noflat.t, df_noflat.total_electricity, c = df_noflat.cluster)

df.log_electricity.hist(bins=25);
plt.show()

# Neither the data, nor it's log looks in any way similar to normal (on hourly level at least)
# Is this going to be a problem?

# DAILY PARTIAL POOLING MODEL 1: varying intercept by cluster + fixed slope for temperature

# HOURLY PARTIAL POOLING MODEL


with pm.Model(coords=coords) as partial_pooling:
    cluster_hour_idx = pm.Data("cluster_hour_idx", cluster_hour, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster_hour")

    # Expected value per county:
    mu = a_cluster[cluster_hour_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

pp_graph = pm.model_to_graphviz(partial_pooling)
pp_graph.render(filename='img/partial_pooling_hourly')

with partial_pooling:
    partial_pooled_trace = pm.sample(random_seed=RANDOM_SEED)
    partial_pooled_idata = az.from_pymc3(partial_pooled_trace)

# Let's analyse the chains
az.summary(partial_pooled_idata, round_to=2)
az.plot_trace(partial_pooled_idata)
plt.show()

# Let's calculate some predictions from the posterior

partial_pooled_means = np.mean(partial_pooled_trace['a_cluster'], axis =0)
# Create array with predictions
partial_pooled_predictions = []

for hour in range(0,8600):
    for cluster_hour_idx in unique_cluster_hour:
        if cluster_hour[hour] == cluster_hour_idx:
            partial_pooled_predictions.append(partial_pooled_means[cluster_hour_idx])


plt.scatter(x = df[0:8600]['t'],y = partial_pooled_predictions, label='varying intercept model', s = 10)
plt.scatter(x = df[0:8600]['t'], y = df[0:8600]['log_electricity'], label='observed', s = 10)
plt.legend(loc='upper left')
plt.show()


#VARYING INTERCEPT AND TEMPERATURE SLOPE


with pm.Model(coords=coords) as varying_intercept_and_temp:
    cluster_hour_idx = pm.Data("cluster_hour_idx", cluster, dims="obs_id")
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    bh = pm.Normal("bh", mu=0.0, sigma=1.0)
    sigma_bh = pm.Exponential("sigma_bh", 0.5)
    bc = pm.Normal("bc", mu=0.0, sigma=1.0)
    sigma_bc = pm.Exponential("sigma_bc", 0.5)

    # Varying intercepts:
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster_hour")
    # Varying slopes:
    bh_cluster = pm.Normal("bh_cluster", mu=bh, sigma=sigma_bh, dims="Cluster_hour")
    bc_cluster = pm.Normal("bc_cluster", mu=bc, sigma=sigma_bc, dims="Cluster_hour")


    # Electricity prediction
    mu = a_cluster[cluster_hour_idx] + bh_cluster[cluster_hour_idx] * heating_temp + bc_cluster[cluster_hour_idx] * cooling_temp
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

varying_intercept_and_temp_graph = pm.model_to_graphviz(varying_intercept_and_temp)
varying_intercept_and_temp_graph.render(filename='img/varying_intercept_and_temp_hourly')


# Before running the model let’s do some prior predictive checks (not sure what this actually does)

with varying_intercept_and_temp:
    varying_intercept_slope_trace = pm.sample(random_seed=RANDOM_SEED)
    varying_temp_idata = az.from_pymc3(varying_intercept_slope_trace)

# Let's analyse the chains
az.summary(varying_temp_idata, round_to=2)
az.plot_trace(varying_temp_idata)
plt.show()


# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

varying_intercept_slope_means = np.mean(varying_intercept_slope_trace['a_cluster'], axis =0)
varying_intercept_slope_bh = np.mean(varying_intercept_slope_trace['bh_cluster'], axis = 0)
varying_intercept_slope_bc = np.mean(varying_intercept_slope_trace['bc_cluster'], axis = 0)
# Create array with predictions
varying_intercept_slope_predictions = []
# Create array with bounds
varying_intercept_slope_hdi = az.hdi(varying_temp_idata)
varying_intercept_slope_mean_lower = []
varying_intercept_slope_mean_higher= []
varying_intercept_slope_lower = []
varying_intercept_slope_higher= []

for hour in range(0,8600):
    for cluster_hour_idx in unique_cluster_hour:
        if cluster[hour] == cluster_hour_idx:
            varying_intercept_slope_predictions.append(varying_intercept_slope_means[cluster_hour_idx] +
                                                       varying_intercept_slope_bh[cluster_hour_idx] * temp_dep_h[hour] +
                                                       varying_intercept_slope_bc[cluster_hour_idx] * temp_dep_c[hour])

            varying_intercept_slope_mean_lower.append(varying_intercept_slope_hdi['a_cluster'][cluster_hour_idx].sel(hdi = 'lower') +
                                                       varying_intercept_slope_hdi['bh_cluster'][cluster_hour_idx].sel(
                                                           hdi='lower') * temp_dep_h[hour] +
                                                       varying_intercept_slope_hdi['bc_cluster'][cluster_hour_idx].sel(
                                                           hdi='lower') * temp_dep_c[hour])

            varying_intercept_slope_mean_higher.append(varying_intercept_slope_hdi['a_cluster'][cluster_hour_idx].sel(hdi = 'higher') +
                                                       varying_intercept_slope_hdi['bh_cluster'][cluster_hour_idx].sel(
                                                           hdi='higher') * temp_dep_h[hour] +
                                                       varying_intercept_slope_hdi['bc_cluster'][cluster_hour_idx].sel(
                                                           hdi='higher') * temp_dep_c[hour])

            varying_intercept_slope_lower.append( varying_intercept_slope_hdi['a_cluster'][cluster_hour_idx].sel(hdi='lower') +
                varying_intercept_slope_hdi['bh_cluster'][cluster_hour_idx].sel(hdi='lower') * temp_dep_h[hour] +
                varying_intercept_slope_hdi['bc_cluster'][cluster_hour_idx].sel(hdi='lower') * temp_dep_c[hour] -
                                                  varying_intercept_slope_hdi['sigma'].sel(hdi='higher'))


            varying_intercept_slope_higher.append( varying_intercept_slope_hdi['a_cluster'][cluster_hour_idx].sel(hdi='higher') +
                varying_intercept_slope_hdi['bh_cluster'][cluster_hour_idx].sel(hdi='higher') * temp_dep_h[hour] +
                varying_intercept_slope_hdi['bc_cluster'][cluster_hour_idx].sel(hdi='higher') * temp_dep_c[hour]+
                                                  varying_intercept_slope_hdi['sigma'].sel(hdi='higher'))


# Plot HDI
plt.figure(figsize=(20,10))
plt.scatter(x = df[0:8600]['t'], y = df[0:8600]['log_electricity'], label='Observed', s = 10, zorder = 4)
plt.scatter(x = df[0:8600]['t'], y = varying_intercept_slope_predictions[0:8600], color = 'orangered',
            label='Varying intercept and slope model', zorder = 3, s = 14)
vlines = plt.vlines(np.arange(8600), varying_intercept_slope_mean_lower[0:8600], varying_intercept_slope_mean_higher[0:8600],
                    color='darkorange', label='Exp. distribution', zorder=2)
vlines = plt.vlines(np.arange(8600), varying_intercept_slope_lower[0:8600], varying_intercept_slope_higher[0:8600],
                    color='bisque', label='Exp. mean HPD', zorder=1)
plt.legend(ncol = 2)
plt.show()
