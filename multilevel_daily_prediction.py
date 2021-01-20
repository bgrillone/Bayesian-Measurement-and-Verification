import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings
from theano import tensor as tt

df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/daily_multilevel_office.csv")
RANDOM_SEED = 8924

# Let's follow the radon notebook and implement step by step the different models

# We'll try to implement a daily model with covarying intercept and temperature slope depending on the
# cluster variable

# We'll start with the centered version, to then move to the non-centered version for more efficient sampling

# It might be that the model can not work properly if the cluster index variable does not start from zero
df.cluster = df.cluster -1
cluster = df.cluster
unique_clusters = cluster.unique()

# Create local copies of variables.
electricity = df.total_electricity
df["log_electricity"] = log_electricity = np.log(electricity + 0.1).values
df['temperature_deviations'] = np.where(df['temp_dep_h'] > 0, df['temp_dep_h'], df['temp_dep_c'])
df['t'] = pd.to_datetime(df['t'], format= '%Y-%m-%d')
n_hours = len(df.index)
n_clusters = len(unique_clusters)
temperature = df.temperature
temperature_deviation = df.temperature_deviations
temp_dep_c = df.temp_dep_c
temp_dep_h = df.temp_dep_h
GHI = df.GHI_dep
coords = {"obs_id": np.arange(temperature.size)}
coords["Cluster"] = unique_clusters

# Distribution of electricity levels (log scale):
log_cons_plt = df.log_electricity.hist(bins=25)
log_cons_plt.set_xlabel('log consumption')
plt.show()
cons_plt = df.total_electricity.hist(bins = 25)
cons_plt.set_xlabel('consumption (kWh)')
plt.show()

plt
# The variable that we're trying to model is clearly not normal, let's see if within each cluster
# consumption is actually normally distributed:

fig, axs = plt.subplots(3,3)
axs[0,0].hist(df[df['cluster']==0].total_electricity, bins = 30)
axs[0,1].hist(df[df['cluster']==1].total_electricity, bins = 30)
axs[0,2].hist(df[df['cluster']==2].total_electricity, bins = 30)
axs[1,0].hist(df[df['cluster']==3].total_electricity, bins = 30)
axs[1,1].hist(df[df['cluster']==4].total_electricity, bins = 30)
axs[1,2].hist(df[df['cluster']==5].total_electricity, bins = 30)
axs[2,0].hist(df[df['cluster']==6].total_electricity, bins = 30)
plt.show()


fig, axs = plt.subplots(3,3)
axs[0,0].hist(df[df['cluster']==0].log_electricity, bins = 30)
axs[0,1].hist(df[df['cluster']==1].log_electricity, bins = 30)
axs[0,2].hist(df[df['cluster']==2].log_electricity, bins = 30)
axs[1,0].hist(df[df['cluster']==3].log_electricity, bins = 30)
axs[1,1].hist(df[df['cluster']==4].log_electricity, bins = 30)
axs[1,2].hist(df[df['cluster']==5].log_electricity, bins = 30)
axs[2,0].hist(df[df['cluster']==6].log_electricity, bins = 30)
plt.show()

# Looks closer to a normal distribution

# COMPLETE POOLING MODEL WITH TEMPERATURE TERM

with pm.Model(coords=coords) as complete_pooling:
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    bh = pm.Normal("bh", mu=0.0, sigma=10.0)
    bc = pm.Normal("bc", mu=0.0, sigma=10.0)

    # Expected value per county:
    mu = a + bh * heating_temp + bc * cooling_temp
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

cp_graph = pm.model_to_graphviz(complete_pooling)
cp_graph.render(filename='img/complete_pooling')

# Before running the model let’s do some prior predictive checks.

with complete_pooling:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

#Let's run the model
with complete_pooling:
    pooled_trace = pm.sample(random_seed=RANDOM_SEED)
    pooled_idata = az.from_pymc3(pooled_trace)

az.summary(pooled_idata, round_to=2)
# The chains look very good (good R hat, good effective sample size, small sds

with complete_pooling:
    ppc = pm.sample_posterior_predictive(pooled_trace, random_seed=RANDOM_SEED)
    pooled_idata = az.from_pymc3(pooled_trace, posterior_predictive=ppc, prior=prior_checks)

# We have now converted our trace and posterior predictive samples into an arviz.InferenceData object.
# InferenceData is specifically designed to centralize all the relevant quantities of a Bayesian inference
# workflow into a single object.

az.plot_trace(pooled_idata)
plt.show()

#Chains and coefficients look good
# Let's try to plot something that can help us understand if the predictions are good or not.

with complete_pooling:
    ppc = pm.sample_posterior_predictive(
        pooled_trace, var_names=["a", "b", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.

az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=complete_pooling))
plt.show()

# The way I interpret this graph: we're trying to force normality on a variable that is not normal.
# What are the consequences of this? Discuss with Sotiris

pooled_means = pooled_idata.posterior.mean(dim=("chain", "draw"))

plt.scatter(x = df['t'],y = pooled_trace["a"].mean() + pooled_trace["bh"].mean() * temp_dep_h +
                            pooled_trace['bc'].mean() * temp_dep_c, label='pooled model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# Would be good to add a HDI on the previous plot
# Not sure if we can calculate it like this but seems like the only option at the moment

pooled_hdi = az.hdi(pooled_idata)
lower_bound_mean = pooled_hdi.a.sel(hdi = 'lower').values + pooled_hdi.bc.sel(hdi = 'lower').values * temp_dep_c + \
              pooled_hdi.bh.sel(hdi = 'lower').values * temp_dep_h
lower_bound = lower_bound_mean - pooled_hdi.sigma.sel(hdi = 'higher').values
higher_bound_mean = pooled_hdi.a.sel(hdi = 'higher').values + pooled_hdi.bc.sel(hdi = 'higher').values * temp_dep_c + \
              pooled_hdi.bh.sel(hdi = 'higher').values * temp_dep_h
higher_bound = higher_bound_mean + pooled_hdi.sigma.sel(hdi = 'higher').values

plt.scatter(x = df['t'], y = df['log_electricity'], label='observed', s = 10, zorder = 4)
plt.plot(pooled_trace["a"].mean() + pooled_trace["bh"].mean() * temp_dep_h +
                            pooled_trace['bc'].mean() * temp_dep_c, color = 'orangered',label='pooled model', zorder = 3)
plt.vlines(np.arange(n_hours), lower_bound, higher_bound, color = 'bisque', label = 'Exp. distribution', zorder = 1)
plt.vlines(np.arange(n_hours), lower_bound_mean, higher_bound_mean, color = 'darkorange', label = 'Exp. mean HPD',
           alpha = 0.3, zorder = 2)
plt.legend(ncol = 2)
plt.show()

# In the radon notebook the HDI they draw is grouped by 'Level' so we can't use it in this case, might need
# to look into other notebooks to draw the HDI. It's clear that this model can't perform that well being identified
# by a constant term plus a weather dependent term.

# Let's move to another model, the unpooled one (one coefficient per cluster, without a temperature term)

# UNPOOLED MODEL (NO TEMPERATURE TERM)

with pm.Model(coords=coords) as unpooled_model:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    a = pm.Normal("a", mu=0.0, sigma=10.0, dims = "Cluster")

    # Expected value per cluster:
    mu = a[cluster_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")


um_graph = pm.model_to_graphviz(unpooled_model)
um_graph.render(filename='img/unpooled_model')

# Before running the model let’s do some prior predictive checks.

with unpooled_model:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

with unpooled_model:
    unpooled_trace = pm.sample(random_seed=RANDOM_SEED)
    unpooled_idata = az.from_pymc3(unpooled_trace)

# Let's analyse the chains
az.summary(unpooled_idata, round_to=2)

with unpooled_model:
    ppc = pm.sample_posterior_predictive(unpooled_trace, random_seed=RANDOM_SEED)
    unpooled_idata = az.from_pymc3(unpooled_trace, posterior_predictive=ppc, prior=prior_checks)

az.plot_trace(unpooled_idata)
plt.show()

# The chains look good (good R hat, good effective sample size, small sds, good mixing in the traceplot)
# Cluster intercept + HDI visualisation

az.plot_forest(
    unpooled_idata, var_names="a", figsize=(6, 8), r_hat=True, combined=True, textsize=8
)
plt.show()


# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
unpooled_means = unpooled_idata.posterior.mean(dim=("chain", "draw"))

#Calculate mean for each cluster value from the posterior samples
cluster_unpooled_means = np.mean(unpooled_trace['a'], axis =0)
# Create array with predictions
unpooled_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            unpooled_predictions.append(cluster_unpooled_means[cluster_idx])
        else: None

plt.scatter(x = df['t'],y = unpooled_predictions, label='unpooled model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# Let's plot the posterior
with unpooled_model:
    ppc = pm.sample_posterior_predictive(
        unpooled_trace, var_names=["a", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=unpooled_model))
plt.show()

# It looks like this posterior predictive adapts way better to the observed values. But I'm not
# 100% sure of how to interpret this, could ask Sotiris

# UNPOOLED TEMPERATURE MODEL

with pm.Model(coords=coords) as unpooled_temp_model:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")
    a = pm.Normal("a", mu=0.0, sigma=10.0, dims = "Cluster")
    bh = pm.Normal("bh", mu=0.0, sigma=10.0)
    bc = pm.Normal("bc", mu=0.0, sigma=10.0)

    # Expected value per cluster:
    mu = a[cluster_idx] + bh * heating_temp +  bc * cooling_temp
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

# Model graph
umt_graph = pm.model_to_graphviz(unpooled_temp_model)
umt_graph.render(filename='img/unpooled_temp_model')

# Before running the model let’s do some prior predictive checks.

with unpooled_temp_model:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

with unpooled_temp_model:
    unpooled_temp_trace = pm.sample(random_seed=RANDOM_SEED)
    unpooled_temp_idata = az.from_pymc3(unpooled_temp_trace)

# Let's analyse the chains
az.summary(unpooled_temp_idata, round_to=2)

with unpooled_temp_model:
    ppc = pm.sample_posterior_predictive(unpooled_temp_trace, random_seed=RANDOM_SEED)
    unpooled_temp_idata = az.from_pymc3(unpooled_temp_trace, posterior_predictive=ppc, prior=prior_checks)

az.plot_trace(unpooled_temp_idata)
plt.show()

# The chains look good (good R hat, good effective sample size, small sds, good mixing in the traceplot)
# Cluster intercept + HDI visualisation

az.plot_forest(
    unpooled_temp_idata, var_names=["a", "bh", "bc"], figsize=(6, 8), r_hat=True, combined=True, textsize=8
)
plt.show()

# Adding the temperature contributed to a slight shift of the intercepts
# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
#Calculate mean for each cluster value from the posterior samples
cluster_unpooled_temp_means = np.mean(unpooled_temp_trace['a'], axis =0)
unpooled_temp_bh = np.mean(unpooled_temp_trace['bh'])
unpooled_temp_bc = np.mean(unpooled_temp_trace['bc'])
# Create array with predictions
unpooled_temp_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            unpooled_temp_predictions.append(cluster_unpooled_means[cluster_idx] +
                                             unpooled_temp_bh * temp_dep_h[day] +  unpooled_temp_bc * temp_dep_c[day])
        else: None

plt.scatter(x = df['t'],y = unpooled_temp_predictions, label='unpooled model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# Let's plot the posterior
with unpooled_temp_model:
    ppc = pm.sample_posterior_predictive(
        unpooled_temp_trace, var_names=["a", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=unpooled_temp_model))
plt.show()

#Similar plot to the one for the previous case. Not sure how to interpret


# VARYING INTERCEPT MODEL WITHOUT TEMPERATURE SLOPE

with pm.Model(coords=coords) as varying_intercept:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster")

    # Electricity prediction
    mu = a_cluster[cluster_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

# Let's get the graphviz
varying_intercept_graph = pm.model_to_graphviz(varying_intercept)
varying_intercept_graph.render(filename='img/varying_intercept')

# Before running the model let’s do some prior predictive checks.
with varying_intercept:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

with varying_intercept:
    varying_intercept_trace = pm.sample(random_seed=RANDOM_SEED)
    varying_intercept_idata = az.from_pymc3(varying_intercept_trace)

# Let's analyse the chains
az.summary(varying_intercept_idata, round_to=2)

with varying_intercept:
    ppc = pm.sample_posterior_predictive(varying_intercept_trace, random_seed=RANDOM_SEED)
    varying_intercept_idata = az.from_pymc3(varying_intercept_trace, posterior_predictive=ppc, prior=prior_checks)

az.plot_trace(varying_intercept_idata)
plt.show()

# The chains look good (good R hat, good effective sample size, small sds, good mixing in the traceplot)
# Cluster intercept + HDI visualisation

az.plot_forest(
    varying_intercept_idata, var_names="a_cluster", figsize=(6, 8), r_hat=True, combined=True, textsize=8
)
plt.show()

# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

cluster_varying_intercept_means = np.mean(varying_intercept_trace['a_cluster'], axis =0)
# Create array with predictions
varying_intercept_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            varying_intercept_predictions.append(cluster_varying_intercept_means[cluster_idx])
        else: None

plt.scatter(x = df['t'],y = varying_intercept_predictions, label='varying intercept model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# At a first glance it seems that the varying intercept model is improving the estimate, by
# moving the low consumption clusters closer to their observed value, also reducing the consumption of the most extreme clusters
# Let's plot the posterior
with varying_intercept:
    ppc = pm.sample_posterior_predictive(
        varying_intercept_trace, var_names=["a", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=varying_intercept))
plt.show()

#Similar plot to the one for the previous case. Not sure how to interpret, and also not sure what we're actually plotting
# (what is the 'var_names=["a", "y"]' term? Why is it taking 'a'?)


# VARYING INTERCEPT MODEL WITH FIXED TEMPERATURE SLOPE

with pm.Model(coords=coords) as varying_intercept_fixed_temp:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster")

    # Common slopes
    bh = pm.Normal("bh", mu=0.0, sigma=10.0)
    bc = pm.Normal("bc", mu=0.0, sigma=10.0)

    # Electricity prediction
    mu = a_cluster[cluster_idx] + bh * heating_temp + bc * cooling_temp
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

# Let's get the graphviz
varying_temp_graph = pm.model_to_graphviz(varying_intercept_fixed_temp)
varying_temp_graph.render(filename='img/varying_temp')

# Before running the model let’s do some prior predictive checks.
with varying_intercept_fixed_temp:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

with varying_intercept_fixed_temp:
    varying_temp_trace = pm.sample(random_seed=RANDOM_SEED)
    varying_temp_idata = az.from_pymc3(varying_temp_trace)

# Let's analyse the chains
az.summary(varying_temp_idata, round_to=2)

with varying_intercept_fixed_temp:
    ppc = pm.sample_posterior_predictive(varying_temp_trace, random_seed=RANDOM_SEED)
    varying_temp_idata = az.from_pymc3(varying_temp_trace, posterior_predictive=ppc, prior=prior_checks)

az.plot_trace(varying_temp_idata)
plt.show()

# The chains look good (good R hat, good effective sample size, small sds, good mixing in the traceplot)
# Cluster intercept + HDI visualisation

az.plot_forest(
    varying_temp_idata, var_names=["a_cluster", "bh", "bc"], figsize=(6, 8), r_hat=True, combined=True, textsize=8
)
plt.show()

# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

cluster_varying_temp_means = np.mean(varying_temp_trace['a_cluster'], axis =0)
varying_temp_bh = np.mean(varying_temp_trace['bh'])
varying_temp_bc = np.mean(varying_temp_trace['bc'])
# Create array with predictions
varying_temp_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            varying_temp_predictions.append(cluster_varying_temp_means[cluster_idx] +
                                             varying_temp_bh * temp_dep_h[day] +  varying_temp_bc * temp_dep_c[day])
        else: None

plt.scatter(x = df['t'],y = varying_temp_predictions, label='varying intercept temp model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()




# At a first glance it seems that the varying intercept model is improving the estimate, by
# moving the low consumption clusters closer to their observed value, also reducing the consumption of the most extreme clusters
# Let's plot the posterior
with varying_intercept_fixed_temp:
    ppc = pm.sample_posterior_predictive(
        varying_temp_trace, var_names=["a", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=varying_intercept_fixed_temp))
plt.show()

#Similar plot to the one for the previous case. Not sure how to interpret, and also not sure what we're actually plotting
# (what is the 'var_names=["a", "y"]' term? Why is it taking 'a'?)

# VARYING INTERCEPT AND SLOPE MODEL

with pm.Model(coords=coords) as varying_intercept_and_temp:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
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
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster")
    # Varying slopes:
    bh_cluster = pm.Normal("bh_cluster", mu=bh, sigma=sigma_bh, dims="Cluster")
    bc_cluster = pm.Normal("bc_cluster", mu=bc, sigma=sigma_bc, dims="Cluster")


    # Electricity prediction
    mu = a_cluster[cluster_idx] + bh_cluster[cluster_idx] * heating_temp + bc_cluster[cluster_idx] * cooling_temp
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

varying_intercept_and_temp_graph = pm.model_to_graphviz(varying_intercept_and_temp)
varying_intercept_and_temp_graph.render(filename='img/varying_intercept_and_temp')


# Before running the model let’s do some prior predictive checks (not sure what this actually does)
with varying_intercept_and_temp:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

with varying_intercept_and_temp:
    varying_intercept_slope_trace = pm.sample(random_seed=RANDOM_SEED)
    varying_temp_idata = az.from_pymc3(varying_intercept_slope_trace)

# Let's analyse the chains
az.summary(varying_temp_idata, round_to=2)

with varying_intercept_and_temp:
    ppc = pm.sample_posterior_predictive(varying_intercept_slope_trace, random_seed=RANDOM_SEED)
    varying_intercept_slope_idata = az.from_pymc3(varying_intercept_slope_trace, posterior_predictive=ppc, prior=prior_checks)

az.plot_trace(varying_intercept_slope_idata)
plt.show()

# Here we get some sampling errors, low effective samples number, as well as bad traceplots
# Might be due to the fact that in some clusters there's no cooling/heating, so in those
# clusters it's hard to estimate the coefficients
# Cluster intercept + HDI visualisation

az.plot_forest(
    varying_intercept_slope_idata, var_names=["a_cluster", "bh_cluster", "bc_cluster"], figsize=(6, 8), r_hat=True, combined=True, textsize=8
)
plt.show()

# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

cluster_varying_intercept_slope_means = np.mean(varying_intercept_slope_trace['a_cluster'], axis =0)
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

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            varying_intercept_slope_predictions.append(cluster_varying_intercept_slope_means[cluster_idx] +
                                                       varying_intercept_slope_bh[cluster_idx] * temp_dep_h[day] +
                                                       varying_intercept_slope_bc[cluster_idx] * temp_dep_c[day])

            varying_intercept_slope_mean_lower.append(varying_intercept_slope_hdi['a_cluster'][cluster_idx].sel(hdi = 'lower') +
                                                       varying_intercept_slope_hdi['bh_cluster'][cluster_idx].sel(
                                                           hdi='lower') * temp_dep_h[day] +
                                                       varying_intercept_slope_hdi['bc_cluster'][cluster_idx].sel(
                                                           hdi='lower') * temp_dep_c[day])

            varying_intercept_slope_mean_higher.append(varying_intercept_slope_hdi['a_cluster'][cluster_idx].sel(hdi = 'higher') +
                                                       varying_intercept_slope_hdi['bh_cluster'][cluster_idx].sel(
                                                           hdi='higher') * temp_dep_h[day] +
                                                       varying_intercept_slope_hdi['bc_cluster'][cluster_idx].sel(
                                                           hdi='higher') * temp_dep_c[day])

            varying_intercept_slope_lower.append( varying_intercept_slope_hdi['a_cluster'][cluster_idx].sel(hdi='lower') +
                varying_intercept_slope_hdi['bh_cluster'][cluster_idx].sel(hdi='lower') * temp_dep_h[day] +
                varying_intercept_slope_hdi['bc_cluster'][cluster_idx].sel(hdi='lower') * temp_dep_c[day] -
                                                  varying_intercept_slope_hdi['sigma'].sel(hdi='higher'))


            varying_intercept_slope_higher.append( varying_intercept_slope_hdi['a_cluster'][cluster_idx].sel(hdi='higher') +
                varying_intercept_slope_hdi['bh_cluster'][cluster_idx].sel(hdi='higher') * temp_dep_h[day] +
                varying_intercept_slope_hdi['bc_cluster'][cluster_idx].sel(hdi='higher') * temp_dep_c[day]+
                                                  varying_intercept_slope_hdi['sigma'].sel(hdi='higher'))

plt.scatter(x = df['t'],y = varying_intercept_slope_predictions, label='varying intercept temp model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# Plot HDI
plt.scatter(x = df[0:365]['t'], y = df[0:365]['log_electricity'], label='Observed', s = 10, zorder = 4)
plt.scatter(x = df[0:365]['t'], y = varying_intercept_slope_predictions[0:365], color = 'orangered',
            label='Varying intercept and slope model', zorder = 3, s = 14)
vlines = plt.vlines(np.arange(365), varying_intercept_slope_mean_lower[0:365], varying_intercept_slope_mean_higher[0:365],
                    color='darkorange', label='Exp. distribution', zorder=2)
vlines = plt.vlines(np.arange(365), varying_intercept_slope_lower[0:365], varying_intercept_slope_higher[0:365],
                    color='bisque', label='Exp. mean HPD', zorder=1)
plt.legend(ncol = 2)
plt.show()

# Let's plot the posterior
with varying_intercept_and_temp:
    ppc = pm.sample_posterior_predictive(
        varying_intercept_slope_trace, var_names=["a", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=varying_intercept_and_temp))
plt.show()

# How to interpret? Also not sure what we're actually plotting (what is the 'var_names=["a", "y"]' term? Why is it taking 'a'?)

# VARYING INTERCEPT, TEMPERATURE AND GHI SLOPE

with pm.Model(coords=coords) as varying_intercept_temp_GHI:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")
    horizontal_irradiance = pm.Data("horizontal_irradiance", GHI, dims = "obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    bh = pm.Normal("bh", mu=0.0, sigma=1.0)
    sigma_bh = pm.Exponential("sigma_bh", 0.5)
    bc = pm.Normal("bc", mu=0.0, sigma=1.0)
    sigma_bc = pm.Exponential("sigma_bc", 0.5)
    bghi = pm.Normal("bghi", mu=0.0, sigma=1.0)
    sigma_bghi = pm.Exponential("sigma_bghi", 0.5)

    # Varying intercepts:
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster")
    # Varying slopes:
    bh_cluster = pm.Normal("bh_cluster", mu=bh, sigma=sigma_bh, dims="Cluster")
    bc_cluster = pm.Normal("bc_cluster", mu=bc, sigma=sigma_bc, dims="Cluster")
    bghi_cluster = pm.Normal("bghi_cluster", mu=bghi, sigma=sigma_bghi, dims="Cluster")


    # Electricity prediction
    mu = a_cluster[cluster_idx] + bh_cluster[cluster_idx] * heating_temp + bc_cluster[cluster_idx] * cooling_temp + bghi_cluster[cluster_idx] * horizontal_irradiance
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

varying_intercept_temp_GHI_graph = pm.model_to_graphviz(varying_intercept_temp_GHI)
varying_intercept_temp_GHI_graph.render(filename='img/varying_intercept_temp_GHI')


# Before running the model let’s do some prior predictive checks (not sure what this actually does)
with varying_intercept_temp_GHI:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

with varying_intercept_temp_GHI:
    varying_intercept_temp_GHI_trace = pm.sample(random_seed=RANDOM_SEED)
    varying_intercept_temp_GHI_idata = az.from_pymc3(varying_intercept_temp_GHI_trace)

# We get sampling warnings, might need to try the uncentered version
# Let's analyse the chains
az.summary(varying_intercept_temp_GHI_idata, round_to=2)

with varying_intercept_temp_GHI:
    ppc = pm.sample_posterior_predictive(varying_intercept_temp_GHI_trace, random_seed=RANDOM_SEED)
    varying_intercept_temp_GHI_idata = az.from_pymc3(varying_intercept_temp_GHI_trace, posterior_predictive=ppc, prior=prior_checks)

az.plot_trace(varying_intercept_temp_GHI_idata)
plt.show()

# Here we get some sampling errors, low effective samples number, as well as bad traceplots
# Might be due to the fact that in some clusters there's no cooling/heating, so in those
# clusters it's hard to estimate the coefficients
# Cluster intercept + HDI visualisation

az.plot_forest(
    varying_intercept_temp_GHI_idata, var_names=["a_cluster", "bh_cluster", "bc_cluster", "bghi_cluster"], figsize=(6, 8), r_hat=True, combined=True, textsize=8
)
plt.show()

# Let's sample from the posterior to plot the predictions and have a rough estimate of the model accuracy
# Calculate mean for each cluster value from the posterior samples

cluster_varying_intercept_temp_GHI_means = np.mean(varying_intercept_temp_GHI_trace['a_cluster'], axis =0)
varying_intercept_slope_bh = np.mean(varying_intercept_temp_GHI_trace['bh_cluster'], axis = 0)
varying_intercept_slope_bc = np.mean(varying_intercept_temp_GHI_trace['bc_cluster'], axis = 0)
varying_intercept_slope_bghi = np.mean(varying_intercept_temp_GHI_trace['bghi_cluster'], axis = 0)
# Create array with predictions
varying_intercept_temp_GHI_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            varying_intercept_temp_GHI_predictions.append(cluster_varying_intercept_temp_GHI_means[cluster_idx] +
                                                       varying_intercept_slope_bh[cluster_idx] * temp_dep_h[day] +
                                                       varying_intercept_slope_bc[cluster_idx] * temp_dep_c[day] +
                                                       varying_intercept_slope_bghi[cluster_idx] * GHI[day])

plt.scatter(x = df['t'],y = varying_intercept_temp_GHI_predictions, label='varying intercept temp GHI model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# At a first glance adding the GHI as a predictor helps in creating more variance in the low consumption cluster
# (which in the varying intercept and temp model is pretty flat
# Let's plot the posterior
with varying_intercept_temp_GHI:
    ppc = pm.sample_posterior_predictive(
        varying_intercept_slope_trace, var_names=["a", "y"], random_seed=RANDOM_SEED
    )

# Now, ppc contains 4000 generated data sets (containing 100 samples each), each using a different
# parameter setting from the posterior.
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=varying_intercept_temp_GHI))
plt.show()

# How to interpret? Also not sure what we're actually plotting (what is the 'var_names=["a", "y"]' term? Why is it taking 'a'?)

# COVARYING INTERCEPT AND 1 TEMPERATURE SLOPE

coords["param"] = ["a", "b"]
coords["param_bis"] = ["a", "b"]
with pm.Model(coords=coords) as covariation_intercept_temp:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    temp_dev = pm.Data("temp_dev", temperature_deviation, dims="obs_id")

    # prior stddev in temp slope
    sd_dist = pm.Exponential.dist(0.5)

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)

    # prior for average intercept:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    # prior for average slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # population of varying effects:
    ab_cluster = pm.MvNormal("ab_cluster", mu=tt.stack([a, b]), chol=chol, dims=("Cluster", "param"))

    # Electricity prediction
    mu = ab_cluster[cluster_idx, 0] + ab_cluster[cluster_idx, 1] * temp_dev
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

pm.model_to_graphviz(covariation_intercept_temp)

with covariation_intercept_temp:
    covarying_intercept_temp_trace = pm.sample(random_seed=RANDOM_SEED)
    covarying_intercept_temp_idata = az.from_pymc3(covarying_intercept_temp_trace)

az.summary(covarying_intercept_temp_idata, round_to=2)

with covariation_intercept_temp:
    ppc = pm.sample_posterior_predictive(covarying_intercept_temp_trace, random_seed=RANDOM_SEED)
    covarying_intercept_temp_idata = az.from_pymc3(covarying_intercept_temp_trace, posterior_predictive=ppc)

az.plot_trace(covarying_intercept_temp_idata)
plt.show()

# Let's try to compute predictions (actually retrodictions)

covarying_int_temp_ab_means = np.mean(covarying_intercept_temp_trace['ab_cluster'], axis = 0)
# Create array with predictions
covarying_intercept_temp_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            covarying_intercept_temp_predictions.append(covarying_int_temp_ab_means[cluster_idx,0] +
                                                        covarying_int_temp_ab_means[cluster_idx,1] * temperature_deviation[day])


plt.scatter(x = df['t'],y = covarying_intercept_temp_predictions, label='covarying intercept temperature model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()


# COVARYING INTERCEPT AND 1 TEMPERATURE SLOPE (UNCENTERED)

coords["param"] = ["a", "b"]
coords["param_bis"] = ["a", "b"]
with pm.Model(coords=coords) as covariation_intercept_temp_uc:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    temp_dev = pm.Data("temp_dev", temperature_deviation, dims="obs_id")

    # prior stddev in temp slope
    sd_dist = pm.Exponential.dist(0.5)

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)

    # prior for average intercept:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    # prior for average slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    # population of varying effects:
    z = pm.Normal("z", 0.0, 1.0, dims=("param", "Cluster"))
    ab_cluster = pm.Deterministic("ab_cluster", tt.dot(chol, z).T, dims=("Cluster", "param"))

    # Electricity prediction
    mu = a + ab_cluster[cluster_idx, 0] + (b + ab_cluster[cluster_idx, 1]) * temp_dev
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

pm.model_to_graphviz(covariation_intercept_temp_uc)

with covariation_intercept_temp_uc:
    covarying_intercept_temp_uc_trace = pm.sample(2000, tune = 2000, target_accept = 0.99, random_seed=RANDOM_SEED)

with covariation_intercept_temp_uc:
    covarying_intercept_temp_uc_idata = az.from_pymc3(covarying_intercept_temp_uc_trace)

az.summary(covarying_intercept_temp_uc_idata, round_to=2)

az.plot_trace(covarying_intercept_temp_uc_idata)
plt.show()

az.plot_trace(
    covarying_intercept_temp_uc_idata,
    var_names=["~z", "~chol"],
    lines=[("chol_corr", {}, 0.0)],
    compact=True,
    chain_prop={"ls": "-"},
)
plt.show()

# How to interpret this graph?

# Let's try to compute predictions (actually retrodictions)

covarying_int_temp_uc_a_means = np.mean(covarying_intercept_temp_uc_trace['a'])
covarying_int_temp_uc_b_means = np.mean(covarying_intercept_temp_uc_trace['b'])

covarying_int_temp_uc_ab_means = np.mean(covarying_intercept_temp_uc_trace['ab_cluster'], axis = 0)
# Create array with predictions
covarying_intercept_temp_uc_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            covarying_intercept_temp_uc_predictions.append(covarying_int_temp_uc_a_means + covarying_int_temp_uc_ab_means[cluster_idx,0] +
                                                           (covarying_int_temp_uc_b_means +
                                                            covarying_int_temp_uc_ab_means[cluster_idx,1] * temperature_deviation[day]))


plt.scatter(x = df['t'],y = covarying_intercept_temp_uc_predictions, label='covarying intercept temperature model')
plt.scatter(x = df['t'], y = df['log_electricity'], label='observed')
plt.legend(loc='upper left')
plt.show()

# The chains look good, except for one of the chol.

# VARYING INTERCEPT, COVARYING TEMPERATURE AND GHI SLOPE

coords["param"] = ["bt", "bghi"]
coords["param_bis"] = ["bt", "bghi"]
with pm.Model(coords=coords) as covariation_temp_GHI_slopes:
    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    temp_dev = pm.Data("temp_dev", temperature_deviation, dims="obs_id")
    horizontal_irradiance = pm.Data("horizontal_irradiance", GHI, dims="obs_id")

    # prior stddev in temp and GHI slopes
    sd_dist = pm.Exponential.dist(0.5)

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)

    #Hyperprior for varying intercept
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster")

    # prior for average temp slope:
    btemp = pm.Normal("btemp", mu=0.0, sigma=1.0)
    # prior for average GHI slope:
    bghi = pm.Normal("bghi", mu=0.0, sigma=1.0)

    # population of varying effects:
    b1b2_cluster = pm.MvNormal("b1b2_cluster", mu=tt.stack([btemp, bghi]), chol=chol, dims=("Cluster", "param"))

    # Expected value per county:
    mu = a_cluster[cluster_idx] + b1b2_cluster[cluster_idx, 0] * temp_dev + b1b2_cluster[cluster_idx, 1] * horizontal_irradiance
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

pm.model_to_graphviz(covariation_temp_GHI_slopes)

with covariation_temp_GHI_slopes:
    covarying_temp_ghi_slopes_trace = pm.sample(random_seed=RANDOM_SEED)
    covarying_temp_ghi_slopes_idata = az.from_pymc3(covarying_slopes_trace)

az.summary(covarying_temp_ghi_slopes_idata, round_to=2)

with covariation_intercept_temp:
    ppc = pm.sample_posterior_predictive(covarying_temp_ghi_slopes_trace, random_seed=RANDOM_SEED)
    covarying_temp_ghi_slopes_idata = az.from_pymc3(covarying_temp_ghi_slopes_trace, posterior_predictive=ppc)

az.plot_trace(covarying_temp_ghi_slopes_idata)
plt.show()

# Let's try to compute predictions (actually retrodictions)

covarying_int_temp_ab_means = np.mean(covarying_intercept_temp_trace['ab_county'], axis = 0)
# Create array with predictions
covarying_intercept_temp_predictions = []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            covarying_intercept_temp_predictions.append(covarying_int_temp_ab_means[cluster_idx,0] +
                                                        covarying_int_temp_ab_means[cluster_idx,1] * temperature_deviation[day])



# COVARYING INTERCEPT AND 2 TEMPERATURE SLOPES

with pm.Model(coords=coords) as covarying_intercept_and_temp:

    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")

   # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    bh = pm.Normal("bh", mu=0.0, sigma=1.0)
    bc = pm.Normal("bc", mu=0.0, sigma=1.0)

    sd_dist = pm.Exponential.dist(0.5)

    chol, corr, stds = pm.LKJCholeskyCov("chol", n=3, eta=2.0, sd_dist=sd_dist, compute_corr=True)

   # Correlated varying intercept and slopes within clusters
   # Note that I don't really know how to use the dims argument, but you could replace
   # the shape bit with something using that I'm sure.
    coefs = pm.MvNormal("coefficients", mu=tt.stack([a, bh, bc]), chol=chol, shape=(n_clusters, 3))

   # You can now pick out the a, bh and bc if you like and they should be correlated within clusters
    a_cluster = coefs[:, 0]
    bh_cluster = coefs[:, 1]
    bc_cluster = coefs[:, 2]

   # Electricity prediction
    mu = a_cluster[cluster_idx] + bh_cluster[cluster_idx] * heating_temp + bc_cluster[cluster_idx] * cooling_temp

   # Model error:
    sigma = pm.Exponential("sigma", 1.0)
    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")

pm.model_to_graphviz(covarying_intercept_and_temp)

with covarying_intercept_and_temp:
    covarying_slopes_trace = pm.sample(2000, tune = 2000, target_accept = 0.99, random_seed=RANDOM_SEED, cores=4)
with covarying_intercept_and_temp:
    covarying_slopes_idata = az.from_pymc3(covarying_slopes_trace)

az.summary(covarying_slopes_idata)
az.plot_trace(covarying_slopes_idata)
plt.show()

# Let's try to compute predictions (actually retrodictions)

covarying_coefficients_means = np.mean(covarying_slopes_trace['coefficients'], axis = 0)
# Create array with predictions

covarying_slopes_hdi = az.hdi(covarying_slopes_idata)
covarying_slopes_predictions = []
covarying_slopes_mean_lower = []
covarying_slopes_mean_higher= []
covarying_slopes_lower = []
covarying_slopes_higher= []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            covarying_slopes_predictions.append(covarying_coefficients_means[cluster_idx,0] +
                                                        covarying_coefficients_means[cluster_idx,1] * temp_dep_h[day] +
                                                        covarying_coefficients_means[cluster_idx, 2] * temp_dep_c[day])

            covarying_slopes_mean_lower.append(covarying_slopes_hdi['coefficients'][cluster_idx][0].sel(hdi = 'lower') +
                                                      covarying_slopes_hdi['coefficients'][cluster_idx][1].sel(hdi='lower') *
                                                      temp_dep_h[day] +
                                                      covarying_slopes_hdi['coefficients'][cluster_idx][2].sel(
                                                          hdi='lower') * temp_dep_c[day])

            covarying_slopes_mean_higher.append(covarying_slopes_hdi['coefficients'][cluster_idx][0].sel(hdi = 'higher') +
                                                      covarying_slopes_hdi['coefficients'][cluster_idx][1].sel(hdi='higher') *
                                                      temp_dep_h[day] +
                                                      covarying_slopes_hdi['coefficients'][cluster_idx][2].sel(
                                                          hdi='higher') * temp_dep_c[day])

            covarying_slopes_lower.append(covarying_slopes_hdi['coefficients'][cluster_idx][0].sel(hdi = 'lower') +
                                           covarying_slopes_hdi['coefficients'][cluster_idx][1].sel(hdi='lower') *
                                           temp_dep_h[day] +
                                           covarying_slopes_hdi['coefficients'][cluster_idx][2].sel(hdi='lower') *
                                           temp_dep_c[day] -
                                           covarying_slopes_hdi['sigma'].sel(hdi='higher'))

            covarying_slopes_higher.append(covarying_slopes_hdi['coefficients'][cluster_idx][0].sel(hdi = 'higher') +
                                           covarying_slopes_hdi['coefficients'][cluster_idx][1].sel(hdi='higher') *
                                           temp_dep_h[day] +
                                           covarying_slopes_hdi['coefficients'][cluster_idx][2].sel(hdi='higher') *
                                           temp_dep_c[day]+
                                           covarying_slopes_hdi['sigma'].sel(hdi='higher'))


# Plot HDI
plt.scatter(x = df[0:365]['t'], y = df[0:365]['log_electricity'], label='Observed', s = 10, zorder = 4)
plt.scatter(x = df[0:365]['t'], y = covarying_slopes_predictions[0:365], color = 'orangered',
            label='Varying intercept and slope model', zorder = 3, s = 14)
vlines = plt.vlines(df[0:365]['t'], covarying_slopes_mean_lower[0:365], covarying_slopes_mean_higher[0:365],
                    color='darkorange', label='Exp. distribution', zorder=2)
vlines = plt.vlines(df[0:365]['t'], covarying_slopes_lower[0:365], covarying_slopes_higher[0:365],
                    color='bisque', label='Exp. mean HPD', zorder=1)
plt.legend(ncol = 2)
plt.show()

# COVARYING INTERCEPT TEMPERATURE AND GHI

with pm.Model(coords=coords) as covarying_intercept_temp_ghi:

    cluster_idx = pm.Data("cluster_idx", cluster, dims="obs_id")
    heating_temp = pm.Data("heating_temp", temp_dep_h, dims="obs_id")
    cooling_temp = pm.Data("cooling_temp", temp_dep_c, dims="obs_id")
    horizontal_irradiance = pm.Data("horizontal_irradiance", GHI, dims = "obs_id")

   # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    bh = pm.Normal("bh", mu=0.0, sigma=1.0)
    bc = pm.Normal("bc", mu=0.0, sigma=1.0)
    bghi = pm.Normal("bghi", mu=0.0, sigma=1.0)

    sd_dist = pm.Exponential.dist(0.5)

    chol, corr, stds = pm.LKJCholeskyCov("chol", n=4, eta=2.0, sd_dist=sd_dist, compute_corr=True)

   # Correlated varying intercept and slopes within clusters
   # Note that I don't really know how to use the dims argument, but you could replace
   # the shape bit with something using that I'm sure.
    coefs = pm.MvNormal("coefficients", mu=tt.stack([a, bh, bc, bghi]), chol=chol, shape=(n_clusters, 4))

   # You can now pick out the a, bh and bc if you like and they should be correlated within clusters
    a_cluster = coefs[:, 0]
    bh_cluster = coefs[:, 1]
    bc_cluster = coefs[:, 2]
    bghi_cluster = coefs[:, 3]

   # Electricity prediction
    mu = a_cluster[cluster_idx] + bh_cluster[cluster_idx] * heating_temp + bc_cluster[cluster_idx] * cooling_temp + \
         bghi_cluster[cluster_idx] * horizontal_irradiance

   # Model error:
    sigma = pm.Exponential("sigma", 1.0)
    y = pm.Normal("y", mu, sigma=sigma, observed=log_electricity, dims="obs_id")


with covarying_intercept_temp_ghi:
    covarying_slopes_ghi_trace = pm.sample(2000, tune = 2000, target_accept = 0.99, random_seed=RANDOM_SEED, cores=4)
with covarying_intercept_temp_ghi:
    covarying_slopes_ghi_idata = az.from_pymc3(covarying_slopes_ghi_trace)


az.summary(covarying_slopes_ghi_idata)
az.plot_trace(covarying_slopes_ghi_idata)
plt.show()

# Let's try to compute predictions (actually retrodictions)

covarying_coefficients_ghi_means = np.mean(covarying_slopes_ghi_trace['coefficients'], axis = 0)
# Create array with predictions

covarying_slopes_ghi_hdi = az.hdi(covarying_slopes_ghi_idata)
covarying_slopes_ghi_predictions = []
covarying_slopes_ghi_mean_lower = []
covarying_slopes_ghi_mean_higher= []
covarying_slopes_ghi_lower = []
covarying_slopes_ghi_higher= []

for day in range(0,len(df)):
    for cluster_idx in unique_clusters:
        if cluster[day] == cluster_idx:
            covarying_slopes_ghi_predictions.append(covarying_coefficients_ghi_means[cluster_idx,0] +
                                                    covarying_coefficients_ghi_means[cluster_idx,1] * temp_dep_h[day] +
                                                    covarying_coefficients_ghi_means[cluster_idx, 2] * temp_dep_c[day] +
                                                    covarying_coefficients_ghi_means[cluster_idx, 3] * GHI[day])

            covarying_slopes_ghi_mean_lower.append(covarying_slopes_ghi_hdi['coefficients'][cluster_idx][0].sel(hdi = 'lower') +
                                                      covarying_slopes_ghi_hdi['coefficients'][cluster_idx][1].sel(hdi='lower') *
                                                      temp_dep_h[day] +
                                                      covarying_slopes_ghi_hdi['coefficients'][cluster_idx][2].sel(
                                                          hdi='lower') * temp_dep_c[day] +
                                                   covarying_slopes_ghi_hdi['coefficients'][cluster_idx][3].sel(
                                                       hdi='lower') * GHI[day]
                                                   )

            covarying_slopes_ghi_mean_higher.append(covarying_slopes_ghi_hdi['coefficients'][cluster_idx][0].sel(hdi = 'higher') +
                                                      covarying_slopes_ghi_hdi['coefficients'][cluster_idx][1].sel(hdi='higher') *
                                                      temp_dep_h[day] +
                                                      covarying_slopes_ghi_hdi['coefficients'][cluster_idx][2].sel(
                                                          hdi='higher') * temp_dep_c[day]+
                                                   covarying_slopes_ghi_hdi['coefficients'][cluster_idx][3].sel(
                                                       hdi='higher') * GHI[day]
                                                   )

            covarying_slopes_ghi_lower.append(covarying_slopes_ghi_hdi['coefficients'][cluster_idx][0].sel(hdi = 'lower') +
                                           covarying_slopes_ghi_hdi['coefficients'][cluster_idx][1].sel(hdi='lower') *
                                           temp_dep_h[day] +
                                           covarying_slopes_ghi_hdi['coefficients'][cluster_idx][2].sel(hdi='lower') *
                                           temp_dep_c[day] + covarying_slopes_ghi_hdi['coefficients'][cluster_idx][3].sel(hdi='lower') *
                                           GHI[day] - covarying_slopes_ghi_hdi['sigma'].sel(hdi='higher'))

            covarying_slopes_ghi_higher.append(covarying_slopes_ghi_hdi['coefficients'][cluster_idx][0].sel(hdi = 'higher') +
                                           covarying_slopes_ghi_hdi['coefficients'][cluster_idx][1].sel(hdi='higher') *
                                           temp_dep_h[day] +
                                           covarying_slopes_ghi_hdi['coefficients'][cluster_idx][2].sel(hdi='higher') *
                                           temp_dep_c[day]+
                                           covarying_slopes_ghi_hdi['coefficients'][cluster_idx][3].sel(hdi='higher') *
                                           GHI[day] +
                                           covarying_slopes_ghi_hdi['sigma'].sel(hdi='higher'))


# Plot HDI
plt.scatter(x = df[0:365]['t'], y = df[0:365]['log_electricity'], label='Observed', s = 10, zorder = 4)
plt.scatter(x = df[0:365]['t'], y = covarying_slopes_ghi_predictions[0:365], color = 'orangered',
            label='Covarying intercept temp GHI slopes model', zorder = 3, s = 14)
vlines = plt.vlines(df[0:365]['t'], covarying_slopes_ghi_mean_lower[0:365], covarying_slopes_ghi_mean_higher[0:365],
                    color='darkorange', label='Exp. mean HPD', zorder=2)
vlines = plt.vlines(df[0:365]['t'], covarying_slopes_ghi_lower[0:365], covarying_slopes_ghi_higher[0:365],
                    color='bisque', label='Exp. distribution', zorder=1)
plt.legend(ncol = 2)
plt.show()



# Let's run model comparison (leave one out cross validation)
df_comp_loo = az.compare({ "M1": pooled_trace, "M2":unpooled_trace, "M3": unpooled_temp_trace,
                           "M4": varying_intercept_trace, "M5":varying_temp_trace,
                           "M6":varying_intercept_slope_trace,
                           "M7": varying_intercept_temp_GHI_trace,
                           "M8": covarying_slopes_trace,
                           "M9":covarying_slopes_ghi_trace})

#Pooled M1, Unpooled M2, Unpooled + temp M3, varying intercept M4, varying int + fix slope M5,
# varying int + varying temp slopes M6, varying intercept + varying temp and GHI slopes M7, covarying temp slopes M8,
# covarying temp ghi slopes M9

df_comp_loo
az.plot_compare(df_comp_loo, insample_dev=False)
plt.show()