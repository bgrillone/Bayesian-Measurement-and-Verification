import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings

df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/hourly_multilevel_office.csv")

# Let's follow the radon notebook and implement step by step the different models


# We'll try to implement a model with covarying intercept and temperature slope depending on the
# cluster_hour variable, that includes information about the hour of the day and the consumption profile

# We'll start with the centered version, to then move to the non-centered version for more efficient sampling

# We  need a lookup table (dict) for each unique cluster_hour, for indexing.

df.cluster_hour= df.cluster_hour.map(str.strip)
cluster_hours = df.cluster_hour.unique()
n_cluster_hour = len(cluster_hours)
cluster_hour_lookup = dict(zip(cluster_hours, range(n_cluster_hour)))

# Finally, create local copies of variables.

cluster_hour = df["cluster_hour_index"] = df.cluster_hour.replace(cluster_hour_lookup).values
# Create local copies of variables.

electricity = df.total_electricity
df["log_electricity"] = log_electricity = np.log(electricity + 0.1).values
temperature = df.outdoor_temp

# Distribution of electricity levels (log scale):
df.log_electricity.hist(bins=25);
plt.show()

# Neither the data, nor it's log looks in any way similar to normal (on hourly level at least)
# Is this going to be a problem?

# DAILY PARTIAL POOLING MODEL 1: varying intercept by cluster + fixed slope for temperature

# HOURLY PARTIAL POOLING MODEL

coords = {"obs_id": np.arange(temperature.size)}
coords["Cluster_hour"] = cluster_hour


with pm.Model(coords=coords) as partial_pooling:
    cluster_idx = pm.Data("county_idx", cluster_hour, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_cluster = pm.Normal("a_cluster", mu=a, sigma=sigma_a, dims="Cluster_hour")

    # Expected value per county:
    mu = a_cluster[cluster_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=electricity, dims="obs_id")

dot = pm.model_to_graphviz(partial_pooling)

dot.render('partial_pooled.gv', view=True)

with partial_pooling:
    partial_pooling_idata = pm.sample(tune=2000, return_inferencedata=True)

az.plot_trace(partial_pooling_idata);
az.plot_posterior(partial_pooling_idata);

plt.show()
plt.savefig('post.pdf')
