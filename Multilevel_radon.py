import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings

from theano import tensor as tt

print(f"Running on PyMC3 v{pm.__version__}")
warnings.simplefilter(action="ignore", category=FutureWarning)
RANDOM_SEED = 8924
np.random.seed(286)

# We'll be using the Gelman radon dataset, in which we try to predict the amount of radon gas
# present in an household depending on the location and presence of a basement
# We'll try to predict radon levels in Minnesota.

az.style.use("arviz-darkgrid")
# Import radon data
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()

# Next, obtain the county-level predictor, uranium, by combining two variables.

srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
cty = pd.read_csv(pm.get_data("cty.dat"))
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

# Use the merge method to combine home- and county-level information in a single DataFrame.

srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
u = np.log(srrs_mn.Uppm).unique()

n = len(srrs_mn)
srrs_mn.head()

# We also need a lookup table (dict) for each unique county, for indexing.

srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(counties)))

# Finally, create local copies of variables.

county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
floor = srrs_mn.floor.values

# Distribution of radon levels in MN (log scale):
srrs_mn.log_radon.hist(bins=25);
plt.show()

# The two conventional alternatives to modeling radon exposure represent the two extremes of the bias-variance tradeoff:
# Complete pooling: Treat all counties the same, and estimate a single radon level.
# No pooling: Model radon in each county independently.

# We’ll start by estimating the slope and intercept for the complete pooling model.


coords = {"Level": ["Basement", "Floor"], "obs_id": np.arange(floor.size)}

with pm.Model(coords=coords) as pooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    a = pm.Normal("a", 0.0, sigma=10.0, dims="Level")

    mu = a[floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_radon, dims="obs_id")

pm.model_to_graphviz(pooled_model)

# Before running the model let’s do some prior predictive checks.


with pooled_model:
    prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    idata_prior = az.from_pymc3(prior=prior_checks)

# We want scatter plot of the mean log radon level prior (which is stored in variable a)
# for each of the two levels we are considering.
_, ax = plt.subplots()
idata_prior.prior.plot.scatter(x="Level", y="a", color="k", alpha=0.2, ax=ax)
ax.set_ylabel("Mean log radon level");
plt.show()

# Before seing the data, these priors seem to allow for quite a wide range of the mean log radon level
# We can always change these priors if sampling gives us hints that they might not be appropriate
# Let's now run the model

with pooled_model:
    pooled_trace = pm.sample(random_seed=RANDOM_SEED)
    pooled_idata = az.from_pymc3(pooled_trace)

az.summary(pooled_idata, round_to=2)
# Here the chains look very good (good R hat, good effective sample size, small sd)

# To estimate the uncertainty around the household radon levels (not the average level, but measurements
# that would be likely in households), we need to sample the likelihood y from the model. In another words,
# we need to do posterior predictive checks

with pooled_model:
    ppc = pm.sample_posterior_predictive(pooled_trace, random_seed=RANDOM_SEED)
    pooled_idata = az.from_pymc3(pooled_trace, posterior_predictive=ppc, prior=prior_checks)

# We have now converted our trace and posterior predictive samples into an arviz.InferenceData object.
# InferenceData is specifically designed to centralize all the relevant quantities of a Bayesian inference
# workflow into a single object.

pooled_idata

# We now want to calculate the highest density interval given by the posterior predictive on Radon levels.
# However, we are not interested in the HDI of each observation but in the HDI of each level.

# We want ArviZ to reduce all dimensions in each groupby group. Here, each groupby will have the same 3 dimensions
# as the original input (chain, draw, obs_id) what will change is the length of the obs_id dimension, in the first group
# it will be the number of basement level observations and in the second the number of floor level observations.
# In az.hdi, the dimensions to be reduced can be specified with the input_core_dims argument.


hdi_helper = lambda ds: az.hdi(ds, input_core_dims=[["chain", "draw", "obs_id"]])
hdi_ppc = pooled_idata.posterior_predictive.y.groupby(pooled_idata.constant_data.floor_idx).apply(hdi_helper).y
hdi_ppc

# We will now add one extra coordinate to the observed_data group: the Level labels (not indices). This will allow
# xarray to automatically generate the correct xlabel and xticklabels so we don’t have to worry about labeling too much.
# Then we sort by Level to make sure Basement is the first value and goes at the left of the plot

level_labels = pooled_idata.posterior.Level[pooled_idata.constant_data.floor_idx]
pooled_idata.observed_data = pooled_idata.observed_data.assign_coords(Level=level_labels).sortby("Level")

pooled_means = pooled_idata.posterior.mean(dim=("chain", "draw"))

_, ax = plt.subplots()
pooled_idata.observed_data.plot.scatter(x="Level", y="y", label="Observations", alpha=0.4, ax=ax)

az.plot_hdi(
    [0, 1], hdi_data=hdi_ppc, fill_kwargs={"alpha": 0.2, "label": "Exp. distrib. of Radon levels"}, ax=ax
)

az.plot_hdi(
    [0, 1], pooled_idata.posterior.a, fill_kwargs={"alpha": 0.5, "label": "Exp. mean HPD"}, ax=ax
)
ax.plot([0, 1], pooled_means.a, label="Exp. mean")

ax.set_ylabel("Log radon level")
ax.legend(ncol=2, fontsize=9, frameon=True);
plt.show()

# This graph represents the model prediction for radon levels depending on whether the measurement was collected on
# the basement or on the first floor. The 94% interval of the expected value is very narrow, and even narrower for
# basement measurements, meaning that the model is slightly more confident about these observations.
# We can infer that floor level does account for some of the variation in radon levels.
# We can see however that the model underestimates the dispersion in radon levels across households

# Let’s compare it to the unpooled model, where we estimate the radon level for each county:


coords["County"] = mn_counties
with pm.Model(coords=coords) as unpooled_model:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    a = pm.Normal("a", 0.0, sigma=10.0, dims=("County", "Level"))

    mu = a[county_idx, floor_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_radon, dims="obs_id")

pm.model_to_graphviz(unpooled_model)

with unpooled_model:
    unpooled_idata = pm.sample(return_inferencedata=True, random_seed=RANDOM_SEED)

az.plot_forest(
    unpooled_idata, var_names="a", figsize=(6, 32), r_hat=True, combined=True, textsize=8
);
plt.show()

# Sampling was good for all counties, but you can see that some are more uncertain than others,
# and all of these uncertain estimates are for floor measurements. This probably comes from the
# fact that some counties just have a handful of floor measurements, so the model is pretty uncertain about them.
#To identify counties with high radon levels, we can plot the ordered mean estimates, as well as their 94% HPD:

unpooled_means = unpooled_idata.posterior.mean(dim=("chain", "draw"))
unpooled_hdi = az.hdi(unpooled_idata)

# We will now take advantage of label based indexing for Datasets with the sel method and of automatical sorting
# capabilities. We first sort using the values of a specific 1D variable a. Then, thanks to unpooled_means and
# unpooled_hdi both having the County dimension, we can pass a 1D DataArray to sort the second dataset too.

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
xticks = np.arange(0, 86, 6)
fontdict = {"horizontalalignment": "right", "fontsize": 10}
for ax, level in zip(axes, ["Basement", "Floor"]):
    unpooled_means_iter = unpooled_means.sel(Level=level).sortby("a")
    unpooled_hdi_iter = unpooled_hdi.sel(Level=level).sortby(unpooled_means_iter.a)
    unpooled_means_iter.plot.scatter(x="County", y="a", ax=ax, alpha=0.8)
    ax.vlines(
        np.arange(counties),
        unpooled_hdi_iter.a.sel(hdi="lower"),
        unpooled_hdi_iter.a.sel(hdi="higher"),
        color="orange",
        alpha=0.6,
    )
    ax.set(title=f"{level.title()} estimates", ylabel="Radon estimate", ylim=(-2, 4.5))
    ax.set_xticks(xticks)
    ax.set_xticklabels(unpooled_means_iter.County.values[xticks], fontdict=fontdict)
    ax.tick_params(rotation=30)

fig.tight_layout();
plt.show()

# There seems to be more dispersion in radon levels for floor measurements than for basement ones.
# Moreover, as we saw in the forest plot, floor estimates are globally more uncertain, especially
# in some counties. We speculated that this is due to smaller sample sizes in the data, but let’s verify it!

n_floor_meas = srrs_mn.groupby("county").sum().floor
uncertainty = unpooled_hdi.a.sel(hdi="higher", Level="Floor") - unpooled_hdi.a.sel(
    hdi="lower", Level="Floor"
)

plt.plot(n_floor_meas, uncertainty, "o", alpha=0.4)
plt.xlabel("Nbr floor measurements in county")
plt.ylabel("Estimates' uncertainty");
plt.show()

# This makes sense: it’s very hard to estimate floor radon levels in counties where there are no floor measurements,
# and the model is telling us that by being very uncertain in its estimates for those counties.

# The advantage of using partial pooling (hierarchical models) becomes evident when plotting model predictions
# in cases with small sample sizes. The pooled estimates represent maximum overfitting, where no difference is
# contemplated between counties, while the unpooled estimates represent maximum underfitting, with very extreme
# cases showing higher radon levels on floor rather than on basement, for counties with few measurements.

# Let's implement a hierarchical model. A partial pooling model represents a compromise between the pooled and unpooled extremes,
# approximately a weighted average (based on sample size) of the unpooled county estimates and the pooled estimates.
# Estimates for counties with smaller sample sizes will shrink towards the state-wide average.
# Estimates for counties with larger sample sizes will be closer to the unpooled county estimates and
# will influence the the state-wide average.

with pm.Model(coords=coords) as partial_pooling:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")

    # Expected value per county:
    mu = a_county[county_idx]
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_radon, dims="obs_id")

pm.model_to_graphviz(partial_pooling)

with partial_pooling:
    partial_pooling_idata = pm.sample(tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED)

# Let's run an unpooled model without floor predictor for comparison

with pm.Model(coords=coords) as unpooled_bis:
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    a_county = pm.Normal("a_county", 0.0, sigma=10.0, dims="County")

    theta = a_county[county_idx]
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    unpooled_idata_bis = pm.sample(tune=2000, return_inferencedata=True, random_seed=RANDOM_SEED)

# Let’s compare both models’ estimates for all 85 counties. We’ll plot the estimates against each county’s sample size,
# to see more clearly what hierarchical models bring

N_county = srrs_mn.groupby("county")["idnum"].count().values

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, idata, level in zip(
    axes,
    (unpooled_idata_bis, partial_pooling_idata),
    ("no pooling", "partial pooling"),
):

    # add variable with x values to xarray dataset
    idata.posterior = idata.posterior.assign_coords({"N_county": ("County", N_county)})
    # plot means
    idata.posterior.mean(dim=("chain", "draw")).plot.scatter(
        x="N_county", y="a_county", ax=ax, alpha=0.9
    )
    ax.hlines(
        partial_pooling_idata.posterior.a.mean(),
        0.9,
        max(N_county) + 1,
        alpha=0.4,
        ls="--",
        label="Est. population mean",
    )

    # plot hdi
    hdi = az.hdi(idata).a_county
    ax.vlines(N_county, hdi.sel(hdi="lower"), hdi.sel(hdi="higher"), color="orange", alpha=0.5)

    ax.set(
        title=f"{level.title()} Estimates",
        xlabel="Nbr obs in county (log scale)",
        xscale="log",
        ylabel="Log radon",
    )
    ax.legend(fontsize=10)
fig.tight_layout();
plt.show()

# We can see that the model is skeptical of extreme deviations from the population mean in counties where data is sparse.
# Let's now include the floor predictor variable. We'll use the indicator variable approach, so that we can
# see how a model with slope works. The following model has a varying intercept for each county and a fixed slope
# for the floor variable.

with pm.Model(coords=coords) as varying_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=10.0)

    # Expected value per county:
    mu = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", mu, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(varying_intercept)

with varying_intercept:
    varying_intercept_idata = pm.sample(
        tune=2000, init="adapt_diag", random_seed=RANDOM_SEED, return_inferencedata=True
    )

az.plot_forest(
    varying_intercept_idata, var_names=["a", "a_county"], r_hat=True, combined=True, textsize=9
);
plt.show()

az.plot_trace(varying_intercept_idata, var_names=["a", "sigma_a", "b", "sigma"]);
plt.show()

az.summary(varying_intercept_idata, var_names=["a", "sigma_a", "b", "sigma"], round_to=2)

# , the estimate for the floor coefficient is reliably negative and centered around -0.66.
# This can be interpreted as houses without basements having about half the radon levels
# of those with basements, after accounting for county. This is only the relative effect
# of floor on radon levels: conditional on being in a given county, radon is expected to
# be half lower in houses without basements than in houses with. To see how much difference a
# basement makes on the absolute level of radon, we’d have to push the parameters through the
# model. Let's do it.

xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
post = varying_intercept_idata.posterior  # alias for readability
theta = (
    (post.a_county + post.b * xvals).mean(dim=("chain", "draw")).to_dataset(name="Mean log radon")
)

_, ax = plt.subplots()
theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)  # scatter
ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
# add lines too
ax.set_title("MEAN LOG RADON BY COUNTY");
plt.show()

# The graph shows, for each county, the expected log radon level and the average effect of having no basement
# Two notes: 1) this graph doesn’t show the uncertainty for each county. 2) his graph doesn’t show the uncertainty
# for each county.


# it is easy to show that the partial pooling model provides more objectively reasonable estimates than either the
# pooled or unpooled models, at least for counties with small sample sizes:
SAMPLE_COUNTIES = (
    "LAC QUI PARLE",
    "AITKIN",
    "KOOCHICHING",
    "DOUGLAS",
    "CLAY",
    "STEARNS",
    "RAMSEY",
    "ST LOUIS",
)

unpooled_idata.observed_data = unpooled_idata.observed_data.assign_coords(
    {
        "County": ("obs_id", mn_counties[unpooled_idata.constant_data.county_idx]),
        "Level": (
            "obs_id",
            np.array(["Basement", "Floor"])[unpooled_idata.constant_data.floor_idx],
        ),
    }
)


fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
for ax, c in zip(axes.ravel(), SAMPLE_COUNTIES):
    sample_county_mask = unpooled_idata.observed_data.County.isin([c])

    # plot obs:
    unpooled_idata.observed_data.where(sample_county_mask, drop=True).sortby("Level").plot.scatter(
        x="Level", y="y", ax=ax, alpha=0.4
    )

    # plot models:
    ax.plot([0, 1], unpooled_means.a.sel(County=c), "k:", alpha=0.5, label="No pooling")
    ax.plot([0, 1], pooled_means.a, "r--", label="Complete pooling")

    ax.plot([0, 1], theta["Mean log radon"].sel(County=c), "b", label="Partial pooling")

    ax.set_title(c)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)

axes[0, 0].set_ylabel("Log radon level")
axes[1, 0].set_ylabel("Log radon level")
axes[0, 0].legend(fontsize=8, frameon=True), axes[1, 0].legend(fontsize=8, frameon=True)
fig.tight_layout();
plt.show()

# we clearly see the notion that partial-pooling is a compromise between no pooling and complete pooling,
# as its mean estimates are usually between the other models’ estimates. And interestingly, the bigger
# (smaller) the sample size in a given county, the closer the partial-pooling estimates are to the no-pooling
# (complete-pooling) estimates.

# We see however that counties vary by more than just their baseline rates: the effect of floor seems to be
# different from one county to another. It would be great if our model could take that into account.
# To do that, we need to allow the slope to vary by county

# Let's now run the varying intercept and slope model

with pm.Model(coords=coords) as varying_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    sigma_b = pm.Exponential("sigma_b", 0.5)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Varying slopes:
    b_county = pm.Normal("b_county", mu=b, sigma=sigma_b, dims="County")

    # Expected value per county:
    theta = a_county[county_idx] + b_county[county_idx] * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

pm.model_to_graphviz(varying_intercept_slope)

# Running this model returns a lot of divergences, we want to avoid that as divergences represent
# non optimal sampling, possibly meaning erros in the estimation of the posterior. What comes to our help
# in this case is something called 'non-centered parametrization'. Non-centered parametrization allows
# to have the same interpretation for the estimated parameters, while avoiding divergencies (more
# efficient sampling). The big change is that the counties estimates are converted into z scores.

with pm.Model(coords=coords) as varying_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    sigma_b = pm.Exponential("sigma_b", 0.5)

    # Varying intercepts:
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    # Varying slopes:
    zb_county = pm.Normal("zb_county", mu=0.0, sigma=1.0, dims="County")

    # Expected value per county:
    theta = (a + za_county[county_idx] * sigma_a) + (b + zb_county[county_idx] * sigma_b) * floor
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    varying_intercept_slope_idata = pm.sample(
        2000, tune=2000, target_accept=0.99, random_seed=RANDOM_SEED, return_inferencedata=True
    )

az.plot_trace(varying_intercept_slope_idata, compact=True, chain_prop={"ls": "-"});
plt.show()

# All chains look good and we get a negative b coefficient, illustrating the mean downward effect
# of no-basement on radon levels at the state level. But notice that sigma_b often gets very near
# zero – which would indicate that counties don’t vary that much in their answer to the floor “treatment”.
# That’s probably what bugged MCMC when using the centered parametrization: these situations usually yield a
# weird geometry for the sampler, causing the divergence

# Let's finally plot the relationship between radon and floor for each county:

xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
post = varying_intercept_slope_idata.posterior  # alias for readability
avg_a_county = (post.a + post.za_county * post.sigma_a).mean(dim=("chain", "draw"))
avg_b_county = (post.b + post.zb_county * post.sigma_b).mean(dim=("chain", "draw"))
theta = (avg_a_county + avg_b_county * xvals).to_dataset(name="Mean log radon")

_, ax = plt.subplots()
theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)  # scatter
ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
# add lines too
ax.set_title("MEAN LOG RADON BY COUNTY");
plt.show()

#  we can see that now both the intercept and the slope vary by county. The next step is to
# analyse the covariance between slope and intercept: when baseline radon is low in a given county,
# maybe that means the difference between floor and basement measurements will decrease – because there
# isn’t that much radon anyway. Each county’s parameters come from a common distribution with mean a for
# intercepts and b for slopes, and slopes and intercepts co-vary according to the covariance matrix

coords["param"] = ["a", "b"]
coords["param_bis"] = ["a", "b"]
with pm.Model(coords=coords) as covariation_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")

    # prior stddev in intercepts & slopes (variation across counties):
    sd_dist = pm.Exponential.dist(0.5)

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)

    # prior for average intercept:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    # prior for average slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    # population of varying effects:
    ab_county = pm.MvNormal("ab_county", mu=tt.stack([a, b]), chol=chol, dims=("County", "param"))

    # Expected value per county:
    theta = ab_county[county_idx, 0] + ab_county[county_idx, 1] * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

# Let's write the non-centered version of this model, that will help us with the sampling

with pm.Model(coords=coords) as covariation_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")

    # prior stddev in intercepts & slopes (variation across counties):
    sd_dist = pm.Exponential.dist(0.5)

    # get back standard deviations and rho:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)

    # prior for average intercept:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    # prior for average slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    # population of varying effects:
    z = pm.Normal("z", 0.0, 1.0, dims=("param", "County"))
    ab_county = pm.Deterministic("ab_county", tt.dot(chol, z).T, dims=("County", "param"))

    # Expected value per county:
    theta = a + ab_county[county_idx, 0] + (b + ab_county[county_idx, 1]) * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    covariation_intercept_slope_idata = pm.sample(
        2000,
        tune=2000,
        target_accept=0.99,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
        idata_kwargs={"dims": {"chol_stds": ["param"], "chol_corr": ["param", "param_bis"]}},
    )


# the correlation between slopes and intercepts seems to be negative. But the uncertainty
# is wide on Rho so it’s possible the correlation goes the other way around or is simply close to zero.
# let’s do a forest plot and compare the estimates with the model that doesn’t include the covariation
# between slopes and intercepts:

az.plot_forest(
    [varying_intercept_slope_idata, covariation_intercept_slope_idata],
    model_names=["No covariation", "With covariation"],
    var_names=["a", "b", "sigma_a", "sigma_b", "chol_stds", "chol_corr"],
    combined=True,
    figsize=(8, 6),
);
plt.show()
# let’s visually compare estimates of both models at the county level:

# posterior means of covariation model:
a_county_cov = (
    covariation_intercept_slope_idata.posterior["a"]
    + covariation_intercept_slope_idata.posterior["ab_county"].sel(param="a")
).mean(dim=("chain", "draw"))
b_county_cov = (
    covariation_intercept_slope_idata.posterior["b"]
    + covariation_intercept_slope_idata.posterior["ab_county"].sel(param="b")
).mean(dim=("chain", "draw"))

# plot both and connect with lines
plt.scatter(avg_a_county, avg_b_county, label="No cov estimates", alpha=0.6)
plt.scatter(
    a_county_cov,
    b_county_cov,
    facecolors="none",
    edgecolors="k",
    lw=1,
    label="With cov estimates",
    alpha=0.8,
)
plt.plot([avg_a_county, a_county_cov], [avg_b_county, b_county_cov], "k-", alpha=0.5)
plt.xlabel("Intercept")
plt.ylabel("Slope")
plt.legend();
plt.show()

# The differences between the two models occur at extreme slope and intercept values.
# This is because the second model used the slightly negative correlation between
# intercepts and slopes to adjust their estimates: when intercepts are larger (smaller)
# than average, the model pushes down (up) the associated slopes. on average the model with
# covariation will be more accurate – because it squeezes additional information from the data,
# to shrink estimates in both dimensions.

# Group level predictors. Another advantage of using multilevel models is that they can handle predictors on
# multiple levels simultaneously. let's return to the varying intercepts model we saw before, but this time
# instead of using a simple random effects to describe the variation of the intercepts, we will use
# another regression model with a county-level covariate, in this case uranium readings (supposed to be connected
# to radon). This means that at the same time we are incorporating a house-level predictor (floor or basement) and
# a county-level predictor (uranium). n classical regression, this would result in collinearity. In a multilevel
# model, the partial pooling of the intercepts towards the expected value of the group-level linear model avoids this.
# Here is the model:

with pm.Model(coords=coords) as hierarchical_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    uranium = pm.Data("uranium", u, dims="County")

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, shape=2)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = g[0] + g[1] * uranium
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(hierarchical_intercept)

# But we're actually gonna use the non-centered form again:

with pm.Model(coords=coords) as hierarchical_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    uranium = pm.Data("uranium", u, dims="County")

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, shape=2)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = pm.Deterministic("a", g[0] + g[1] * uranium, dims="County")
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    a_county = pm.Deterministic("a_county", a + za_county * sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    hierarchical_intercept_idata = pm.sample(
        2000, tune=2000, target_accept=0.99, random_seed=RANDOM_SEED, return_inferencedata=True
    )

# Let's see the results

uranium = hierarchical_intercept_idata.constant_data.uranium
post = hierarchical_intercept_idata.posterior.assign_coords(uranium=uranium)
avg_a = post["a"].mean(dim=("chain", "draw")).sortby("uranium")
avg_a_county = post["a_county"].mean(dim=("chain", "draw"))
avg_a_county_hdi = az.hdi(post, var_names="a_county")["a_county"]

_, ax = plt.subplots()
ax.plot(avg_a.uranium, avg_a, "k--", alpha=0.6, label="Mean intercept")
az.plot_hdi(
    uranium,
    post["a"],
    fill_kwargs={"alpha": 0.1, "color": "k", "label": "Mean intercept HPD"},
    ax=ax,
)
ax.scatter(uranium, avg_a_county, alpha=0.8, label="Mean county-intercept")
ax.vlines(
    uranium,
    avg_a_county_hdi.sel(hdi="lower"),
    avg_a_county_hdi.sel(hdi="higher"),
    alpha=0.5,
    color="orange",
)
plt.xlabel("County-level uranium")
plt.ylabel("Intercept estimate")
plt.legend(fontsize=9);
plt.show()

# The graph above shows the average relationship and its uncertainty: the baseline radon level in an
# average county as a function of uranium, as well as the 94% HPD of this radon level (grey line and envelope).
# If we compare the county-intercepts for this model with those of the partial-pooling model without a county-level
# covariate:

az.plot_forest(
    [varying_intercept_idata, hierarchical_intercept_idata],
    model_names=["W/t. county pred.", "With county pred."],
    var_names=["a_county"],
    combined=True,
    figsize=(6, 40),
    textsize=9,
);
plt.show()

# We see that the compatibility intervals are narrower for the model including the county-level covariate.
# With this model we were able to squeeze even more information out of the data.
