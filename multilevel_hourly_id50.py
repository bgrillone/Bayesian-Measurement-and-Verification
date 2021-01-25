import arviz  as az
import matplotlib.pyplot as plt
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr
import warnings

# Data import
df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/Id50_hourly.csv")

#Cleaning: there's 54 NAs
df.isna().sum()

# Preprocessing
df["log_v"] = log_electricity = np.log(df["Value"]).values

# Plotting data
measured = df[np.isfinite(df["Value"])].Value
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

measured_log = df[np.isfinite(df["Value"])].log_v
hist_l, edges_l = np.histogram(measured_log, density = True, bins = 50)
x_l = np.linspace (0, 12, num=20)
p2 = make_plot("Log Electricity Hist", hist_l, edges_l, x_l)
show(gridplot([p1,p2], ncols = 2))

# They're not normal, even after logging: what do we do? GLM?

# Clustering

# Temperature dependence detection

# Fourier Series on the hour of the day 

# Multilevel model
