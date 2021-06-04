import arviz  as az
from bokeh.plotting import figure, output_file, show, save
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
from bayes_functions_multiprocessing import *
import subprocess
import openpyxl
import multiprocessing
# Import ASHRAE dataset

bdg_df = pd.read_csv("/root/benedetto/data/bdg/electricity_cleaned.csv")
bdg_weather = pd.read_csv("/root/benedetto/data/bdg/weather.csv")

# Run one building subset at the time

subset_df = bdg_df.loc[:, bdg_df.columns.str.startswith(('Crow', 'timestamp'))]

buildings = []

for building in subset_df.loc[:, subset_df.columns != 'timestamp']:
    weather_df = bdg_weather.loc[bdg_weather.site_id.str.startswith(building[0:3]),:][['timestamp','airTemperature']]
    df = pd.merge(subset_df[['timestamp',building]], weather_df, on = 'timestamp')
    df = df.dropna()
    buildings.append(df)

pool = multiprocessing.Pool(8)
pool.map(multiprocessing_bayesian_comparison, buildings)





