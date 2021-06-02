import arviz  as az
from bokeh.plotting import figure, output_file, show, save
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
from bayes_functions import bayesian_model_comparison
import subprocess
import openpyxl

# Import ASHRAE dataset

bdg_df = pd.read_csv("/root/benedetto/data/bdg/electricity_cleaned.csv")
bdg_weather = pd.read_csv("/root/benedetto/data/bdg/weather.csv")

# First let's try only with a small subset (Crow = 5 buildings)

crow_df = bdg_df.loc[:, bdg_df.columns.str.startswith(('Crow', 'timestamp'))]

# Create for loop to create a dataset from one column at a time

for building in crow_df.loc[:, crow_df.columns != 'timestamp']:
    weather_df = bdg_weather.loc[bdg_weather.site_id.str.startswith(building[0:3]),:][['timestamp','airTemperature']]
    df = pd.merge(crow_df[['timestamp',building]], weather_df, on = 'timestamp')
    df = df.dropna()
    df.columns = ['t', 'total_electricity', 'outdoor_temp']

    edif = "current_df"
    df.to_csv(edif + ".csv", index = False)
    subprocess.run(["Rscript", "ashrae_preprocess_server.R", edif])

    df_preprocessed = pd.read_csv(edif + "_preprocess.csv")
    print(df_preprocessed.head())

    model_results = bayesian_model_comparison(df_preprocessed)
    model_results['id'] = building
    # read the csv with the values from previous buildings
    # append to that Excel
    try:
        dat = pd.read_csv("bayes_results.csv")
        final_export = dat.append(model_results)
    except:
        final_export = model_results

    final_export.to_csv("bayes_results.csv", index = False)
