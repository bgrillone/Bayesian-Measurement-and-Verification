from bayes_functions_multiprocessing import *
import multiprocessing

# Import ASHRAE dataset

bdg_df = pd.read_csv("/root/benedetto/data/bdg/electricity_cleaned.csv")
bdg_weather = pd.read_csv("/root/benedetto/data/bdg/weather.csv")

# Run one building subset at the time

subset_df = bdg_df
#subset_df = bdg_df.loc[:, bdg_df.columns.str.startswith(('Shrew', 'Swan', 'Wolf', 'Eagle', 'Cockatoo', 'Mouse', 'Hog', 'timestamp'))]

buildings = []

for building in subset_df.loc[:, subset_df.columns != 'timestamp']:
    weather_df = bdg_weather.loc[bdg_weather.site_id.str.startswith(building[0:3]),:][['timestamp','airTemperature']]
    df = pd.merge(subset_df[['timestamp',building]], weather_df, on = 'timestamp')
    df = df.dropna()
    buildings.append(df)

print("Start multithreading pool")

with multiprocessing.Pool(8) as pool:
    print("Launch calculations")
    tasks = [pool.apply_async(multiprocessing_bayesian_comparison,(x,)) for x in buildings]
    [t.get() for t in tasks]
    print("End")


