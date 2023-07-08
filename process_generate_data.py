import snscrape.modules.twitter as sntwitter
import pandas as pd
import warnings

import random
import csv

database = pd.read_csv("database.csv")
def generate_random_data(keyword, total_data):
    data_without_nan = database.dropna()
    filtered_data = data_without_nan[data_without_nan['content'].str.contains(keyword)]
    sample_data = filtered_data.sample(total_data)
    return sample_data 
