
import pandas as pd
import numpy as np

def normalize(df):
    df = (df-df.mean())/df.std()
    return df
