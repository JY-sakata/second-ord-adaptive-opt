import os
import pandas as pd


class Data_Visualisation():
    def __init__(self, optimiser_name, file_path_list, variables):
        self.optimiser_name = optimiser_name
        self.collection = {}
        for file_path in file_path_list:
            for root, dirs, files in os.walk(file_path, topdown=False):
                for f in files:
                    if f.startswith('progress'):
                        df = pd.read_csv(os.path.join(root, f))
                        for var in variables:
                            if var in df.columns:
                                if var in self.collection:
                                    self.collection[var].append(df[var])
                                else:
                                    self.collection[var] = [df[var]]
        

    def get_series_mean(self, variable_name):
        if variable_name in self.collection:
            if len(self.collection[variable_name]) == 1:
                series_df = self.collection[variable_name][0]
       
            else:
                df = pd.concat(self.collection[variable_name], axis=1)
                series_df = df.mean(axis=1)
            return series_df
        else:
            return None

    def get_series_std(self, variable_name):

        if variable_name in self.collection:
            if len(self.collection[variable_name]) == 1:
                series_df = self.collection[variable_name][0]
       
            else:
                df = pd.concat(self.collection[variable_name], axis=1)
                series_df = df.std(axis=1)
            return series_df
        else:
            return 'No Supported Data To Plot'