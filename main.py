import os, json
import pandas as pd
import matplotlib.pyplot as plt

path_to_json_files = 'all-data-as-json/'

#get all JSON file names as a list
json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

dataframes = []

for filename in json_file_names:
    with open(path_to_json_files+filename) as f:
        dictionary_json = json.load(f)
        bias = []
        for s in dictionary_json['sentences']:
            bias_data = s['bias']['score']['pro-russia']

            df = pd.json_normalize(bias_data)
            dataframes.append(df)

data = pd.concat(dataframes) #every index is 0, which should not be the case...

print(data.head())
s = pd.Series(range(len(data)))
data = data.set_index(s)



data_avg = data[['pro-russia.avg', 'pro-west.avg']]
data_maj = data[['pro-russia.maj', 'pro-west.maj']]
data_intensified = data[['pro-russia.intensified', 'pro-west.intensified']]

plt.plot(data['pro-west.avg'], label = 'pro-west.avg')
plt.plot(data['pro-russia.avg'], label = 'pro-russia.avg')
plt.show()