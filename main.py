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
            bias_data = s['bias']['score']['pro-west']

            df = pd.json_normalize(bias_data)
            dataframes.append(df)

data = pd.concat(dataframes) #every index is 0, which should not be the case...

average = data['avg']
majority = data['maj']
intensified = data['intensified']

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19.2, 4.8))
fig.suptitle('Pro-west bias scores', fontsize=18)

ax[0].hist(average, bins=15)
ax[1].hist(majority, bins=15)
ax[2].hist(intensified, bins=15)

ax[0].set_xlabel('average')
ax[1].set_xlabel('majority')
ax[2].set_xlabel('intensified')

plt.setp(ax, xlim=[0.0, 3.0])
plt.savefig('distributions/bias_pro-west.png')

print('done')