import os, json
import pandas as pd
import matplotlib.pyplot as plt

path_to_json_files = 'all-data-as-json/'

#get all JSON file names as a list
json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

dataframes = []
dfs_russia = []
dfs_west = []
dfs_ukraine = []

for filename in json_file_names:
    with open(path_to_json_files+filename) as f:
        dictionary_json = json.load(f)
        bias = []

        # we explore bias for each sentence in the dataset
        for s in dictionary_json['sentences']:
            framing_russia = s['framing']['score']['russia']
            framing_ukraine = s['framing']['score']['ukraine']
            framing_west = s['framing']['score']['west']

            df_west = pd.json_normalize(framing_west)
            df_russia = pd.json_normalize(framing_russia)
            df_ukraine = pd.json_normalize(framing_ukraine)

            dfs_west.append(df_west)
            dfs_ukraine.append(df_ukraine)
            dfs_russia.append(df_russia)

framing_west = pd.concat(dfs_west) #every index is 0, which should not be the case...
framing_russia = pd.concat(dfs_russia)
framing_ukraine = pd.concat(dfs_ukraine)
print("done")

for tag, framing in {"West":framing_west, "Russia":framing_russia, "Ukraine":framing_ukraine}.items():
    average = framing['avg']
    majority = framing['maj']
    intensified = framing['intensified']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19.2, 4.8))
    fig.suptitle('{} framing scores'.format(tag), fontsize=18)

    ax[0].hist(average, bins=15)
    ax[1].hist(majority, bins=15)
    ax[2].hist(intensified, bins=15)

    ax[0].set_xlabel('average')
    ax[1].set_xlabel('majority')
    ax[2].set_xlabel('intensified')

    plt.setp(ax, xlim=[-2.0, 2.0])
    plt.savefig('distributions/framing_{}.png'.format(tag))

print('done')