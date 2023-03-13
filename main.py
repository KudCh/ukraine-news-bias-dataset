import os, json
import pandas as pd
import matplotlib.pyplot as plt

path_to_json_files = 'all-data-as-json/'

#get all JSON file names as a list
json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

dfs_subjectivity = []
dfs_hidden_assumptions = []

for filename in json_file_names:
    with open(path_to_json_files+filename) as f:
        dictionary_json = json.load(f)
        bias = []

        # we explore bias for each sentence in the dataset
        for s in dictionary_json['sentences']:
            subjectivity = s['subjectivity']['score']
            hidden_assumptions = s['hidden_assumptions']['score']

            df_subjectivity = pd.json_normalize(subjectivity)
            df_hidden_assumptions = pd.json_normalize(hidden_assumptions)

            dfs_subjectivity.append(df_subjectivity)
            dfs_hidden_assumptions.append(df_hidden_assumptions)

subjectivity = pd.concat(dfs_subjectivity) #every index is 0, which should not be the case...
hidden_assumptions = pd.concat(dfs_hidden_assumptions)
print("done")

for tag, bias in {"Subjectivity":subjectivity, "Hidden assumptions":hidden_assumptions}.items():
    average = bias['avg']
    majority = bias['maj']
    intensified = bias['intensified']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19.2, 4.8))
    fig.suptitle('{} scores'.format(tag), fontsize=18)

    ax[0].hist(average, bins=15)
    ax[1].hist(majority, bins=15)
    ax[2].hist(intensified, bins=15)

    ax[0].set_xlabel('average')
    ax[1].set_xlabel('majority')
    ax[2].set_xlabel('intensified')

    plt.setp(ax, xlim=[0.0, 3.0])
    plt.savefig('distributions/{}.png'.format(tag))

print('done')