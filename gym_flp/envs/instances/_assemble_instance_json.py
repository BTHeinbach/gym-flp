import json
import os
import pandas as pd
import numpy as np

loc1 = r'./continual'
loc2 = r'./discrete/problems'

'''
files = [f for f in os.listdir(loc2) if f.endswith('.dat')]
problems = {}
for file in files:

    file_path = os.path.join(loc2, file)
    with open(file_path, "r") as f:
        txt = f.read().splitlines()
        data = [x.strip().split() for x in txt if x]
        name = file.split('.')[0]

        n = int(data[0][0])
        problems[file.split('.')[0]] = {'d': pd.DataFrame([x for x in data[1:1+n] if x]).to_dict(),
                                        'f': pd.DataFrame([x for x in data[1+n:] if x]).to_dict()
                                        }

with open(f"discrete.json", "w") as outfile:
    json.dump(problems, outfile)

'''
files = [f for f in os.listdir(os.path.join(loc1, 'flows')) if f.endswith('.prn')]
problems = {}

for file in files:
    file_path = os.path.join(loc1, 'flows', file)
    with open(file_path, "r") as f:
        txt = f.read().splitlines()
        data = [x.strip().split() for x in txt if x]
        name = file.split('.')[0]
        n = int(data[0][0]) if len(data[0]) == 1 else len(data)
        data = data[1:] if len(data[0]) == 1 else data
        flows = pd.DataFrame([x for x in data if x]).to_dict()

        with open(os.path.join(loc1, 'areas', name+'.prn')) as g:
            txt = g.read().splitlines()
            data = [x.strip().split() for x in txt if x]
            starting_index = 2
            area_data = pd.DataFrame([x for x in data[starting_index:n+starting_index] if x], columns=data[1]).to_dict()
            plant_data = pd.DataFrame([x for x in data[n + starting_index:n + starting_index + 2] if x]).to_dict()

            flows = [list(i.values()) if isinstance(i, dict) else i for i in flows.values()]
            areas = [list(i.values()) if isinstance(i, dict) else i for i in area_data['Area'].values()] if 'Area' in area_data.keys() else None
            heights = None
            lengths = [list(i.values()) if isinstance(i, dict) else i for i in area_data['Length'].values()] if 'Length' in area_data.keys() else None
            z = set(area_data).intersection(set(['Width', 'Height']))
            if len(z)>0:
                heights = [list(i.values()) if isinstance(i, dict) else i for i in area_data[[i for i in z][0]].values()]

            problems[name] = {
                'flows': flows,
                'areas': areas,
                'widths': heights,
                'lengths': lengths,
                'plantX': float(plant_data[1][0]) if len(plant_data) > 0 else None,
                'plantY': float(plant_data[1][1]) if len(plant_data) > 0 else None,
            }
with open(f"continuous.json", "w") as outfile:
    json.dump(problems, outfile)
