import json
import numpy as np
machines = {}

machines['instance_size'] = 2
machines['machines'] = {"machine1": {"name": 'mill','flow': [0,1]},
                        "machine2": {"name": 'lathe','flow': [1,0], 'can_be_dirty': 'Yes'}}
machines['flows'] = np.array([[0,1,2,3],
                             [1,0,1,1]]).tolist()

json_dump = json.dumps(machines)

print(machines)