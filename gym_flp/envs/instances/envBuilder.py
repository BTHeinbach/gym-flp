import json

import tkinter as tk
import numpy as np
import pandas as pd

from tkinter import filedialog
from itertools import compress


def load_from_file(target_env: str, target_size: int):
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()

    data = _open(file_path)

    env_info = _dismantle_data(data, target_env)

    _sanity_check(env_info, target_size, target_env)

    return env_info


def _open(path: str) -> pd.DataFrame:
    f = open(path, 'r')

    if not f.name.lower().endswith(('.txt', '.prn', '.json')):
        raise Exception("Invalid file type")

    elif f.name.lower().endswith('.json'):
        file = json.load(f)
        data = pd.DataFrame.from_dict(file)
        # data = data.values.tolist()

    elif f.name.lower().endswith(('.txt', '.prn')):
        file = f.read().splitlines()
        data = pd.DataFrame([x.strip().split() for x in file if x])

    return data


def _dismantle_data(data: pd.DataFrame, target_env: str):
    if target_env.lower() == 'qap':

        flows = []
        distances = []

        required = ['f', 'd']

        indices = [data.columns.get_loc(c) for c in list(compress(required, data.columns.isin(required)))]

        x = indices[0]
        y = indices[1]

        flows = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, x]]
        distances = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, y]]

        env_info = {'flows': flows, 'distances': distances}
    else:

        flows = []
        areas = []
        widths = []
        lengths = []

        required = ['f', 'a', 'w', 'l', 'X', 'Y']
        indices = [data.columns.get_loc(c) for c in list(compress(required, [r in data.columns for r in required]))]

        if 'f' in data.columns:
            x = data.columns.get_loc('f')
            flows = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, x]]
        else:
            raise Exception("No flow data found in input data. "
                            "Please make sure flow information with key 'f' is present")

        if 'a' in data.columns:
            y = data.columns.get_loc('a')
            areas = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, y]]

        if 'w' in data.columns:
            y = data.columns.get_loc('w')
            widths = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, y]]

        if 'l' in data.columns:
            y = data.columns.get_loc('l')
            lengths = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, y]]

        if len(areas) == 0:
            if len(widths) > 0:
                if len(lengths) > 0:
                    areas = widths*lengths

            elif len(lengths) > 0:
                areas = lengths*lengths


        if 'Y' in data.columns:
            y = data.columns.get_loc('Y')
            plantY = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, y]]

        if 'X' in data.columns:
            y = data.columns.get_loc('X')
            plantX = [list(i.values()) if isinstance(i, dict) else i for i in data.iloc[:, y]]

        env_info = {'flows': flows, 'areas': areas, 'widths': widths, 'lengths': lengths, 'plantX': np.mean(plantX), 'plantY': np.mean(plantY)}
    return env_info


def _sanity_check(env_info: dict, target_size: int, target_env: str):
    assert 'flows' in env_info.keys()
    assert np.array(env_info['flows']).shape == (target_size, target_size)

    if target_env == 'qap':
        assert 'distances' in env_info.keys()
        assert np.array(env_info['distances']).shape == (target_size, target_size)
    else:
        assert 'areas' in env_info.keys()
        assert 'plantX' in env_info.keys()
        assert 'plantY' in env_info.keys()
        assert len(env_info['areas']) == target_size

    return True