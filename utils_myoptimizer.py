from abc import ABC, abstractmethod
from pymoo.core.problem import Problem
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
import pandas as pd
import os

def optimization(problem, algorithm, xtol=1e-8, cvtol=1e-8, ftol=25e-4, period=30,
                 n_gen=100, n_max_evals=10000):

    termination = DefaultMultiObjectiveTermination(
        xtol=xtol, cvtol=cvtol, ftol=ftol, period=period, n_max_gen=n_gen, n_max_evals=n_max_evals
    )

    results = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=True)  # STANDARD FOR NSGA
    return results


def load_snn_ground_truth(folder_path):
    data = {}
    for cell in ['granule_cell', 'golgi_cell', 'MLI_cell', 'purkinje_cell']:
        df = pd.read_csv(os.path.join(folder_path, f'{cell}_fr_for_TF_3.csv'))
        data[cell] = {
            'freqs': df.iloc[:, 0].values,
            'mean': df.iloc[:, 1].values,
            'std': df.iloc[:, 2].values,
        }
    return data