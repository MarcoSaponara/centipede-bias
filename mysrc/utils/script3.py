import numpy as np
import matplotlib.pyplot as plt
import argparse

from itertools import product
from pickle import dump
from tqdm import tqdm

from egttools.analytical import PairwiseComparison
from egttools.games import Matrix2PlayerGameHolder
from egttools.utils import calculate_stationary_distribution

from centipedeGame import CentipedeGame
from kStrategy import KStrategy
from kernel import brain_unbiased, brain_prosocial, brain_proself

from scipy.spatial.distance import jensenshannon


# Define the parser
parser = argparse.ArgumentParser(description='Parameters of script3')

parser.add_argument('--popsize', action="store", default=100)
parser.add_argument('--nbsteps', action="store", default=6)
parser.add_argument('--gamma', action="store", default=0.)
parser.add_argument('--nbpoints', action="store", default=11)
parser.add_argument('--expref', action="store", default=None)

args = parser.parse_args()

Z=int(args.popsize)
nb_steps = int(args.nbsteps)
ref = str(args.expref)
assert ref in ['KT2012', 'MP1992']

gamma = float(args.gamma)
nb_points = int(args.nbpoints)

def get_payoffs(nb_steps : int,
                payoff_step0_pl1 : float = 0.4, 
                payoff_step0_pl2 : float = 0.1,
                multiplicative_factor : float = 2.,
               ):
    payoffs_pl1 = np.tile([payoff_step0_pl1, payoff_step0_pl2], nb_steps//2) * multiplicative_factor**np.arange(nb_steps, dtype=float)
    payoffs_pl2 = np.tile([payoff_step0_pl2, payoff_step0_pl1], nb_steps//2) * multiplicative_factor**np.arange(nb_steps, dtype=float)

    # add payoffs in final node
    payoffs_pl1 = np.append(payoffs_pl1, payoff_step0_pl1 * multiplicative_factor ** nb_steps)
    payoffs_pl2 = np.append(payoffs_pl2, payoff_step0_pl2 * multiplicative_factor ** nb_steps)

    return payoffs_pl1, payoffs_pl2


if __name__ == '__main__':
    
    if ref=='KT2012':
        experimental_reference = np.array([.023, .023, .023, .068, .568, .227, .068], dtype = float) # T. Kawagoe, H. Takizawa, 2012
    elif ref=='MP1992':
        experimental_reference = np.array([0., .103, .103, .310, .345, .103, .034], dtype = float) # McKelvey, Palfrey 1992 (only first session)
        
 
    beta_values = np.logspace(-3., 0., nb_points)
    eps_values = np.linspace(0., .31, nb_points)
 
    payoffs_pl1, payoffs_pl2 = get_payoffs(nb_steps)
    
    model_predictions = np.zeros((nb_points, nb_points, nb_steps+1), dtype=float)
    
    nb_k_levels = nb_steps
    k_levels = np.arange(1, nb_k_levels)
    
    start = np.zeros(nb_steps+2, dtype = float)
    # start = [0, ..., 1, 0, ..., 1, 0]
    start[-2] = 1.
    start[nb_steps//2] = 1.
    start_arrays = np.array([
                            start,
                            ], dtype = float)

    start_k_pairs = list(product(start_arrays, k_levels))
    
    
    for i, eps in enumerate(eps_values):
        strategies = [start]
        for brain in [brain_unbiased, brain_proself, brain_prosocial]:
            matrix = brain(nb_steps, gamma, eps)
            strategy = KStrategy(matrix)

            for pair in start_k_pairs:
                start, k = pair
                strategies.append(strategy.calculate_mixed_strategy(k = k, start = start))

        strategyNE = np.zeros(nb_steps+2, dtype = float)
        strategyNE[0]=1.
        strategyNE[nb_steps//2 + 1]=1.
        strategies.append(strategyNE)

        strategies = np.array(strategies)
        nb_strategies = len(strategies)

        cg = CentipedeGame(payoffs_pl1, payoffs_pl2, strategies)
        game = Matrix2PlayerGameHolder(nb_strategies, cg.payoffs())
        evolver = PairwiseComparison(population_size=Z, game=game)

        for j, beta in enumerate(beta_values):
            transition_matrix, fixation_probabilities = evolver.calculate_transition_and_fixation_matrix_sml(beta)
            stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())
            model_predictions[i,j,:] = np.dot(stationary_distribution, np.dot(stationary_distribution, cg.get_take_distributions()))
    
    js = jensenshannon(model_predictions, experimental_reference, axis = -1)
    
    #print(np.min(js))
    
    ind_js = np.unravel_index(np.argmin(js, axis=None), js.shape)
    best_fit = model_predictions[ind_js[0],ind_js[1],:]
    beta_best_fit = beta_values[ind_js[1]]
    eps_best_fit = eps_values[ind_js[0]]
    
    dict_result = {'beta_values': beta_values,
                    'eps_values': eps_values,
                    'js': js,
                    'best_fit': best_fit,
                    'beta_best_fit': beta_best_fit,
                    'eps_best_fit': eps_best_fit,
                    'experimental_reference': experimental_reference,
    }
    
    # save dictionary to .pkl file
    file_name = f'./results/fig3-Z={Z}-nbstep={nb_steps}-ref{ref}.pkl'
    with open(file_name, 'wb') as fp:
        dump(dict_result, fp)
        print(f'dictionary saved successfully to file {file_name}')


