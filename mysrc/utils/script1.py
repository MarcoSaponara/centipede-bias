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


# Define the parser
parser = argparse.ArgumentParser(description='Parameters of script1')

parser.add_argument('--popsize', action="store", default=100)
parser.add_argument('--eps', action="store", default=0.)
parser.add_argument('--nbsteps', action="store", default=6)
parser.add_argument('--gamma', action="store", default=0.)
parser.add_argument('--nbpoints', action="store", default=51)

args = parser.parse_args()

Z=int(args.popsize)
eps = float(args.eps)
nb_steps = int(args.nbsteps)

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
    payoffs_pl1, payoffs_pl2 = get_payoffs(nb_steps)

    nb_k_levels = nb_steps
    k_levels = np.arange(1, nb_k_levels)
    
    start = np.zeros(nb_steps+2, dtype = float)
    # start = [0, ..., 1, 0, ..., 1, 0]
    start[nb_steps//2] = 1.
    start[-2] = 1.
    start_arrays = np.array([
                            start,
                            ], dtype = float)

    start_k_pairs = list(product(start_arrays, k_levels))

    
    beta_values = np.logspace(-4., 1., nb_points)
    
    strategy_labels = ["0"]
    for brain in ["neu", "pes", "opt"]:
        for i in range(1, nb_k_levels):
            strategy_labels.append(str(i)+brain)
    strategy_labels.append("NE")
    nb_strategies = len(strategy_labels)
    
    results_fig1 = np.zeros((nb_points, nb_strategies), dtype=float)

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

    cg = CentipedeGame(payoffs_pl1, payoffs_pl2, strategies)
    game = Matrix2PlayerGameHolder(nb_strategies, cg.payoffs())
    evolver = PairwiseComparison(population_size=Z, game=game)
    
    for i, beta in enumerate(beta_values):
        transition_matrix, fixation_probabilities = evolver.calculate_transition_and_fixation_matrix_sml(beta)
        stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())

        results_fig1[i,:] = stationary_distribution
        
    assert np.allclose(results_fig1.sum(axis=1), 1.) 
    
    dict_result = {'strategy_labels' : strategy_labels,
                   'beta_values' : beta_values,
                   'results_fig1': results_fig1,
                  }
    
    # save dictionary to .pkl file
    file_name = f'./results/fig1-Z={Z}-nbstep={nb_steps}-eps={eps}.pkl'
    with open(file_name, 'wb') as fp:
        dump(dict_result, fp)
        print(f'dictionary saved successfully to file {file_name}')
    
    
    
##



