import numpy as np
import random
from typing import List, Dict, Any, Tuple

from problem import ContentSegmentationProblem

def evaluate_config(
    alg_class,
    param_dict: Dict[str, Any],
    examples: List[Tuple[List[str], List[int]]],
    num_reps: int = 1
) -> Dict:
    """
    Runs the algorithm with the REAL (cached) evaluator multiple times.
    Evaluates across multiple examples and averages the F1 score.

    Returns:
        - f1_scores: list of F1s obtained (averaged across examples for each rep)
        - final_fitness: list of actual fitnesses of the best solution (averaged across examples for each rep)
        - evaluations: list of total evaluations performed (averaged across examples for each rep)
        - avg_history: average best fitness over time (averaged across all reps and examples)
        - params: the parameter dictionary used
    """
    all_f1_scores_reps = []
    all_final_fitness_reps = []
    all_evaluations_reps = []
    all_histories_reps = []  # This will store histories for each (rep, example) combination initially

    for rep in range(num_reps):

        f1_scores_this_rep = []
        final_fitness_this_rep = []
        evaluations_this_rep = []
        histories_this_rep = []

        for contents, ground_truth in examples:
            problem = ContentSegmentationProblem(contents)

            algo = alg_class(**param_dict)
            best_sol, best_fit, state = algo.solve(problem, verbose=False)

            f1 = f1_score(best_sol, ground_truth)  # defined below
            f1_scores_this_rep.append(f1)
            final_fitness_this_rep.append(best_fit)
            evaluations_this_rep.append(state['evaluations'])
            histories_this_rep.append(state['history_best_fitness'])

        # Average across examples for this repetition
        all_f1_scores_reps.append(np.mean(f1_scores_this_rep))
        all_final_fitness_reps.append(np.mean(final_fitness_this_rep))
        all_evaluations_reps.append(np.mean(evaluations_this_rep))

        # Pad histories for this rep (across examples) before averaging
        max_len_rep = max(len(h) for h in histories_this_rep)
        padded_rep = [h + [h[-1]] * (max_len_rep - len(h)) for h in histories_this_rep]
        all_histories_reps.append(np.mean(padded_rep, axis=0).tolist())

    # Average histories across all repetitions to get a single avg_history curve
    max_len_overall = max(len(h) for h in all_histories_reps)
    padded_overall = [h + [h[-1]] * (max_len_overall - len(h)) for h in all_histories_reps]
    avg_history_overall = np.mean(padded_overall, axis=0).tolist()


    return {
        'f1_scores': all_f1_scores_reps,  # Now, this is a list of averaged F1s per rep
        'final_fitness': all_final_fitness_reps,
        'evaluations': all_evaluations_reps,
        'avg_history': avg_history_overall,
        'params': param_dict
    }

def f1_score(pred: List[int], true: List[int]) -> float:
    if len(pred) != len(true):
        raise ValueError("Predicted and true cuts must have the same length.")
    tp = sum(1 for p, t in zip(pred, true) if p == 1 and t == 1)
    pred_pos = sum(pred)
    true_pos = sum(true)
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / true_pos if true_pos > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)