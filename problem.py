"""
Problem definitions for the metaheuristic framework.
-----------------------------------------------------
This module contains the abstract ``Problem`` interface and a concrete
``ContentSegmentationProblem`` that maximises segment coherence using
an external LLM evaluator.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import random
import math

from llm import evaluate_segmentation


class Problem(ABC):
    """
    Abstract representation of the optimization problem.

    All problem-specific knowledge (solution encoding, evaluation,
    neighborhood generation, and genetic operators) is encapsulated
    here so that metaheuristics remain domain-independent.
    """

    @abstractmethod
    def create_solution(self) -> Any:
        """Generate a feasible solution for the problem."""
        ...

    @abstractmethod 
    def objective_function(self, solution: Any) -> float:
        """Compute the fitness of a candidate solution.

        The metaheuristic framework treats higher values as better.
        For minimization problems, return the negated cost if needed.
        """
        ...

    def get_neighbor(self, solution: Any, neighborhood_size: int) -> Any:
        """Return a neighboring solution for local search.

        This method is required when using single-solution metaheuristics.
        """
        raise NotImplementedError(
            "get_neighbor is required for single-solution methods"
        )

    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Create two children from two parent solutions.

        This method is used by genetic algorithms.
        """
        raise NotImplementedError(
            "crossover is required for the genetic algorithm"
        )

    def mutate(self, solution: Any) -> None:
        """Apply a small random perturbation to a solution in place.

        This method is used by genetic operators in population-based search.
        """
        raise NotImplementedError(
            "mutate is required for the genetic algorithm"
        )


class ContentSegmentationProblem(Problem):
    """
    Text segmentation problem that maximises the sum of segment
    coherence scores, evaluated by an external LLM.

    A **solution** is a binary list of length ``n-1`` where
    ``cuts[i] = 1`` indicates a segment break after the i‑th content
    element. The number of elements ``n`` is given by the length of the
    ``contents`` list.

    Parameters:
        contents:    list of text fragments (length >= 2).
        evaluator:   an object with a method
                     ``evaluate_solution(contents, cuts)`` that returns
                     a list of coherence scores for the segments defined
                     by ``cuts``.
        random_seed: optional seed for reproducible random numbers.
    """

    def __init__(self, contents: List[str]):
        self.contents = contents
        self.n = len(contents)
        if self.n < 2:
            raise ValueError(
                "At least two fragments are required to define cuts."
            )

  

    # -----------------------------------------------------------------
    # Implementation of Problem interface
    # -----------------------------------------------------------------

    def create_solution(self) -> List[int]:
        """
        Generate a random binary string of length ``n-1``.
        Each cut is independently set to 0 or 1 with equal probability.
        """
        return [random.randint(0, 1) for _ in range(self.n - 1)]


    def objective_function(self, solution: List[int]) -> float:
        """
        Evaluate a solution by summing the coherence scores of each
        segment. The actual scoring is delegated to the evaluator.

        Args:
            solution: a list of 0/1 of length ``self.n - 1``.

        Returns:
            Total fitness (higher is better).
        """
        if len(solution) != self.n - 1:
            raise ValueError(
                f"Solution length {len(solution)} does not match "
                f"expected {self.n - 1}"
            )
        return evaluate_segmentation(self.contents, solution)
    

    def get_neighbor(self, solution: List[int], neighborhood_size: int = 1) -> Tuple[List[int], float]:
        """
        Generate a neighboring solution by exploring multiple candidates.
        It returns the best neighbor found and its fitness.
        """
        best_neighbor = solution.copy()
        best_neighbor_fitness = -float('inf') # Initialize to negative infinity to ensure any improvement is captured

        if self.n <= 1:
            return solution.copy(), self.objective_function(solution.copy())

        for _ in range(neighborhood_size):
            candidate_neighbor = solution.copy()
            operation_type = random.choice(['add_cut', 'remove_cut', 'move_cut'])

            if operation_type == 'add_cut':
                zero_indices = [i for i, x in enumerate(candidate_neighbor) if x == 0]
                if zero_indices:
                    idx = random.choice(zero_indices)
                    candidate_neighbor[idx] = 1
            elif operation_type == 'remove_cut':
                one_indices = [i for i, x in enumerate(candidate_neighbor) if x == 1]
                if one_indices:
                    idx = random.choice(one_indices)
                    candidate_neighbor[idx] = 0
            elif operation_type == 'move_cut':
                one_indices = [i for i, x in enumerate(candidate_neighbor) if x == 1]
                if one_indices:
                    old_idx = random.choice(one_indices)
                    candidate_neighbor[old_idx] = 0 # Remove the old cut

                    zero_indices = [i for i, x in enumerate(candidate_neighbor) if x == 0]
                    if zero_indices:
                        new_idx = random.choice(zero_indices)
                        candidate_neighbor[new_idx] = 1
                    else:
                        # If no new position, put the cut back at old_idx
                        candidate_neighbor[old_idx] = 1

            candidate_neighbor_fitness = self.objective_function(candidate_neighbor)

            if candidate_neighbor_fitness > best_neighbor_fitness:
                best_neighbor = candidate_neighbor
                best_neighbor_fitness = candidate_neighbor_fitness

        return best_neighbor, best_neighbor_fitness

    def crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Single‑point crossover on the binary cut vectors.
        A random cut point is chosen, and the tails are swapped.
        """
        if self.n <= 2:
            # Not enough room to meaningfully cut – just return copies.
            return parent1.copy(), parent2.copy()

        point = random.randint(1, self.n - 2)  # cut between cuts
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, solution: List[int]) -> None:
        """
        Flip each bit with probability ``1/(n-1)``.
        It is guaranteed that at least one bit is flipped.
        The modification happens **in‑place**.
        """
        if self.n <= 1:
            return
        changed = False
        for i in range(len(solution)):
            if random.random() < 1.0 / (self.n - 1):
                solution[i] = 1 - solution[i]
                changed = True
        if not changed:
            idx = random.randrange(len(solution))
            solution[idx] = 1 - solution[idx]