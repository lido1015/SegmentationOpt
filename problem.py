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

from llm import eval


class Problem(ABC):
    """
    Abstract representation of the optimisation problem.

    All problem‑specific knowledge (solution encoding, evaluation,
    neighbourhood, genetic operators) is encapsulated here so that
    metaheuristics remain domain‑independent.
    """

    @abstractmethod
    def create_solution(self) -> Any:
        """Return a new, random or heuristic, feasible solution."""
        ...

    @abstractmethod 
    def objective_function(self, solution: Any) -> float:
            """
            Compute the quality of a solution.
            **The framework assumes maximisation.** If your problem is a
            minimisation one, either return the negative of the cost or adjust
            the acceptance rules in the concrete metaheuristic.
            """
            ...

    def get_neighbor(self, solution: Any, neighborhood_size: int) -> Any:
        """
        Generate a neighbouring solution.
        Must be implemented if the problem is to be used with
        single‑solution metaheuristics.
        """
        raise NotImplementedError(
            "get_neighbor is required for single‑solution methods"
        )

    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """
        Produce two offspring from two parents.
        Must be implemented for genetic algorithms.
        """
        raise NotImplementedError(
            "crossover is required for the genetic algorithm"
        )

    def mutate(self, solution: Any) -> None:
        """
        Apply a small random perturbation **in‑place** to the solution.
        Must be implemented for genetic algorithms.
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

    # def create_solution(self) -> List[int]:
    #     """
    #     Generate a heuristic binary string of length ``n-1``.
    #     If `randomize` is True, a completely random solution is generated.
    #     Otherwise, a heuristic solution is generated based on content length.
    #     """
    #     if self.n <= 1:
    #         return []

    #     num_cuts_vector_len = self.n - 1
    #     cuts = [0] * num_cuts_vector_len

    #     if num_cuts_vector_len == 0:
    #         return cuts

    #     # Determine 'i' such that 2^(i-1) < num_cuts_vector_len <= 2^i
    #     # This means i = floor(log2(num_cuts_vector_len)) + 1 based on example
    #     # Let's adjust based on the user's explicit example:
    #     # n=2 to 4 -> 0 cuts (i-1 = 0 -> i=1) i.e. num_cuts_vector_len 1-3
    #     # n=5 to 8 -> 1 cut (i-1 = 1 -> i=2) i.e. num_cuts_vector_len 4-7
    #     # n=9 to 16 -> 2 cuts (i-1 = 2 -> i=3) i.e. num_cuts_vector_len 8-15

    #     # This pattern matches k = max(0, floor(log2(num_cuts_vector_len)) - 1)
    #     # for num_cuts_vector_len >= 1. For num_cuts_vector_len=1, floor(log2(1)) = 0, k = -1 -> 0
    #     # For num_cuts_vector_len=2, floor(log2(2)) = 1, k = 0
    #     # For num_cuts_vector_len=3, floor(log2(3)) = 1, k = 0
    #     # For num_cuts_vector_len=4, floor(log2(4)) = 2, k = 1

    #     i_val = math.floor(math.log2(num_cuts_vector_len))
    #     k_cuts = max(0, i_val - 1)

    #     if k_cuts > 0:
    #         for j in range(1, k_cuts + 1):
    #             # Distribute cuts equitably (j / (k_cuts + 1) fractions)
    #             cut_index = round(num_cuts_vector_len * (j / (k_cuts + 1))) -1 # Adjust for 0-based indexing
    #             if 0 <= cut_index < num_cuts_vector_len:
    #                 cuts[cut_index] = 1

    #     return cuts

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
        return eval(self.contents, solution)
    

    # def get_neighbor(self, solution: List[int]) -> List[int]:
    #     """
    #     Create a neighbour by flipping exactly one random cut.
    #     Returns a new list; the original is not modified.
    #     """
    #     neighbor = solution.copy()
    #     idx = random.randrange(self.n - 1)
    #     neighbor[idx] = 1 - neighbor[idx]
    #     return neighbor

    def get_neighbor(self, solution: List[int], neighborhood_size: int = 1) -> Tuple[List[int], float]:
        """
        Generate a neighboring solution by exploring multiple candidates.
        It returns the best neighbor found and its fitness.
        """
        best_neighbor = solution.copy()
        best_neighbor_fitness = -float('inf') # Initialize to negative infinity to ensure any improvement is captured

        if self.n <= 1:
            return solution.copy(), self.objective_function(solution.copy())

        num_cuts_vector_len = self.n - 1

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