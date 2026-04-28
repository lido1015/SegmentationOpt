from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from llm import CoherenceScore, LlmEvaluator


@dataclass
class SegmentationSolution:
    cuts: List[int]
    score: float


def binary_to_segments(sequence: Sequence[str], cuts: Sequence[int]) -> List[Sequence[str]]:
    segments: List[Sequence[str]] = []
    start = 0
    for cut in sorted(cuts):
        segments.append(sequence[start:cut])
        start = cut
    segments.append(sequence[start:])
    return segments


def default_initial_solution(sequence: Sequence[str]) -> List[int]:
    return []


def random_solution(sequence: Sequence[str], max_cuts: Optional[int] = None) -> List[int]:
    n = len(sequence)
    if n < 2:
        return []
    if max_cuts is None:
        max_cuts = max(1, n // 4)
    candidate = random.sample(range(1, n), k=min(max_cuts, n - 1))
    return sorted(candidate)


class BaseSegmentationSearch:
    def __init__(self, evaluator: LlmEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, sequence: Sequence[str], cuts: Sequence[int]) -> float:
        segments = binary_to_segments(sequence, cuts)
        return sum(self.evaluator.score_segment(segment).score for segment in segments)


class HillClimbing(BaseSegmentationSearch):
    def __init__(self, evaluator: LlmEvaluator, restarts: int = 3) -> None:
        super().__init__(evaluator)
        self.restarts = restarts

    def optimize(self, sequence: Sequence[str]) -> SegmentationSolution:
        best_solution = SegmentationSolution(cuts=[], score=float("-inf"))
        for _ in range(self.restarts):
            current_cuts = random_solution(sequence)
            current_score = self.evaluate(sequence, current_cuts)
            improved = True
            while improved:
                improved = False
                candidate = self._find_best_neighbor(sequence, current_cuts)
                if candidate is not None and candidate.score > current_score:
                    current_cuts, current_score = candidate.cuts, candidate.score
                    improved = True
            if current_score > best_solution.score:
                best_solution = SegmentationSolution(cuts=current_cuts, score=current_score)
        return best_solution

    def _find_best_neighbor(self, sequence: Sequence[str], cuts: List[int]) -> Optional[SegmentationSolution]:
        n = len(sequence)
        best_neighbor: Optional[SegmentationSolution] = None

        def consider(cuts_candidate: List[int]) -> None:
            nonlocal best_neighbor
            score = self.evaluate(sequence, cuts_candidate)
            if best_neighbor is None or score > best_neighbor.score:
                best_neighbor = SegmentationSolution(cuts=list(cuts_candidate), score=score)

        active = set(cuts)
        for cut in range(1, n):
            if cut in active:
                candidate = [c for c in cuts if c != cut]
                consider(candidate)
            else:
                candidate = sorted(cuts + [cut])
                consider(candidate)

        for cut in list(cuts):
            for direction in (-1, 1):
                new_cut = cut + direction
                if 1 <= new_cut < n and new_cut not in active:
                    candidate = [c for c in cuts if c != cut] + [new_cut]
                    consider(sorted(candidate))

        return best_neighbor


class SimulatedAnnealing(BaseSegmentationSearch):
    def __init__(
        self,
        evaluator: LlmEvaluator,
        temperature: float = 5.0,
        cooling_rate: float = 0.95,
        iterations_per_temp: int = 10,
    ) -> None:
        super().__init__(evaluator)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp

    def optimize(self, sequence: Sequence[str]) -> SegmentationSolution:
        current_cuts = random_solution(sequence)
        current_score = self.evaluate(sequence, current_cuts)
        best_solution = SegmentationSolution(cuts=list(current_cuts), score=current_score)
        temperature = self.temperature

        while temperature > 0.1:
            for _ in range(self.iterations_per_temp):
                candidate_cuts = self._neighbor(sequence, current_cuts)
                candidate_score = self.evaluate(sequence, candidate_cuts)
                delta = candidate_score - current_score
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-8)):
                    current_cuts, current_score = candidate_cuts, candidate_score
                    if current_score > best_solution.score:
                        best_solution = SegmentationSolution(cuts=list(current_cuts), score=current_score)
            temperature *= self.cooling_rate

        return best_solution

    def _neighbor(self, sequence: Sequence[str], cuts: List[int]) -> List[int]:
        n = len(sequence)
        if n < 2:
            return []

        moves = [self._add_cut, self._remove_cut, self._move_cut]
        return random.choice(moves)(sequence, cuts)

    def _add_cut(self, sequence: Sequence[str], cuts: List[int]) -> List[int]:
        n = len(sequence)
        available = [i for i in range(1, n) if i not in cuts]
        if not available:
            return list(cuts)
        return sorted(cuts + [random.choice(available)])

    def _remove_cut(self, sequence: Sequence[str], cuts: List[int]) -> List[int]:
        if not cuts:
            return []
        candidate = list(cuts)
        candidate.remove(random.choice(candidate))
        return candidate

    def _move_cut(self, sequence: Sequence[str], cuts: List[int]) -> List[int]:
        if not cuts:
            return self._add_cut(sequence, cuts)
        candidate = list(cuts)
        cut = random.choice(candidate)
        candidate.remove(cut)
        n = len(sequence)
        available = [i for i in range(1, n) if i not in candidate]
        if not available:
            return sorted(candidate)
        candidate.append(random.choice(available))
        return sorted(candidate)


class GeneticAlgorithm(BaseSegmentationSearch):
    def __init__(
        self,
        evaluator: LlmEvaluator,
        population_size: int = 20,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
    ) -> None:
        super().__init__(evaluator)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def optimize(self, sequence: Sequence[str]) -> SegmentationSolution:
        population = [random_solution(sequence) for _ in range(self.population_size)]
        scores = [self.evaluate(sequence, individual) for individual in population]

        best_index = int(max(range(len(population)), key=lambda idx: scores[idx]))
        best_solution = SegmentationSolution(cuts=list(population[best_index]), score=scores[best_index])

        for _ in range(self.generations):
            new_population: List[List[int]] = []
            while len(new_population) < self.population_size:
                parent_a = self._select(population, scores)
                parent_b = self._select(population, scores)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(sequence, child)
                new_population.append(child)

            population = new_population
            scores = [self.evaluate(sequence, individual) for individual in population]
            current_best_index = int(max(range(len(population)), key=lambda idx: scores[idx]))
            if scores[current_best_index] > best_solution.score:
                best_solution = SegmentationSolution(
                    cuts=list(population[current_best_index]),
                    score=scores[current_best_index],
                )

        return best_solution

    def _select(self, population: List[List[int]], scores: List[float]) -> List[int]:
        tournament = random.sample(list(range(len(population))), k=min(3, len(population)))
        best = max(tournament, key=lambda idx: scores[idx])
        return list(population[best])

    def _crossover(self, parent_a: List[int], parent_b: List[int]) -> List[int]:
        if random.random() > self.crossover_rate or not parent_a or not parent_b:
            return list(parent_a if random.random() < 0.5 else parent_b)
        cut = random.choice(parent_a + parent_b)
        child = sorted(set(parent_a[: parent_a.index(cut) + 1] + parent_b[parent_b.index(cut) :] if cut in parent_a and cut in parent_b else parent_a))
        return [value for value in child if value is not None]

    def _mutate(self, sequence: Sequence[str], cuts: List[int]) -> List[int]:
        n = len(sequence)
        new_cuts = sorted(set(cuts))
        for index in range(len(new_cuts)):
            if random.random() < self.mutation_rate:
                new_cuts[index] = random.randint(1, n - 1)
        if random.random() < self.mutation_rate and n > 1:
            new_cut = random.randint(1, n - 1)
            if new_cut not in new_cuts:
                new_cuts.append(new_cut)
        new_cuts = [cut for cut in sorted(set(new_cuts)) if 1 <= cut < n]
        return new_cuts
