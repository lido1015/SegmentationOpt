from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from llm import CoherenceScore, LlmEvaluator


class OptimalSegmentationDP:
    """Implementación exacta de programación dinámica para segmentación óptima."""

    def __init__(self, evaluator: LlmEvaluator) -> None:
        self.evaluator = evaluator
        self.score_cache: Dict[Tuple[int, int], float] = {}

    def _segment_score(self, sequence: Sequence[str], start: int, end: int) -> float:
        key = (start, end)
        if key in self.score_cache:
            return self.score_cache[key]

        segment = sequence[start:end]
        coherence = self.evaluator.score_segment(segment)
        self.score_cache[key] = coherence.score
        return coherence.score

    def solve(self, sequence: Sequence[str]) -> Tuple[List[int], float]:
        n = len(sequence)
        dp: List[float] = [0.0] * (n + 1)
        backpointer: List[int] = [-1] * (n + 1)

        for i in range(1, n + 1):
            best_score = float("-inf")
            best_cut = 0
            for j in range(0, i):
                candidate_score = dp[j] + self._segment_score(sequence, j, i)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_cut = j
            dp[i] = best_score
            backpointer[i] = best_cut

        cuts: List[int] = []
        position = n
        while position > 0:
            previous = backpointer[position]
            if previous is None:
                break
            if previous != 0:
                cuts.append(previous)
            position = previous

        return list(reversed(cuts)), dp[n]

    @staticmethod
    def decode_segments(sequence: Sequence[str], cuts: Sequence[int]) -> List[Sequence[str]]:
        segments: List[Sequence[str]] = []
        start = 0
        for cut in cuts:
            segments.append(sequence[start:cut])
            start = cut
        segments.append(sequence[start:])
        return segments
