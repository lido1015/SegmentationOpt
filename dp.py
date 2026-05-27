"""Exact dynamic programming solver for segmenting text content.

This module implements a simple O(n^2) dynamic programming solution
for optimal segmentation based on segment coherence scores.
"""

from typing import List, Tuple
from llm import get_coherence


def dp_solution(
    contents: List[str],
    unit_coherence: float = 1.0,
    alpha: float = 0.5
) -> Tuple[List[int], float]:
    """Compute the optimal segmentation and score using dynamic programming.

    Args:
        contents: list of text fragments.
        unit_coherence: coherence score for singleton segments.
        alpha: penalty added for each segment created.

    Returns:
        A tuple with the optimal cut vector and the best overall score.
    """

    n = len(contents)
    if n <= 1:
        return [], -float('inf')

    # dp[i] = best score for the first i fragments
    dp = [-float('inf')] * (n + 1)
    dp[0] = 0.0
    # from_idx[i] = start index of the last segment in the optimal prefix ending at i
    from_idx = [0] * (n + 1)

    # Fill DP table in O(n^2)
    for i in range(1, n + 1):
        for j in range(i):
            coherence = get_coherence(contents, j, i, unit_coherence)
            value = dp[j] + (coherence - alpha)
            if value > dp[i]:
                dp[i] = value
                from_idx[i] = j

    # Reconstrucción de los cortes
    cuts = [0] * (n - 1)
    i = n
    while i > 0:
        j = from_idx[i]          # inicio del último segmento
        if j > 0:
            cuts[j - 1] = 1      # corte después del elemento j-1
        i = j

    return cuts, dp[n]