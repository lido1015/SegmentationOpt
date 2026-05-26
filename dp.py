from typing import List, Tuple
from llm import get_coherence


def dp_solution(
    contents: List[str],
    unit_coherence: float = 1.0,
    alpha: float = 0.5
) -> Tuple[List[int], float]:

    n = len(contents)
    if n <= 1:
        return [], -float('inf')  # No hay posibles cortes

    # dp[i] = mejor puntuación para los primeros i elementos
    dp = [-float('inf')] * (n + 1)
    dp[0] = 0.0
    # from[i] = inicio del último segmento en la solución óptima para prefijo i
    from_idx = [0] * (n + 1)

    # Llenado de la tabla DP (O(n^2))
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