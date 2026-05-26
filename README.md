# CohereSeg

Proyecto base para segmentación óptima de contenido basada en coherencia semántica.

## Estructura del proyecto

- `data_loader.py`: carga datasets desde archivos JSON o CSV y convierte el contenido en listas de elementos.
- `llm.py`: interfaz con un LLM para evaluar la coherencia de segmentos. Incluye caché local y una métrica alternativa basada en embeddings.
- `dp.py`: implementa la programación dinámica exacta para encontrar la segmentación óptima.
- `metaheuristics.py`: implementa las bases de Hill Climbing, Simulated Annealing y Algoritmo Genético.
- `experiments.ipynb`: notebook de ejemplo para ejecutar pruebas y analizar resultados.

## Requisitos

- `openai` para consultas al LLM.


## Uso básico

1. Cargar una secuencia desde un archivo con `DataLoader`.
2. Evaluar segmentos con `LlmEvaluator`.
3. Calcular la segmentación exacta con `OptimalSegmentationDP` o buscar aproximaciones con las metaheurísticas.

## Ejemplo rápido

```python
from data_loader import DataLoader
from llm import LlmEvaluator
from dp import OptimalSegmentationDP
from metaheuristics import HillClimbing

loader = DataLoader()
sequence = ["Fragmento uno.", "Fragmento dos.", "Fragmento tres."]
evaluator = LlmEvaluator(model_name="gpt-4", cache_path="llm_cache.json")

solver = OptimalSegmentationDP(evaluator)
cuts, score = solver.solve(sequence)
print("Cortes:", cuts, "Puntuación:", score)

hc = HillClimbing(evaluator)
solution = hc.optimize(sequence)
print(solution)
```

## Notas

- Configure `OPENAI_API_KEY` para usar el LLM de OpenAI.
- La caché se guarda en `llm_cache.json`.
