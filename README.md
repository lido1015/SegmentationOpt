# SegmentationOpt

Proyecto base para la optimización de segmentación de texto mediante coherencia semántica.

## Estructura del proyecto

- `llm.py`: funciones para evaluar la coherencia de segmentos con un LLM externo. Incluye caché local y gestión de claves.
- `dp.py`: solución exacta por programación dinámica para encontrar la segmentación óptima.
- `problem.py`: definición de la interfaz `Problem` y el problema concreto de segmentación de contenido.
- `metaheuristics.py`: implementa plantillas y algoritmos concretos como Hill Climbing, Simulated Annealing, Genetic Algorithm y Particle Swarm Optimization.
- `experiments.ipynb`: notebook para experimentar y analizar resultados.

## Requisitos

- `google-genai` o la biblioteca necesaria para acceder a Gemini y otros modelos de Google.
- Un archivo `keys.txt` con las claves de API válidas, una clave por línea.

## Uso básico

1. Preparar una lista de fragmentos de texto.
2. Crear un `ContentSegmentationProblem` con esos fragmentos.
3. Usar `dp_solution` para obtener la segmentación exacta o aplicar una metaheurística.

## Ejemplo rápido

```python
from problem import ContentSegmentationProblem
from dp import dp_solution
from metaheuristics import HillClimbing

sequence = ["Fragmento uno.", "Fragmento dos.", "Fragmento tres."]

# Segmentación exacta con programación dinámica
cuts, score = dp_solution(sequence, unit_coherence=1.0, alpha=0.5)
print("Cortes exactos:", cuts)
print("Puntuación:", score)

# Optimización aproximada con Hill Climbing
problem = ContentSegmentationProblem(sequence)
solver = HillClimbing(stagnation_limit=50, max_evaluations=500, neighborhood_size=2)
best_solution, best_score, state = solver.solve(problem, verbose=True)
print("Cortes aproximados:", best_solution)
print("Puntuación:", best_score)
```

## Notas

- El archivo `keys.txt` debe contener las claves de acceso para la API de LLM.
- La caché de evaluaciones se guarda en `data/llm_cache.json`.
- `dp_solution` opera sobre una lista de fragmentos y retorna un vector de cortes binario.
