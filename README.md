# SegmentationOpt

Proyecto base para la optimización de segmentación de texto mediante coherencia semántica.

## Estructura del proyecto

- `main.py`: punto de entrada para ejecutar soluciones exactas y metaheurísticas desde la línea de comandos.
- `llm.py`: funciones para evaluar la coherencia de segmentos con un LLM externo. Incluye caché local y gestión de claves.
- `dp.py`: solución exacta por programación dinámica para encontrar la segmentación óptima.
- `problem.py`: definición de la interfaz `Problem` y el problema concreto de segmentación de contenido.
- `metaheuristics.py`: implementa plantillas y algoritmos concretos como Hill Climbing, Simulated Annealing, Genetic Algorithm y Particle Swarm Optimization.
- `experiments.ipynb`: notebook para experimentar y analizar resultados.

## Requisitos

- Python 3.10+.
- Dependencias del proyecto:
  - `google-genai`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `pandas`

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

## Configuración del LLM

Este proyecto usa el paquete `google-genai` con claves de API de Google Gen AI.

1. Crea un archivo `keys.txt` en la raíz del proyecto.
2. Añade una clave de Google Gen AI por línea. El código usa estas claves para autenticar las llamadas a Gemini y rotarlas si se recibe un error 429.

Ejemplo de `keys.txt`:

```text
API_KEY_DE_GOOGLE_GEN_AI_1
API_KEY_DE_GOOGLE_GEN_AI_2
```

3. El modelo configurado por defecto en `llm.py` es `gemini-3.1-flash-lite`. Cambia `MODEL_NAME` si quieres usar otro modelo compatible.
4. La caché de evaluaciones se guarda en `data/llm_cache.json` para evitar llamadas repetidas al modelo.

## Ejecución

### Ejecutar los algoritmos sobre el dataset

El archivo `main.py` ejecuta los cuatro algoritmos metaheurísticos sobre la primera instancia del dataset.json:

- **Hill Climbing**
- **Simulated Annealing**
- **Genetic Algorithm**
- **Particle Swarm Optimization**

Cada algoritmo se ejecuta con parámetros óptimos preconfigurados:

```bash
python3 main.py
```

La salida muestra para cada algoritmo:
- Fitness final
- Cortes encontrados
- Número de evaluaciones
- Número de iteraciones o generaciones


## Configuración del problema

- `ContentSegmentationProblem` ahora acepta parámetros `unit_coherence` y `alpha` para que el mismo problema pueda evaluarse con distintas configuraciones.
- `dp.py` permite ejecutar la estrategia exacta de programación dinámica con parámetros `unit_coherence` y `alpha`.

## Ejemplo de uso en código

```python
from dp import dp_solution
from problem import ContentSegmentationProblem
from metaheuristics import HillClimbing

sequence = ["Fragmento uno.", "Fragmento dos.", "Fragmento tres."]

cuts, score = dp_solution(sequence, unit_coherence=0.7, alpha=1.0)
print(cuts, score)

problem = ContentSegmentationProblem(sequence, unit_coherence=0.7, alpha=1.0)
solver = HillClimbing(max_evaluations=500, neighborhood_size=2)
best_solution, best_score, state = solver.solve(problem)
print(best_solution, best_score)
```

## Notas

- `keys.txt` debe existir en la raíz del proyecto.
- El archivo de caché `data/llm_cache.json` se genera automáticamente.
- Usa `python main.py --method dp` para comparar resultados exactos con las metaheurísticas.

