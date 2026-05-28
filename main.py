import json
from typing import List

from metaheuristics import (
    GeneticAlgorithm,
    HillClimbing,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
)
from problem import ContentSegmentationProblem


def load_first_instance() -> List[str]:
    with open("data/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset[0]["contents"]


def run_hill_climbing(contents: List[str], unit_coherence: float = 0.6, alpha: float = 1.0):
    print("\n[Hill Climbing]")
    print("-" * 50)
    problem = ContentSegmentationProblem(contents, unit_coherence=unit_coherence, alpha=alpha)
    solver = HillClimbing(max_evaluations=1000, neighborhood_size=2, stagnation_limit=100)
    cuts, score, state = solver.solve(problem, verbose=False)
    print(f"Fitness: {score:.4f}")
    print(f"Cortes: {cuts}")
    print(f"Evaluaciones: {state['evaluations']}")
    print(f"Iteraciones: {state['iteration']}")


def run_simulated_annealing(contents: List[str], unit_coherence: float = 0.6, alpha: float = 1.0):
    print("\n[Simulated Annealing]")
    print("-" * 50)
    problem = ContentSegmentationProblem(contents, unit_coherence=unit_coherence, alpha=alpha)
    solver = SimulatedAnnealing(
        T0=100.0,
        alpha=0.95,
        iters_per_temp=10,
        T_min=1e-3,
        max_evaluations=1000,
    )
    cuts, score, state = solver.solve(problem, verbose=False)
    print(f"Fitness: {score:.4f}")
    print(f"Cortes: {cuts}")
    print(f"Evaluaciones: {state['evaluations']}")
    print(f"Iteraciones: {state['iteration']}")


def run_genetic_algorithm(contents: List[str], unit_coherence: float = 0.6, alpha: float = 1.0):
    print("\n[Genetic Algorithm]")
    print("-" * 50)
    problem = ContentSegmentationProblem(contents, unit_coherence=unit_coherence, alpha=alpha)
    solver = GeneticAlgorithm(
        population_size=50,
        mutation_rate=0.05,
        elitism=2,
        max_generations=50,
        max_evaluations=1000,
    )
    cuts, score, state = solver.solve(problem, verbose=False)
    print(f"Fitness: {score:.4f}")
    print(f"Cortes: {cuts}")
    print(f"Evaluaciones: {state['evaluations']}")
    print(f"Generaciones: {state['generation']}")


def run_particle_swarm(contents: List[str], unit_coherence: float = 0.6, alpha: float = 1.0):
    print("\n[Particle Swarm Optimization]")
    print("-" * 50)
    problem = ContentSegmentationProblem(contents, unit_coherence=unit_coherence, alpha=alpha)
    solver = ParticleSwarmOptimization(
        population_size=30,
        max_generations=50,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_evaluations=1000,
    )
    cuts, score, state = solver.solve(problem, verbose=False)
    print(f"Fitness: {score:.4f}")
    print(f"Cortes: {cuts}")
    print(f"Evaluaciones: {state['evaluations']}")
    print(f"Generaciones: {state['generation']}")


def main() -> None:
    contents = load_first_instance()
    print(f"Instancia cargada: {len(contents)} fragmentos")
    print("=" * 50)
    print(f"Ejecutando los 4 algoritmos con parámetros óptimos")
    print("=" * 50)

    run_hill_climbing(contents)
    run_simulated_annealing(contents)
    run_genetic_algorithm(contents)
    run_particle_swarm(contents)

    print("\n" + "=" * 50)
    print("Ejecución completada")


if __name__ == "__main__":
    main()
