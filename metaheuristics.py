"""
Metaheuristic Framework (refactored with standardized metrics)
---------------------------------------------------------------
- Common state dictionary 'self.state' with standard counters and history.
- solve() returns (best_solution, best_fitness, state).
- Subclasses override _initialize_state and _update_state, call super(),
  and add algorithm‑specific keys.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field


from problem import Problem


# =============================================================================
# 1. Top‑level abstract metaheuristic
# =============================================================================
class Metaheuristic(ABC):
    """Common interface for every metaheuristic."""

    @abstractmethod
    def solve(self, problem: Problem, verbose: bool = False) -> Tuple[Any, float, Dict]:
        """
        Run the search.
        Returns (best_solution, best_fitness, state) where state
        is a dictionary with standard metrics and algorithm‑specific data.
        """
        ...


# =============================================================================
# 2. Single‑solution metaheuristic template
# =============================================================================
class SingleSolutionMetaheuristic(Metaheuristic):
    """
    Template for single‑solution metaheuristics.
    Provides concrete state initialisation and update with standard metrics.
    Subclasses must implement:
        _generate, _accept, _termination
    and may override _initialize_state and _update_state
    (calling the superclass implementation).
    """

    # ------------------------------------------------------------------
    # Concrete state management (standard metrics)
    # ------------------------------------------------------------------
    def _initialize_state(self) -> Dict[str, Any]:
        """Return a state dictionary with common counters and history."""
        return {
            "iteration": 0,
            "evaluations": 0,
            "best_fitness": float("-inf"),
            "history_best_fitness": [],      # best fitness after each iteration
        }

    def _update_state(self,
                      evaluations_added: int,
                      candidate: Any,
                      candidate_fit: float,
                      accepted: bool) -> None:
        """
        Update standard state.
        Subclasses **must** call super()._update_state(...) if overridden.
        """
        self.state["iteration"] += 1
        self.state["evaluations"] += evaluations_added
        self.state["history_best_fitness"].append(self.state["best_fitness"])

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------
    @abstractmethod
    def _generate(self, current: Any) -> Any:
        """Produce a new candidate solution from the current one."""
        ...

    @abstractmethod
    def _accept(self,
                candidate: Any, candidate_fit: float,
                current: Any, current_fit: float) -> bool:
        """Decide whether the candidate should replace the current solution."""
        ...

    @abstractmethod
    def _termination(self) -> bool:
        """Return True when the search should stop. Use self.state."""
        ...

    # ------------------------------------------------------------------
    # Template method – DO NOT OVERRIDE
    # ------------------------------------------------------------------
    def solve(self, problem: Problem, verbose: bool = False) -> Tuple[Any, float, Dict]:
        """Run the single‑solution search."""
        self._problem = problem

        # 1. Initial solution
        current = problem.create_solution()
        current_fit = problem.objective_function(current)
        best = current
        best_fit = current_fit

        # 2. Standardised state (metrics & algorithm‑specific)
        self.state = self._initialize_state()
        self.state["evaluations"] = 1                # the initial evaluation
        self.state["best_fitness"] = best_fit

        # 3. Main loop
        while not self._termination():
            candidate, candidate_fit, evals = self._generate(current)


            accepted = self._accept(candidate, candidate_fit,
                                    current, current_fit)

            if accepted:
                current = candidate
                current_fit = candidate_fit

            # Update state with standard counters and history
            self._update_state(evals, candidate, candidate_fit, accepted)

            # Update best
            if current_fit > best_fit:
                best = current
                best_fit = current_fit
                self.state["best_fitness"] = best_fit

        return best, best_fit, self.state


# =============================================================================
# 3. Population‑based metaheuristic template
# =============================================================================
class PopulationMetaheuristic(Metaheuristic):
    """
    Template for population‑based metaheuristics.
    Provides concrete state initialisation and update with standard metrics.
    Subclasses must implement:
        _initialize_population, _select, _breed, _replace, _termination
    and may override _initialize_state and _update_state
    (calling the superclass implementation).
    """

    # ------------------------------------------------------------------
    # Concrete state management (standard metrics)
    # ------------------------------------------------------------------
    def _initialize_state(self) -> Dict[str, Any]:
        """Return a state dictionary with common counters and history."""
        return {
            "generation": 0,
            "evaluations": 0,
            "best_fitness": float("-inf"),
            "history_best_fitness": [],
        }

    def _update_state(self,
                      evaluations_added: int) -> None:
        """
        Update standard state after one generation.
        Subclasses **must** call super()._update_state(...) if overridden.
        """
        self.state["generation"] += 1
        self.state["evaluations"] += evaluations_added
        self.state["history_best_fitness"].append(self.state["best_fitness"])

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------
    @abstractmethod
    def _initialize_population(self) -> Tuple[List[Any], List[float]]:
        """Create and evaluate the initial population."""
        ...

    @abstractmethod
    def _select(self, population: List[Any], fitness: List[float]) -> List[Any]:
        """Choose parent solutions for breeding or update."""
        ...

    @abstractmethod
    def _breed(self, selected: List[Any]) -> List[Any]:
        """Generate offspring from selected parents."""
        ...

    @abstractmethod
    def _replace(self,
                 population: List[Any],
                 fitness: List[float],
                 children: List[Any],
                 children_fitness: List[float]) -> Tuple[List[Any], List[float]]:
        """Replace the current population with a new one."""
        ...

    @abstractmethod
    def _termination(self) -> bool:
        """Access self.state for generation, evaluations, best_fitness."""
        ...

    # ------------------------------------------------------------------
    # Template method – DO NOT OVERRIDE
    # ------------------------------------------------------------------
    def solve(self, problem: Problem, verbose: bool = False) -> Tuple[Any, float, Dict]:
        """Run the population‑based search."""
        self._problem = problem
        self.state = self._initialize_state()

        # 1. Initial population
        population, fitness = self._initialize_population()
        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        best, best_fit = population[best_idx], fitness[best_idx]

        # 2. Standard state

        self.state["evaluations"] = len(population)
        self.state["best_fitness"] = best_fit

        # 3. Evolution
        while not self._termination():
            parents = self._select(population, fitness)
            children = self._breed(parents)
            children_fitness = [problem.objective_function(c) for c in children]

            population, fitness = self._replace(population, fitness,
                                                children, children_fitness)

            # Update standard state (generation, evals, history)
            self._update_state(len(children))

            # Update best so far
            current_best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
            if fitness[current_best_idx] > best_fit:
                best = population[current_best_idx]
                best_fit = fitness[current_best_idx]
                self.state["best_fitness"] = best_fit


        return best, best_fit, self.state


# =============================================================================
# 4. Concrete single‑solution algorithms
# =============================================================================

class HillClimbing(SingleSolutionMetaheuristic):
    """Hill climbing with first-improvement search and stagnation control."""

    def __init__(self, stagnation_limit: int = 100, max_evaluations: int = 1000, neighborhood_size: int = 1):
        """Initialize the hill climbing algorithm parameters."""
        self.stagnation_limit = stagnation_limit
        self.max_evaluations = max_evaluations
        self.neighborhood_size = neighborhood_size

    def _initialize_state(self) -> Dict[str, Any]:
        """Extend common state with the last improvement iteration."""
        state = super()._initialize_state()
        state["last_improvement_iter"] = 0
        return state

    def _generate(self, current) -> Tuple[Any, float, int]:
        """Create a neighbor candidate and return its fitness and evaluation count."""
        candidate, candidate_fit = self._problem.get_neighbor(current, self.neighborhood_size)
        return candidate, candidate_fit, self.neighborhood_size

    def _accept(self, candidate, candidate_fit, current, current_fit) -> bool:
        """Accept the candidate if it has higher fitness than the current solution."""
        return candidate_fit > current_fit

    def _update_state(self, evaluations_added, candidate, candidate_fit, accepted):
        """Update state and record the last improvement iteration."""
        super()._update_state(evaluations_added, candidate, candidate_fit, accepted)
        if accepted:
            self.state["last_improvement_iter"] = self.state["iteration"]

    def _termination(self) -> bool:
        """Stop when evaluation or stagnation limits are reached."""
        if self.state["evaluations"] >= self.max_evaluations:
            return True
        if (
            self.state["iteration"] - self.state["last_improvement_iter"]
            >= self.stagnation_limit
        ):
            return True
        return False


class SimulatedAnnealing(SingleSolutionMetaheuristic):
    """Simulated annealing with geometric cooling and probabilistic acceptance."""

    def __init__(self,
                 T0: float = 100.0,
                 alpha: float = 0.95,
                 iters_per_temp: int = 10,
                 T_min: float = 1e-3,
                 max_evaluations: int = 10000):
        """Configure the annealing schedule and stopping criteria."""
        self.T0 = T0
        self.alpha = alpha
        self.iters_per_temp = iters_per_temp
        self.T_min = T_min
        self.max_evaluations = max_evaluations


    def _initialize_state(self) -> Dict[str, Any]:
        """Extend standard state with current temperature and cooling counter."""
        state = super()._initialize_state()
        state["T"] = self.T0
        state["since_last_cooling"] = 0
        return state

    def _generate(self, current) -> Tuple[Any, float, int]:
        """Generate one neighboring candidate for annealing."""
        candidate, candidate_fit = self._problem.get_neighbor(current, 1)
        return candidate, candidate_fit, 1

    def _accept(self, candidate, candidate_fit, current, current_fit) -> bool:
        """Accept the candidate based on Metropolis criterion."""
        delta = candidate_fit - current_fit   # maximisation
        if delta >= 0:
            return True
        T = self.state["T"]
        if T <= 0:
            return False
        return random.random() < math.exp(delta / T)

    def _update_state(self, evaluations_added, candidate, candidate_fit, accepted):
        """Update annealing state and apply temperature cooling periodically."""
        super()._update_state(evaluations_added, candidate, candidate_fit, accepted)
        self.state["since_last_cooling"] += 1
        if self.state["since_last_cooling"] >= self.iters_per_temp:
            self.state["T"] *= self.alpha
            self.state["since_last_cooling"] = 0
            if self.state["T"] < self.T_min:
                self.state["T"] = 0.0

    def _termination(self) -> bool:
        """Stop when temperature is too low or evaluation budget is exhausted."""
        return (
            self.state["T"] <= self.T_min or
            self.state["evaluations"] >= self.max_evaluations
        )


# =============================================================================
# 5. Concrete population‑based algorithm: Genetic Algorithm
# =============================================================================

class GeneticAlgorithm(PopulationMetaheuristic):
    """Canonical GA with binary tournament selection, elitism, and mutation."""

    def __init__(self,
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 elitism: int = 2,
                 max_generations: int = 500,
                 max_evaluations: int = 10000):
        """Initialize genetic algorithm parameters."""
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.max_generations = max_generations
        self.max_evals = max_evaluations

    # ------------------------------------------------------------------
    # Primitive implementations
    # ------------------------------------------------------------------
    def _initialize_population(self):
        """Create the initial population and evaluate each individual."""
        pop = [self._problem.create_solution() for _ in range(self.pop_size)]
        fit = [self._problem.objective_function(ind) for ind in pop]
        return pop, fit

    def _select(self, population, fitness):
        """Select parents by binary tournament for the next breeding step."""
        selected = []
        for _ in range(self.pop_size - self.elitism):
            i, j = random.sample(range(len(population)), 2)
            winner = population[i] if fitness[i] > fitness[j] else population[j]
            selected.append(winner)
        return selected

    def _breed(self, selected):
        """Combine selected parents into offspring and optionally mutate them."""
        children = []
        parents = selected[:]
        random.shuffle(parents)

        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            child1, child2 = self._problem.crossover(p1, p2)

            if self.mutation_rate > 0 and random.random() < self.mutation_rate:
                self._problem.mutate(child1)
            if self.mutation_rate > 0 and random.random() < self.mutation_rate:
                self._problem.mutate(child2)

            children.append(child1)
            children.append(child2)

        if len(parents) % 2 == 1:
            last = parents[-1]
            child = last.copy() if hasattr(last, 'copy') else last
            if self.mutation_rate > 0 and random.random() < self.mutation_rate:
                self._problem.mutate(child)
            children.append(child)

        return children

    def _replace(self, population, fitness, children, children_fitness):
        """Build the new population using elitism and the best children."""
        elite_idx = sorted(range(len(fitness)),
                           key=lambda i: fitness[i], reverse=True)[:self.elitism]
        new_pop = [population[i] for i in elite_idx]
        new_fit = [fitness[i] for i in elite_idx]

        sorted_children = sorted(zip(children, children_fitness),
                                 key=lambda x: x[1], reverse=True)
        needed = self.pop_size - self.elitism
        for child, cfit in sorted_children[:needed]:
            new_pop.append(child)
            new_fit.append(cfit)

        while len(new_pop) < self.pop_size:
            new_pop.append(new_pop[-1])
            new_fit.append(new_fit[-1])

        return new_pop, new_fit

    def _termination(self) -> bool:
        """Stop when the generation or evaluation budget is exhausted."""
        return (self.state["generation"] >= self.max_generations or
                self.state["evaluations"] >= self.max_evals)


@dataclass
class Particle:
    """Particle state for binary particle swarm optimization."""
    position: List[int]
    velocity: List[float]
    pbest_position: List[int] = field(default_factory=list)
    pbest_fitness: float = -float('inf')
    current_fitness: float = -float('inf')


class ParticleSwarmOptimization(PopulationMetaheuristic):
    """Particle Swarm Optimization for binary cut-vector problems."""

    def __init__(self,
                 population_size: int = 30,
                 max_generations: int = 100,
                 w: float = 0.7,   # Inertia weight
                 c1: float = 1.5,  # Cognitive coefficient
                 c2: float = 1.5,  # Social coefficient
                 max_evaluations: int = 10000,
                 v_max: float = 6.0 # Max velocity for binary PSO sigmoid
                ):
        """Set PSO parameters including inertia, cognitive and social weights."""
        self.pop_size = population_size
        self.max_generations = max_generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_evals = max_evaluations
        self.v_max = v_max # Max velocity for binary PSO, velocities are clipped to [-v_max, v_max]

    def _initialize_state(self) -> Dict[str, Any]:
        """Extend standard state with global best and particle storage."""
        state = super()._initialize_state()
        state["gbest_position"] = None
        state["gbest_fitness"] = -float('inf')
        state["population_of_particles"] = [] # Store actual particle objects
        return state

    def _initialize_population(self) -> Tuple[List[Any], List[float]]:
        """Create an initial swarm and evaluate each particle."""
        # Initialize particles and their pbest_positions
        initial_solutions = []
        initial_fitnesses = []
        gbest_fitness_found = -float('inf')
        gbest_position_found = None

        population_of_particles: List[Particle] = []

        for _ in range(self.pop_size):
            position = self._problem.create_solution() # Use the default heuristic solution here
            current_fitness = self._problem.objective_function(position)

            # Initialize velocity (e.g., random between -v_max and v_max)
            velocity = [random.uniform(-self.v_max, self.v_max) for _ in range(len(position))]

            particle = Particle(
                position=position,
                velocity=velocity,
                pbest_position=position.copy(),
                pbest_fitness=current_fitness,
                current_fitness=current_fitness
            )
            population_of_particles.append(particle)

            initial_solutions.append(position)
            initial_fitnesses.append(current_fitness)

            # Update global best if this particle is better
            if current_fitness > gbest_fitness_found:
                gbest_fitness_found = current_fitness
                gbest_position_found = position.copy()

        self.state["gbest_position"] = gbest_position_found
        self.state["gbest_fitness"] = gbest_fitness_found
        self.state["population_of_particles"] = population_of_particles # Store actual particle objects
        return initial_solutions, initial_fitnesses

    def _select(self, population: List[Any], fitness: List[float]) -> List[Particle]:
        """Update the global best and return the current particles for update."""
        # In PSO, selection is implicit in particle updates. We update gbest and return particles for 'breeding'.
        current_gbest_fitness = self.state["gbest_fitness"]
        current_gbest_position = self.state["gbest_position"]

        particles = self.state["population_of_particles"]

        for i, particle in enumerate(particles):
            if particle.current_fitness > current_gbest_fitness:
                current_gbest_fitness = particle.current_fitness
                current_gbest_position = particle.position.copy() # Ensure deep copy

        self.state["gbest_fitness"] = current_gbest_fitness
        self.state["gbest_position"] = current_gbest_position
        return particles # Return the actual particle objects for breeding


    def _breed(self, selected: List[Particle]) -> List[List[int]]:
        """Update particle velocities and positions using PSO equations."""
        # This is where velocities and positions are updated for each particle
        new_positions = []
        gbest_position = self.state["gbest_position"]

        for particle in selected:
            new_velocity = [0.0] * len(particle.position)
            new_position_cuts = [0] * len(particle.position)

            for i in range(len(particle.position)):
                r1 = random.random()
                r2 = random.random()

                # Update velocity components
                cognitive_component = self.c1 * r1 * (particle.pbest_position[i] - particle.position[i])
                social_component = self.c2 * r2 * (gbest_position[i] - particle.position[i])

                new_velocity[i] = self.w * particle.velocity[i] + cognitive_component + social_component

                # Clamp velocity to V_max to prevent explosion
                if new_velocity[i] > self.v_max:
                    new_velocity[i] = self.v_max
                elif new_velocity[i] < -self.v_max:
                    new_velocity[i] = -self.v_max

                # Binary PSO: Use sigmoid to map velocity to probability, then update position
                # Sigmoid function: 1 / (1 + exp(-v))
                prob_one = 1.0 / (1.0 + math.exp(-new_velocity[i]))
                if random.random() < prob_one:
                    new_position_cuts[i] = 1
                else:
                    new_position_cuts[i] = 0

            particle.velocity = new_velocity
            new_positions.append(new_position_cuts)

        return new_positions # These are the "children" in the PopulationMetaheuristic sense


    def _replace(
        self,
        population: List[List[int]], # These are the old positions from the PopulationMetaheuristic.solve loop
        fitness: List[float],
        children: List[List[int]], # These are the new positions (from _breed) evaluated by solve loop
        children_fitness: List[float]
    ) -> Tuple[List[List[int]], List[float]]:
        """Update each particle with its new position and personal best."""

        updated_population_solutions = []
        updated_population_fitnesses = []

        # Access the actual Particle objects from the state
        particles = self.state["population_of_particles"]

        for i, particle in enumerate(particles):
            # Update particle's current position and fitness
            particle.position = children[i]
            particle.current_fitness = children_fitness[i]

            # Update particle's personal best
            if particle.current_fitness > particle.pbest_fitness:
                particle.pbest_fitness = particle.current_fitness
                particle.pbest_position = particle.position.copy()

            updated_population_solutions.append(particle.position)
            updated_population_fitnesses.append(particle.current_fitness)

        return updated_population_solutions, updated_population_fitnesses

    def _termination(self) -> bool:
        """Stop PSO when generation or evaluation limits are reached."""
        return (
            self.state["generation"] >= self.max_generations or
            self.state["evaluations"] >= self.max_evals
        )