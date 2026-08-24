"""
Módulo principal do Algoritmo Genético para otimização de hiperparâmetros.

Implementa os operadores genéticos (seleção por torneio, cruzamento uniforme
e mutação por gene) e o loop evolutivo principal, com logging estruturado
do progresso a cada geração.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .fitness import SEARCH_SPACE, evaluate_fitness, sample_individual


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """Representa um indivíduo (solução candidata) na população do AG."""
    model_name: str
    genes: dict
    fitness: float = 0.0


@dataclass
class GenerationStats:
    """Estatísticas de uma única geração do AG."""
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_genes: dict
    elapsed_seconds: float


@dataclass
class ExperimentResult:
    """Resultado completo de um experimento do AG."""
    model_name: str
    config: dict
    history: list[GenerationStats] = field(default_factory=list)
    best_individual: Optional[Individual] = None
    baseline_fitness: float = 0.0
    total_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

def _tournament_selection(
    population: list[Individual],
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    """
    Seleção por torneio: sorteia tournament_size indivíduos aleatoriamente
    e retorna o de maior fitness.
    """
    contestants = rng.choice(len(population), size=tournament_size, replace=False)
    winner = max(contestants, key=lambda i: population[i].fitness)
    return population[winner]


def _uniform_crossover(
    parent_a: Individual,
    parent_b: Individual,
    crossover_rate: float,
    rng: np.random.Generator,
) -> Individual:
    """
    Cruzamento uniforme: cada gene do filho é herdado do pai A com
    probabilidade crossover_rate, ou do pai B caso contrário.
    """
    child_genes = {}
    for param in parent_a.genes:
        if rng.random() < crossover_rate:
            child_genes[param] = parent_a.genes[param]
        else:
            child_genes[param] = parent_b.genes[param]
    return Individual(model_name=parent_a.model_name, genes=child_genes)


def _mutate(
    individual: Individual,
    mutation_rate: float,
    rng: np.random.Generator,
) -> Individual:
    """
    Mutação por gene: cada gene é substituído por um valor aleatório novo
    com probabilidade mutation_rate.
    """
    mutated_genes = dict(individual.genes)
    for param in SEARCH_SPACE[individual.model_name]:
        if rng.random() < mutation_rate:
            temp = sample_individual(individual.model_name, rng)
            mutated_genes[param] = temp[param]
    individual.genes = mutated_genes
    return individual


# ---------------------------------------------------------------------------
# Loop evolutivo principal
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 20,
    n_generations: int = 20,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.7,
    tournament_size: int = 3,
    elitism: int = 2,
    cv_folds: int = 5,
    baseline_fitness: float = 0.0,
    output_dir: Optional[Path] = None,
    random_seed: int = 42,
) -> ExperimentResult:
    """
    Executa o Algoritmo Genético para otimizar os hiperparâmetros do modelo.

    Parameters
    ----------
    model_name : str
        Nome do modelo ('logistic_regression', 'random_forest', 'knn', 'svm').
    X : np.ndarray
        Features não pré-escaladas.
    y : np.ndarray
        Target binário (1 = Maligno, 0 = Benigno).
    population_size : int
        Número de indivíduos na população.
    n_generations : int
        Número de gerações a evoluir.
    mutation_rate : float
        Probabilidade de mutação por gene.
    crossover_rate : float
        Probabilidade de herdar gene do pai A no cruzamento.
    tournament_size : int
        Número de competidores em cada torneio.
    elitism : int
        Número de melhores indivíduos preservados a cada geração.
    cv_folds : int
        Folds da validação cruzada.
    baseline_fitness : float
        Recall do modelo original (Fase 1) para comparação.
    output_dir : Path, optional
        Diretório para salvar resultado em JSON.
    random_seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    ExperimentResult com histórico completo e melhor indivíduo.
    """
    rng = np.random.default_rng(random_seed)
    config = {
        "model_name": model_name,
        "population_size": population_size,
        "n_generations": n_generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "tournament_size": tournament_size,
        "elitism": elitism,
        "cv_folds": cv_folds,
        "random_seed": random_seed,
    }

    result = ExperimentResult(
        model_name=model_name,
        config=config,
        baseline_fitness=baseline_fitness,
    )

    experiment_start = time.time()

    logger.info(
        f"[AG] Iniciando | modelo={model_name} | pop={population_size} "
        f"| gerações={n_generations} | mutação={mutation_rate}"
    )

    # --- Geração inicial ---
    population = []
    for _ in range(population_size):
        genes = sample_individual(model_name, rng)
        fitness = evaluate_fitness(model_name, genes, X, y, cv_folds=cv_folds)
        population.append(Individual(model_name=model_name, genes=genes, fitness=fitness))

    # --- Loop evolutivo ---
    for gen in range(1, n_generations + 1):
        gen_start = time.time()

        population.sort(key=lambda ind: ind.fitness, reverse=True)

        fitnesses = [ind.fitness for ind in population]
        stats = GenerationStats(
            generation=gen,
            best_fitness=fitnesses[0],
            mean_fitness=float(np.mean(fitnesses)),
            worst_fitness=fitnesses[-1],
            best_genes=dict(population[0].genes),
            elapsed_seconds=round(time.time() - gen_start, 2),
        )
        result.history.append(stats)

        logger.info(
            f"[AG] Geração {gen:3d}/{n_generations} | "
            f"melhor={stats.best_fitness:.4f} | "
            f"média={stats.mean_fitness:.4f} | "
            f"pior={stats.worst_fitness:.4f}"
        )

        if stats.best_fitness >= 1.0:
            logger.info("[AG] Fitness perfeito atingido — encerrando cedo.")
            break

        # --- Próxima geração ---
        next_gen: list[Individual] = []

        # Elitismo
        next_gen.extend(population[:elitism])

        # Filhos via cruzamento + mutação
        while len(next_gen) < population_size:
            parent_a = _tournament_selection(population, tournament_size, rng)
            parent_b = _tournament_selection(population, tournament_size, rng)
            child = _uniform_crossover(parent_a, parent_b, crossover_rate, rng)
            child = _mutate(child, mutation_rate, rng)
            child.fitness = evaluate_fitness(
                model_name, child.genes, X, y, cv_folds=cv_folds
            )
            next_gen.append(child)

        population = next_gen

    # Resultado final
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    result.best_individual = population[0]
    result.total_seconds = round(time.time() - experiment_start, 2)

    logger.info(
        f"[AG] Concluído em {result.total_seconds}s | "
        f"melhor recall={result.best_individual.fitness:.4f} | "
        f"baseline={baseline_fitness:.4f} | "
        f"ganho={result.best_individual.fitness - baseline_fitness:+.4f}"
    )

    # Salvar JSON se output_dir fornecido
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_name}_result.json"
        _save_result(result, output_path)
        logger.info(f"[AG] Resultado salvo em {output_path}")

    return result


def _save_result(result: ExperimentResult, path: Path) -> None:
    """Serializa ExperimentResult para JSON."""
    data = {
        "model_name": result.model_name,
        "config": result.config,
        "baseline_fitness": result.baseline_fitness,
        "total_seconds": result.total_seconds,
        "best_individual": {
            "genes": result.best_individual.genes,
            "fitness": result.best_individual.fitness,
        } if result.best_individual else None,
        "history": [
            {
                "generation": s.generation,
                "best_fitness": s.best_fitness,
                "mean_fitness": s.mean_fitness,
                "worst_fitness": s.worst_fitness,
                "best_genes": s.best_genes,
                "elapsed_seconds": s.elapsed_seconds,
            }
            for s in result.history
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
