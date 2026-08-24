"""
Módulo de monitoramento e logging do Algoritmo Genético.

Responsável por:
- Configurar o logging estruturado do projeto
- Gerar gráficos de evolução do fitness ao longo das gerações
- Produzir tabelas comparativas entre baseline (Fase 1) e modelos otimizados
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """
    Configura o logging do projeto com formatação estruturada.

    Emite para stdout por padrão; se log_file for fornecido,
    emite também para arquivo (útil para rastrear experimentos longos).

    Parameters
    ----------
    level : int
        Nível de logging (logging.INFO, logging.DEBUG, etc.).
    log_file : Path, optional
        Caminho para arquivo de log adicional.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def plot_evolution(
    history: list,
    model_name: str,
    experiment_label: str,
    output_path: Path | None = None,
) -> None:
    """
    Gera o gráfico de evolução do fitness ao longo das gerações.

    Plota três linhas: melhor fitness, fitness médio e pior fitness,
    permitindo visualizar convergência e diversidade da população.

    Parameters
    ----------
    history : list[GenerationStats]
        Histórico de gerações retornado pelo run_genetic_algorithm.
    model_name : str
        Nome do modelo (usado no título).
    experiment_label : str
        Rótulo do experimento (ex: 'Experimento 1 — População pequena').
    output_path : Path, optional
        Se fornecido, salva o gráfico nesse caminho (PNG).
    """
    generations = [s.generation for s in history]
    best = [s.best_fitness for s in history]
    mean = [s.mean_fitness for s in history]
    worst = [s.worst_fitness for s in history]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(generations, best,  label="Melhor fitness",  color="#2E75B6", linewidth=2)
    ax.plot(generations, mean,  label="Fitness médio",   color="#ED7D31", linewidth=2,
            linestyle="--")
    ax.plot(generations, worst, label="Pior fitness",    color="#A9A9A9", linewidth=1,
            linestyle=":")

    ax.fill_between(generations, worst, best, alpha=0.08, color="#2E75B6",
                    label="Amplitude da população")

    ax.set_title(
        f"Evolução do Fitness — {model_name.replace('_', ' ').title()}\n"
        f"{experiment_label}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Geração", fontsize=11)
    ax.set_ylabel("Recall (validação cruzada 5-fold)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(left=1)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_comparison(
    results: list,
    output_path: Path | None = None,
) -> None:
    """
    Gera gráfico de barras comparando baseline vs. melhor resultado do AG
    para cada modelo e experimento.

    Parameters
    ----------
    results : list[ExperimentResult]
        Lista de resultados dos experimentos.
    output_path : Path, optional
        Se fornecido, salva o gráfico nesse caminho (PNG).
    """
    records = []
    for r in results:
        records.append({
            "Modelo": r.model_name.replace("_", " ").title(),
            "Tipo": "Baseline (Fase 1)",
            "Recall": r.baseline_fitness,
        })
        if r.best_individual:
            records.append({
                "Modelo": r.model_name.replace("_", " ").title(),
                "Tipo": f"AG otimizado\n(pop={r.config['population_size']}, "
                        f"mut={r.config['mutation_rate']})",
                "Recall": r.best_individual.fitness,
            })

    df = pd.DataFrame(records)

    modelos = df["Modelo"].unique()
    tipos = df["Tipo"].unique()
    x = range(len(modelos))
    width = 0.8 / len(tipos)
    colors = ["#2E75B6", "#ED7D31", "#70AD47", "#FFC000"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tipo in enumerate(tipos):
        subset = df[df["Tipo"] == tipo]
        vals = [
            subset[subset["Modelo"] == m]["Recall"].values[0]
            if m in subset["Modelo"].values else 0
            for m in modelos
        ]
        offset = (i - len(tipos) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x], vals,
            width=width * 0.9,
            label=tipo,
            color=colors[i % len(colors)],
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_title("Comparativo: Baseline (Fase 1) vs. AG Otimizado",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Recall (validação cruzada 5-fold)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(modelos, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(bottom=max(0, df["Recall"].min() - 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def build_summary_table(results: list) -> pd.DataFrame:
    """
    Constrói uma tabela resumo com baseline, melhor resultado do AG,
    ganho absoluto e tempo de execução para cada experimento.

    Parameters
    ----------
    results : list[ExperimentResult]
        Lista de resultados dos experimentos.

    Returns
    -------
    pd.DataFrame formatado para exibição em notebook.
    """
    rows = []
    for r in results:
        best = r.best_individual.fitness if r.best_individual else None
        rows.append({
            "Modelo": r.model_name.replace("_", " ").title(),
            "Pop.": r.config["population_size"],
            "Gerações": r.config["n_generations"],
            "Mutação": r.config["mutation_rate"],
            "Baseline (Recall)": f"{r.baseline_fitness:.4f}",
            "AG melhor (Recall)": f"{best:.4f}" if best else "-",
            "Ganho": f"{best - r.baseline_fitness:+.4f}" if best else "-",
            "Tempo (s)": r.total_seconds,
        })
    return pd.DataFrame(rows)
