"""Diferença pareada re-uploading − angle com o mesmo p.

Usa só os CSVs da versão 1.0.0 (`varredura_R.csv` e `varredura_L_var.csv`):
nenhum treino novo. Para cada conjunto, k = R = L_var e semente, calcula
Δ = acc_teste(re-uploading, R = k) − acc_teste(angle, L_var = k), pareando pelo
dataset e pela semente (mesmos pontos, mesma partição, mesma inicialização).

Saídas (results/revisao/):
    metrics/diferenca_pareada.csv          uma linha por (dataset, k, semente)
    metrics/resumo_diferenca_pareada.csv   média, desvio e contagem de sementes
                                           a favor de cada método, por (dataset, k)
    figures/diferenca-pareada.pdf

Uso: python experiments/revisao/diferenca_pareada.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _comum import FIGURES, METRICS, RAIZ, gravar_csv, registrar, salvar_figura

from tccqml.figuras import NOMES, ORDEM_DATASETS, RC

METRICAS = ("acc_teste", "acc_val", "acc_treino")


def diferencas() -> pd.DataFrame:
    r = pd.read_csv(RAIZ / "results/metrics/varredura_R.csv")
    a = pd.read_csv(RAIZ / "results/metrics/varredura_L_var.csv")
    r = r.assign(k=r["R"].astype(int))[["dataset", "k", "seed", *METRICAS]]
    a = a.assign(k=a["L_var"].astype(int))[["dataset", "k", "seed", *METRICAS]]
    d = r.merge(a, on=["dataset", "k", "seed"], suffixes=("_reup", "_angle"), validate="1:1")
    for m in METRICAS:
        d[f"dif_{m}"] = d[f"{m}_reup"] - d[f"{m}_angle"]
    return d.sort_values(["dataset", "k", "seed"])


def resumir(d: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for (dataset, k), g in d.groupby(["dataset", "k"]):
        linha = {"dataset": dataset, "k": k, "n_sementes": len(g)}
        for m in METRICAS:
            x = g[f"dif_{m}"]
            linha |= {
                f"dif_{m}_mean": x.mean(),
                f"dif_{m}_std": x.std(),
                f"dif_{m}_pp": 100 * x.mean(),
                f"{m}_reup_favorece": int((x > 1e-12).sum()),
                f"{m}_angle_favorece": int((x < -1e-12).sum()),
                f"{m}_empates": int((x.abs() <= 1e-12).sum()),
            }
        linhas.append(linha)
    return pd.DataFrame(linhas)


def figura(d: pd.DataFrame, resumo: pd.DataFrame) -> plt.Figure:
    datasets = [x for x in ORDEM_DATASETS if x in set(d["dataset"])]
    fig, eixos = plt.subplots(1, len(datasets), figsize=(6.5, 2.6), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)  # só o espalhamento horizontal dos pontos
    for eixo, dataset in zip(eixos[0], datasets):
        g = d[d["dataset"] == dataset]
        jitter = rng.uniform(-0.12, 0.12, len(g))
        eixo.scatter(
            g["k"] + jitter, 100 * g["dif_acc_teste"], s=14, color="#7f7f7f",
            alpha=0.8, linewidths=0, label="semente",
        )
        rr = resumo[resumo["dataset"] == dataset]
        eixo.plot(
            rr["k"], rr["dif_acc_teste_pp"], color="#c44e52", marker="D", markersize=5,
            linewidth=1.3, label="média",
        )
        eixo.axhline(0.0, color="#1b1b1b", linewidth=0.9)
        eixo.set_xticks(sorted(d["k"].unique()))
        eixo.set_xlabel(f"{NOMES[dataset]}\n$k = R = L_\\mathrm{{var}}$")
    eixos[0][0].set_ylabel("re-uploading − angle\n(pontos percentuais)")
    alcas, rotulos = eixos[0][0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="outside upper center", ncol=2, frameon=False)
    return fig


def main() -> None:
    d = diferencas()
    resumo = resumir(d)
    arquivos = [
        gravar_csv(d, METRICS / "diferenca_pareada.csv"),
        gravar_csv(resumo, METRICS / "resumo_diferenca_pareada.csv"),
    ]
    with plt.rc_context(RC):
        arquivos.append(salvar_figura(figura(d, resumo), FIGURES / "diferenca-pareada.pdf"))
    colunas = [
        "dataset", "k", "dif_acc_teste_pp", "dif_acc_teste_std",
        "acc_teste_reup_favorece", "acc_teste_angle_favorece", "acc_teste_empates",
        "dif_acc_treino_pp", "dif_acc_val_pp",
    ]
    print(resumo[colunas].round(4).to_string(index=False))
    registrar(__file__, arquivos)


if __name__ == "__main__":
    main()
