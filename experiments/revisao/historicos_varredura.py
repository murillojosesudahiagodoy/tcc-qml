"""Varreduras de R e de L_var (k = 1..5) de novo, com histórico e pesos.

Os CSVs da versão 1.0.0 (`varredura_R.csv`,
`varredura_L_var.csv`) só guardam a época escolhida; aqui o MESMO treino
(`experiments.rodar_um`, protocolo `PADRAO`, sementes 42-46) é repetido para
gravar o histórico por época e os parâmetros da época escolhida. Como o treino
é determinístico, as métricas finais têm de coincidir com as da v1.0.0: o
script confere isso e grava o resultado da conferência.

Saídas (results/revisao/):
    metrics/historico_varredura.csv   custo e acurácias por época
    metrics/por_treino_varredura.csv  uma linha por treino
    metrics/conferencia_varredura.csv diferença para os CSVs da v1.0.0
    weights/varredura/<enc>_k<k>_<dataset>_<seed>.npz

Uso: python experiments/revisao/historicos_varredura.py [--rapido]
(`--rapido`: 1 semente, 3 épocas, só para validar o script.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _comum import (
    METRICS,
    RAIZ,
    WEIGHTS,
    gravar_csv,
    registrar,
    rodar_paralelo,
    salvar_pesos,
)

from tccqml.config import PADRAO

KS = PADRAO.R_varredura
DATASETS = PADRAO.datasets_grade
SEMENTES = PADRAO.sementes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    sementes = SEMENTES[:1] if args.rapido else SEMENTES
    epocas = 3 if args.rapido else PADRAO.epocas
    cfgs = [
        {"encoding": enc, "k": k, "dataset": ds, "seed": s, "eta": PADRAO.eta, "epocas": epocas}
        for enc in ("angle", "reuploading")
        for k in KS
        for ds in DATASETS
        for s in sementes
    ]
    saidas = rodar_paralelo(cfgs, rotulo="varredura")
    if args.rapido:
        print("modo rápido: nada gravado")
        return

    pasta_pesos = WEIGHTS / "varredura"
    arquivos = []
    for s in saidas:
        c = s["cfg"]
        arquivos.append(
            salvar_pesos(s, pasta_pesos, f"{c['encoding']}_k{c['k']}_{c['dataset']}_{c['seed']}")
        )
    hist = pd.concat([s["hist"] for s in saidas], ignore_index=True)
    hist = hist.sort_values(["encoding", "k", "dataset", "seed", "epoca"])
    por_treino = pd.DataFrame([s["resumo"] for s in saidas]).sort_values(
        ["encoding", "k", "dataset", "seed"]
    )
    arquivos.append(gravar_csv(hist, METRICS / "historico_varredura.csv"))
    arquivos.append(gravar_csv(por_treino, METRICS / "por_treino_varredura.csv"))

    # Conferência contra a v1.0.0: mesmas métricas finais, treino a treino.
    antigos = pd.concat(
        [
            pd.read_csv(RAIZ / "results/metrics/varredura_R.csv").assign(k=lambda d: d["R"]),
            pd.read_csv(RAIZ / "results/metrics/varredura_L_var.csv").assign(
                k=lambda d: d["L_var"]
            ),
        ]
    )
    antigos["k"] = antigos["k"].astype(int)
    chaves = ["encoding", "k", "dataset", "seed"]
    colunas = ["acc_treino", "acc_val", "acc_teste", "epoca_escolhida"]
    junto = por_treino[chaves + colunas].merge(
        antigos[chaves + colunas], on=chaves, suffixes=("", "_v100")
    )
    for c in colunas:
        junto[f"dif_{c}"] = junto[c] - junto[f"{c}_v100"]
    arquivos.append(gravar_csv(junto, METRICS / "conferencia_varredura.csv"))
    max_dif = float(np.abs(junto[[f"dif_{c}" for c in colunas]].to_numpy()).max())
    print(f"treinos conferidos: {len(junto)}; maior diferença absoluta para a v1.0.0: {max_dif:.3g}")

    registrar(__file__, arquivos, {"treinos": len(saidas), "max_dif_v100": max_dif})


if __name__ == "__main__":
    main()
