"""Avaliação independente em sementes novas, 47-51.

As configurações retestadas em `controle_otimizacao.py` foram escolhidas
depois de ver os resultados de teste da v1.0.0, e a combinação (eta, épocas)
foi escolhida com as sementes 42-46. Para uma estimativa que não dependa de
nenhuma dessas escolhas, cada configuração é treinada de novo com sementes
nunca usadas antes (47-51): a semente fixa dados, partição, inicialização e
ordem dos lotes, então são cinco conjuntos de dados novos. Roda-se a
combinação escolhida em `controle_otimizacao.py` e o protocolo (0,1; 30); quando coincidem, um
treino só. O teste de cada treino é avaliado uma vez e não alimenta nenhuma
escolha.

Entrada: results/revisao/metrics/resumo_controle_otimizacao.csv
Saídas (results/revisao/):
    metrics/sementes_novas.csv                uma linha por treino
    metrics/resumo_sementes_novas.csv         média e desvio por configuração
    metrics/diferenca_pareada_sementes_novas.csv
                                              re-uploading − angle no moons, por semente
    tables/tab_sementes_novas.tex

Uso: python experiments/revisao/sementes_novas.py [--rapido]
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
    SEMENTES_NOVAS,
    TABLES,
    dec,
    gravar_csv,
    md,
    registrar,
    rodar_paralelo,
)

from tccqml.config import PADRAO

PROTOCOLO = (PADRAO.eta, PADRAO.epocas)
CHAVES = ["dataset", "encoding", "k"]
ROT = {
    "angle": "\\textit{Angle}",
    "reuploading": "\\textit{Re-upl.}",
    "moons": "\\textit{Moons}",
    "circles": "\\textit{Circles}",
}


def montar(escolhas: pd.DataFrame, sementes) -> list[dict]:
    cfgs = []
    for e in escolhas.itertuples(index=False):
        combos = {PROTOCOLO, (float(e.eta_escolhido), int(e.epocas_escolhido))}
        for eta, ep in sorted(combos):
            for s in sementes:
                cfgs.append({"encoding": e.encoding, "k": int(e.k), "dataset": e.dataset,
                             "seed": s, "eta": eta, "epocas": ep})
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    escolhas = pd.read_csv(METRICS / "resumo_controle_otimizacao.csv")[
        [*CHAVES, "eta_escolhido", "epocas_escolhido"]
    ]
    if args.rapido:
        cfgs = [dict(c, epocas=3) for c in montar(escolhas, SEMENTES_NOVAS[:1])][:6]
        rodar_paralelo(cfgs, rotulo="novas-rapido")
        print("modo rápido: nada gravado")
        return

    saidas = rodar_paralelo(montar(escolhas, SEMENTES_NOVAS), rotulo="novas")
    df = pd.DataFrame([s["resumo"] for s in saidas]).merge(escolhas, on=CHAVES)
    df["eh_protocolo"] = np.isclose(df["eta"], PROTOCOLO[0]) & (df["epocas"] == PROTOCOLO[1])
    df["eh_escolhida"] = np.isclose(df["eta"], df["eta_escolhido"]) & (
        df["epocas"] == df["epocas_escolhido"]
    )
    df = df.sort_values([*CHAVES, "eta", "epocas", "seed"])

    linhas = []
    for chave, g in df.groupby(CHAVES, sort=False):
        linha = dict(zip(CHAVES, chave))
        linha["eta_escolhido"] = g["eta_escolhido"].iloc[0]
        linha["epocas_escolhido"] = int(g["epocas_escolhido"].iloc[0])
        for rot, col in (("protocolo", "eh_protocolo"), ("escolhido", "eh_escolhida")):
            x = g[g[col]]
            for m in ("acc_treino", "acc_val", "acc_teste"):
                linha[f"{m}_mean_{rot}"] = x[m].mean()
                linha[f"{m}_std_{rot}"] = x[m].std()
            linha[f"n_sementes_{rot}"] = x["seed"].nunique()
        linhas.append(linha)
    resumo = pd.DataFrame(linhas)

    # Diferença pareada re-uploading − angle no moons, mesmo k e mesma semente.
    pares = []
    for rot, col in (("protocolo", "eh_protocolo"), ("escolhido", "eh_escolhida")):
        m = df[(df["dataset"] == "moons") & df[col]]
        r = m[m["encoding"] == "reuploading"][["k", "seed", "acc_teste"]]
        a = m[m["encoding"] == "angle"][["k", "seed", "acc_teste"]]
        p = r.merge(a, on=["k", "seed"], suffixes=("_reup", "_angle"))
        pares.append(p.assign(hiperparametros=rot, dif_acc_teste=p["acc_teste_reup"] - p["acc_teste_angle"]))
    pares = pd.concat(pares)
    resumo_pares = (
        pares.groupby(["hiperparametros", "k"])["dif_acc_teste"]
        .agg(dif_mean="mean", dif_std="std",
             reup_favorece=lambda x: int((x > 1e-12).sum()),
             angle_favorece=lambda x: int((x < -1e-12).sum()),
             empates=lambda x: int((x.abs() <= 1e-12).sum()))
        .reset_index()
    )

    corpo, anterior = [], None
    for r in resumo.itertuples():
        if anterior is not None and r.dataset != anterior:
            corpo.append("\\midrule")
        anterior = r.dataset
        corpo.append(
            f"{ROT[r.dataset]} & {ROT[r.encoding]} & {r.k} & "
            f"{md(r.acc_teste_mean_protocolo, r.acc_teste_std_protocolo)} & "
            f"{dec(r.eta_escolhido, 2)}; {r.epocas_escolhido} & "
            f"{md(r.acc_teste_mean_escolhido, r.acc_teste_std_escolhido)} \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{table}[htb]",
            "\\centering",
            "\\small",
            ("\\caption[Avaliação em sementes novas]{Acurácia de teste em sementes nunca usadas "
            "antes (47 a 51), com o protocolo e com a combinação de $\\eta$ e épocas escolhida "
            "na Tabela~\\ref{tab:controle-otimizacao} (média $\\pm$ desvio padrão sobre cinco "
            "sementes).}"),
            "\\label{tab:sementes-novas}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "Conjunto & Codificação & $k$ & Teste, protocolo & $\\eta$; épocas & Teste, escolhida \\\\",
            "\\midrule",
            *corpo,
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[2pt]",
            ("{\\footnotesize Cada semente gera um conjunto de dados, uma partição e uma "
            "inicialização novos; nenhuma destas sementes participou das escolhas.}"),
            "\\\\[2pt]",
            "{\\footnotesize Fonte: Elaborada pelo autor.}",
            "\\end{table}",
            "",
        ]
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    caminho_tex = TABLES / "tab_sementes_novas.tex"
    caminho_tex.write_text(tex, encoding="utf-8")
    arquivos = [
        gravar_csv(df, METRICS / "sementes_novas.csv"),
        gravar_csv(resumo, METRICS / "resumo_sementes_novas.csv"),
        gravar_csv(pares, METRICS / "diferenca_pareada_sementes_novas.csv"),
        gravar_csv(resumo_pares, METRICS / "resumo_diferenca_pareada_sementes_novas.csv"),
        caminho_tex,
    ]
    print(resumo.round(3).to_string(index=False))
    print(resumo_pares.round(4).to_string(index=False))
    registrar(__file__, arquivos, {"sementes": list(SEMENTES_NOVAS), "treinos": len(saidas)})


if __name__ == "__main__":
    main()
