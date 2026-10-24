"""Curvas de treinamento por semente, oscilações e tabela treino/teste.

Entradas:
    results/metrics/comparacao.csv                 histórico da grade principal (v1.0.0)
    results/metrics/varredura_R.csv, varredura_L_var.csv   métricas finais (v1.0.0)
    results/revisao/metrics/historico_varredura.csv  histórico de k = 1..5
                                                   (`historicos_varredura.py`)

Saídas (results/revisao/):
    figures/curvas-treinamento-sementes.pdf   grade principal: média + uma linha por semente
    figures/curvas-profundidade.pdf           angle (L_var = k) e re-uploading (R = k), k = 4 e 5
    metrics/oscilacoes.csv                    por treino: amplitude das acurácias nas
                                              10 últimas épocas, queda do máximo ao fim,
                                              diferença treino − teste, época do limiar relativo
    metrics/treino_teste_profundidade.csv     treino e teste (média ± desvio) por k
    tables/tab_treino_teste_profundidade.tex

Definições (fixadas aqui e repetidas no texto):
    oscilação: amplitude (máximo − mínimo) de cada acurácia nas épocas 21 a 30.
    Dois indicadores descritivos: `oscila` (validação, amplitude >= 0,05, isto é,
    3 das 60 amostras) e `oscila_treino` (treino, amplitude >= 0,10, isto é,
    18 das 180 amostras). O segundo é o citado no texto: a validação tem só 60
    amostras, e uma amplitude de 3 amostras é comum mesmo em treinos estáveis.
    tempo até o limiar relativo: primeira época em que a acurácia de treino atinge
    90% do seu valor final (`experiments.epoca_para_90pct`, coluna `epoca_90pct`).

Uso: python experiments/revisao/curvas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _comum import (
    FIGURES,
    METRICS,
    RAIZ,
    TABLES,
    gravar_csv,
    md,
    registrar,
    salvar_figura,
)

from tccqml.experiments import epoca_para_90pct
from tccqml.figuras import NOMES, ORDEM_DATASETS, ORDEM_ENCODINGS, RC, _estilo

JANELA = 10
LIMIAR_OSCILACAO = 0.05
LIMIAR_OSCILACAO_TREINO = 0.10
ROT = {"xor": "\\textit{Xor}", "moons": "\\textit{Moons}", "circles": "\\textit{Circles}"}


def _painel_series(eixos, hist, grupos, series, datasets):
    """Uma coluna por dataset; para cada grupo, média grossa e sementes finas."""
    for col, dataset in enumerate(datasets):
        for rotulo, filtro, estilo in grupos:
            fatia = hist[(hist["dataset"] == dataset) & filtro(hist)]
            if fatia.empty:
                continue
            media = fatia.groupby("epoca")[list(series)].mean()
            for linha, s in enumerate(series):
                eixo = eixos[linha][col]
                for _, g in fatia.groupby("seed"):
                    eixo.plot(g["epoca"], g[s], color=estilo["color"], linewidth=0.5,
                              alpha=0.3, linestyle="-")
                eixo.plot(media.index, media[s], label=rotulo, linewidth=1.4,
                          markevery=6, markersize=4, **estilo)
        eixos[-1][col].set_xlabel(f"{NOMES[dataset]}\népoca")


def fig_grade(comp: pd.DataFrame) -> plt.Figure:
    series = ("custo", "acc_treino", "acc_val")
    datasets = [d for d in ORDEM_DATASETS if d in set(comp["dataset"])]
    fig, eixos = plt.subplots(3, len(datasets), figsize=(6.5, 6.2), sharex=True, sharey="row",
                              squeeze=False)
    grupos = [(NOMES[e], (lambda h, e=e: h["encoding"] == e), _estilo(e)) for e in ORDEM_ENCODINGS]
    _painel_series(eixos, comp, grupos, series, datasets)
    for i, r in enumerate(("custo de treino", "acurácia de treino", "acurácia de validação")):
        eixos[i][0].set_ylabel(r)
    alcas, rot = eixos[0][0].get_legend_handles_labels()
    fig.legend(alcas, rot, loc="outside upper center", ncol=len(rot), frameon=False)
    return fig


def fig_profundidade(hist: pd.DataFrame) -> plt.Figure:
    series = ("acc_treino", "acc_val")
    datasets = [d for d in ORDEM_DATASETS if d in set(hist["dataset"])]
    estilos = {
        ("angle", 4): {"color": "#7f7f7f", "marker": "o", "linestyle": "--"},
        ("angle", 5): {"color": "#1b1b1b", "marker": "o", "linestyle": "-"},
        ("reuploading", 4): {"color": "#e69a9c", "marker": "^", "linestyle": "--"},
        ("reuploading", 5): {"color": "#c44e52", "marker": "^", "linestyle": "-"},
    }
    simb = {"angle": "L_\\mathrm{var}", "reuploading": "R"}
    grupos = [
        (f"{NOMES[e]}, ${simb[e]} = {k}$",
         (lambda h, e=e, k=k: (h["encoding"] == e) & (h["k"] == k)), est)
        for (e, k), est in estilos.items()
    ]
    fig, eixos = plt.subplots(2, len(datasets), figsize=(6.5, 4.4), sharex=True, sharey="row",
                              squeeze=False)
    _painel_series(eixos, hist, grupos, series, datasets)
    eixos[0][0].set_ylabel("acurácia de treino")
    eixos[1][0].set_ylabel("acurácia de validação")
    alcas, rot = eixos[0][0].get_legend_handles_labels()
    fig.legend(alcas, rot, loc="outside upper center", ncol=2, frameon=False)
    return fig


def oscilacoes(hist: pd.DataFrame, finais: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    linhas = []
    for chave, g in hist.groupby(chaves):
        g = g.sort_values("epoca")
        fim = g[g["epoca"] > g["epoca"].max() - JANELA]
        linhas.append({
            **dict(zip(chaves, chave)),
            "amp_val_ult10": fim["acc_val"].max() - fim["acc_val"].min(),
            "amp_treino_ult10": fim["acc_treino"].max() - fim["acc_treino"].min(),
            "queda_val_max_ao_fim": g["acc_val"].max() - g["acc_val"].iloc[-1],
            "queda_treino_max_ao_fim": g["acc_treino"].max() - g["acc_treino"].iloc[-1],
            "acc_treino_ultima": g["acc_treino"].iloc[-1],
            "acc_val_ultima": g["acc_val"].iloc[-1],
            "epoca_limiar_relativo_90": epoca_para_90pct(g.reset_index(drop=True)),
        })
    df = pd.DataFrame(linhas).merge(finais, on=chaves)
    df["gap_treino_teste"] = df["acc_treino"] - df["acc_teste"]
    df["oscila"] = df["amp_val_ult10"] >= LIMIAR_OSCILACAO - 1e-12
    df["oscila_treino"] = df["amp_treino_ult10"] >= LIMIAR_OSCILACAO_TREINO - 1e-12
    return df


def tabela_profundidade(v: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    agg = (
        v.groupby(["dataset", "encoding", "k"])[["acc_treino", "acc_teste"]]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    corpo = []
    for i, dataset in enumerate([d for d in ORDEM_DATASETS if d in set(agg["dataset"])]):
        if i:
            corpo.append("\\midrule")
        for k in sorted(agg["k"].unique()):
            a = agg[(agg["dataset"] == dataset) & (agg["encoding"] == "angle") & (agg["k"] == k)].iloc[0]
            r = agg[(agg["dataset"] == dataset) & (agg["encoding"] == "reuploading") & (agg["k"] == k)].iloc[0]
            nome = ROT[dataset] if k == 1 else ""
            corpo.append(
                f"{nome} & {k} & {md(a.acc_treino_mean, a.acc_treino_std)} & "
                f"{md(a.acc_teste_mean, a.acc_teste_std)} & {md(r.acc_treino_mean, r.acc_treino_std)} & "
                f"{md(r.acc_teste_mean, r.acc_teste_std)} \\\\"
            )
    tex = "\n".join([
        "\\begin{table}[htb]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        ("\\caption[Acurácias de treino e de teste por profundidade]{Acurácias de treino e de "
        "teste do \\textit{angle} com $L_\\text{var} = k$ e do \\textit{re-uploading} com $R = k$ "
        "(mesmo $p = 6k$), na época escolhida pela validação (média $\\pm$ desvio padrão sobre "
        "cinco sementes).}"),
        "\\label{tab:treino-teste-profundidade}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        (" & & \\multicolumn{2}{c}{\\textit{Angle} ($L_\\text{var} = k$)} & "
        "\\multicolumn{2}{c}{\\textit{Re-uploading} ($R = k$)} \\\\"),
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "Conjunto & $k$ & Treino & Teste & Treino & Teste \\\\",
        "\\midrule",
        *corpo,
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[2pt]",
        ("{\\footnotesize Com $k = 1$, os dois modelos são o mesmo circuito. As colunas de teste "
        "repetem a Tabela~\\ref{tab:mesmo-p}.}"),
        "\\\\[2pt]",
        "{\\footnotesize Fonte: Elaborada pelo autor.}",
        "\\end{table}",
        "",
    ])
    return agg, tex


def main() -> None:
    comp = pd.read_csv(RAIZ / "results/metrics/comparacao.csv")
    por_treino = pd.read_csv(RAIZ / "results/metrics/por_treino.csv")
    hist_v = pd.read_csv(METRICS / "historico_varredura.csv")
    v = pd.concat([
        pd.read_csv(RAIZ / "results/metrics/varredura_R.csv").assign(k=lambda d: d["R"]),
        pd.read_csv(RAIZ / "results/metrics/varredura_L_var.csv").assign(k=lambda d: d["L_var"]),
    ])
    v["k"] = v["k"].astype(int)

    finais_cols = ["acc_treino", "acc_val", "acc_teste", "epoca_escolhida"]
    osc_grade = oscilacoes(comp, por_treino[["encoding", "dataset", "seed", *finais_cols]],
                           ["encoding", "dataset", "seed"]).assign(experimento="grade")
    osc_var = oscilacoes(hist_v, v[["encoding", "dataset", "k", "seed", *finais_cols]],
                         ["encoding", "dataset", "k", "seed"]).assign(experimento="varredura")
    osc = pd.concat([osc_grade, osc_var], ignore_index=True)

    agg, tex = tabela_profundidade(v)
    TABLES.mkdir(parents=True, exist_ok=True)
    caminho_tex = TABLES / "tab_treino_teste_profundidade.tex"
    caminho_tex.write_text(tex, encoding="utf-8")

    arquivos = [
        gravar_csv(osc, METRICS / "oscilacoes.csv"),
        gravar_csv(agg, METRICS / "treino_teste_profundidade.csv"),
        caminho_tex,
    ]
    with plt.rc_context(RC):
        arquivos.append(salvar_figura(fig_grade(comp), FIGURES / "curvas-treinamento-sementes.pdf"))
        arquivos.append(salvar_figura(fig_profundidade(hist_v), FIGURES / "curvas-profundidade.pdf"))

    print("execuções com oscilação (validação >= 0,05):", int(osc["oscila"].sum()), "de", len(osc))
    print("execuções que oscilam no treino (amplitude de acc_treino nas 10 últimas épocas >= 0,10):")
    cols = ["experimento", "encoding", "dataset", "k", "seed", "amp_val_ult10", "amp_treino_ult10",
            "queda_val_max_ao_fim", "gap_treino_teste", "epoca_escolhida"]
    print(osc[osc["oscila_treino"]][cols].round(3).to_string(index=False))
    print("\ngap treino - teste por grupo (média, máx):")
    print(osc.groupby(["experimento", "encoding", "dataset"])["gap_treino_teste"].agg(["mean", "max"]).round(3))
    registrar(__file__, arquivos, {"janela": JANELA, "limiar_oscilacao": LIMIAR_OSCILACAO,
                                         "limiar_oscilacao_treino": LIMIAR_OSCILACAO_TREINO})


if __name__ == "__main__":
    main()
