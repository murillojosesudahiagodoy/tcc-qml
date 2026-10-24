"""Controle linear clássico com C escolhido pela validação.

Estende `tccqml.verificacoes.rodar_controle_linear`, que usa C = 1 fixo. As
nove características são as mesmas (`caracteristicas_fourier`, Eq. base-9:
produtos de {1, cos x_i, sin x_i}), calculadas nas coordenadas normalizadas
para [0, pi], com as MESMAS partições e sementes da grade principal
(`verificacoes._carregar`, que chama `load_dataset` com `PADRAO`).

Regra de escolha (fixada antes de rodar, a mesma para os três conjuntos):
para cada conjunto, ajusta-se a regressão logística no treino com cada
C em {0,01; 0,1; 1; 10; 100} e cada semente; escolhe-se o C de maior acurácia
MÉDIA de validação nas cinco sementes. Empate (diferença até 1e-9): fica C = 1,
o valor do controle original, se estiver entre os empatados; senão, o menor C
(mais regularizado). O teste só é avaliado depois da escolha, com o C
escolhido e com C = 1 (o controle original); nos demais C a coluna
`acc_teste` fica vazia.

Saídas (results/revisao/):
    metrics/controle_linear_C.csv          uma linha por (dataset, C, semente)
    metrics/resumo_controle_linear_C.csv   C escolhido e C = 1 lado a lado
    tables/tab_controle_linear_C.tex

Uso: python experiments/revisao/controle_linear.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _comum import METRICS, TABLES, gravar_csv, md, registrar
from sklearn.linear_model import LogisticRegression

from tccqml.config import PADRAO
from tccqml.verificacoes import _carregar, caracteristicas_fourier

CS = (0.01, 0.1, 1.0, 10.0, 100.0)
C_ORIGINAL = PADRAO.controle_linear_C
TOL = 1e-9
ROT = {"xor": "\\textit{Xor}", "moons": "\\textit{Moons}", "circles": "\\textit{Circles}"}


def rodar() -> tuple[pd.DataFrame, dict]:
    """Ajusta todos os C no treino; devolve as métricas e os modelos para o teste."""
    linhas, modelos = [], {}
    for dataset in PADRAO.datasets_grade:
        for seed in PADRAO.sementes:
            ds = _carregar(dataset, seed, PADRAO)
            F_tr, F_va, F_te = (caracteristicas_fourier(X) for X in (ds.X_train, ds.X_val, ds.X_test))
            for C in CS:
                clf = LogisticRegression(C=C, max_iter=PADRAO.controle_linear_max_iter)
                clf.fit(F_tr, ds.y_train)
                modelos[(dataset, seed, C)] = (clf, F_te, ds.y_test)
                linhas.append(
                    {
                        "dataset": dataset,
                        "C": C,
                        "seed": seed,
                        "acc_treino": clf.score(F_tr, ds.y_train),
                        "acc_val": clf.score(F_va, ds.y_val),
                        "acc_teste": np.nan,
                        "n_iter": int(np.max(clf.n_iter_)),
                    }
                )
    return pd.DataFrame(linhas), modelos


def avaliar_teste(df: pd.DataFrame, escolhas: dict, modelos: dict) -> pd.DataFrame:
    df = df.copy()
    for i, r in df.iterrows():
        if r["C"] in (escolhas[r["dataset"]], C_ORIGINAL):
            clf, F_te, y_te = modelos[(r["dataset"], r["seed"], r["C"])]
            df.at[i, "acc_teste"] = clf.score(F_te, y_te)
    return df


def escolher(df: pd.DataFrame) -> dict:
    escolhas = {}
    for dataset, g in df.groupby("dataset"):
        medias = g.groupby("C")["acc_val"].mean()
        empatados = sorted(c for c, v in medias.items() if abs(v - medias.max()) <= TOL)
        escolhas[dataset] = C_ORIGINAL if C_ORIGINAL in empatados else empatados[0]
    return escolhas


def resumir(df: pd.DataFrame, escolhas: dict) -> pd.DataFrame:
    linhas = []
    for dataset in PADRAO.datasets_grade:
        g = df[df["dataset"] == dataset]
        linha = {"dataset": dataset, "C_escolhido": escolhas[dataset]}
        for rotulo, C in (("escolhido", escolhas[dataset]), ("C1", C_ORIGINAL)):
            x = g[np.isclose(g["C"], C)]
            for m in ("acc_treino", "acc_val", "acc_teste"):
                linha[f"{m}_mean_{rotulo}"] = x[m].mean()
                linha[f"{m}_std_{rotulo}"] = x[m].std()
        medias_val = g.groupby("C")["acc_val"].mean()
        for C, v in medias_val.items():
            linha[f"acc_val_mean_C{C:g}"] = v
        linha["n_sementes"] = g["seed"].nunique()
        linha["max_iter_atingido"] = bool((g["n_iter"] >= PADRAO.controle_linear_max_iter).any())
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela(resumo: pd.DataFrame) -> str:
    corpo = []
    for r in resumo.itertuples():
        C = f"{r.C_escolhido:g}".replace(".", ",")
        corpo.append(
            f"{ROT[r.dataset]} & {C} & {md(r.acc_treino_mean_escolhido, r.acc_treino_std_escolhido)}"
            f" & {md(r.acc_val_mean_escolhido, r.acc_val_std_escolhido)}"
            f" & {md(r.acc_teste_mean_escolhido, r.acc_teste_std_escolhido)}"
            f" & {md(r.acc_teste_mean_C1, r.acc_teste_std_C1)} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[htb]",
            "\\centering",
            "\\small",
            ("\\caption[Controle linear com regularização escolhida pela validação]{Regressão "
            "logística sobre as nove funções da Eq.~\\eqref{eq:base-9}, com $C$ escolhido em "
            "$\\{0{,}01;\\ 0{,}1;\\ 1;\\ 10;\\ 100\\}$ pela maior acurácia média de validação "
            "(média $\\pm$ desvio padrão sobre cinco sementes).}"),
            "\\label{tab:controle-linear-C}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            " & & \\multicolumn{3}{c}{$C$ escolhido} & $C = 1$ \\\\",
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-6}",
            "Conjunto & $C$ & Treino & Validação & Teste & Teste \\\\",
            "\\midrule",
            *corpo,
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[2pt]",
            ("{\\footnotesize Mesmas partições e sementes da comparação principal; o teste não "
            "participa da escolha de $C$. Empate na validação: fica $C = 1$, se empatado; senão, "
            "o menor $C$. A coluna $C = 1$ repete o controle da Tabela~\\ref{tab:verificacoes}.}"),
            "\\\\[2pt]",
            "{\\footnotesize Fonte: Elaborada pelo autor.}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    df, modelos = rodar()
    escolhas = escolher(df)
    df = avaliar_teste(df, escolhas, modelos)
    resumo = resumir(df, escolhas)
    TABLES.mkdir(parents=True, exist_ok=True)
    tex = TABLES / "tab_controle_linear_C.tex"
    tex.write_text(tabela(resumo), encoding="utf-8")
    arquivos = [
        gravar_csv(df, METRICS / "controle_linear_C.csv"),
        gravar_csv(resumo, METRICS / "resumo_controle_linear_C.csv"),
        tex,
    ]
    print(resumo.round(4).T.to_string())
    registrar(__file__, arquivos, {"C_candidatos": list(CS), "escolhas": escolhas})


if __name__ == "__main__":
    main()
