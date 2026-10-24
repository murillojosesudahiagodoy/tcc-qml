"""Controle de otimização: taxa de aprendizado x orçamento de épocas.

Pergunta: as diferenças entre angle (L_var = k) e re-uploading (R = k) no
moons, e a queda do angle de k = 4 para k = 5 no circles, dependem de
eta = 0,1 e de 30 épocas? Cada configuração é treinada com a mesma grade,

    eta em {0,01; 0,03; 0,1; 0,3}  x  épocas em {30, 100},

e as cinco sementes da grade principal (42-46). O orçamento de seleção é o
mesmo para todos os métodos: 8 combinações x 5 sementes. Tudo o mais é o
protocolo `PADRAO` (`experiments.rodar_um`): mesma partição, inicialização,
lotes, e parada pela época de maior acurácia de validação dentro do orçamento.

Configurações (fixadas antes de rodar):
    moons:   angle k = 1..5 e re-uploading R = 1..5
    circles: angle k = 4 e 5

Regra de escolha, por configuração: a combinação (eta, épocas) de maior
acurácia MÉDIA de validação nas cinco sementes. Empate (até 1e-9): fica o
protocolo (0,1; 30), se empatado; senão, o menor número de épocas e, depois, o
menor eta. O teste não participa; ele é reportado para a combinação escolhida
e para o protocolo. Como a escolha de QUAIS configurações retestar foi
motivada pelos resultados de teste da v1.0.0, a avaliação independente fica em
`sementes_novas.py`.

Reaproveitamento: as combinações idênticas a experimentos da v1.0.0 são
treinadas de novo (o treino é determinístico) e conferidas contra os CSVs
antigos (`conferencia_otimizacao.csv`): (0,1; 30) contra as varreduras e
(eta; 30) do angle k = 2 e do re-uploading R = 3 contra `sensibilidade_eta.csv`.

Saídas (results/revisao/):
    metrics/controle_otimizacao.csv            uma linha por treino
    metrics/historico_controle_otimizacao.csv  custo e acurácias por época
    metrics/resumo_controle_otimizacao.csv     escolha e protocolo lado a lado
    metrics/validacao_controle_otimizacao.csv  acc_val média de cada combinação
    metrics/conferencia_otimizacao.csv
    tables/tab_controle_otimizacao.tex

Uso: python experiments/revisao/controle_otimizacao.py [--rapido]
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
    TABLES,
    dec,
    gravar_csv,
    md,
    registrar,
    rodar_paralelo,
)

from tccqml.config import PADRAO

ETAS = PADRAO.eta_candidatos
EPOCAS = (PADRAO.epocas, 100)
PROTOCOLO = (PADRAO.eta, PADRAO.epocas)
SEMENTES = PADRAO.sementes
TOL = 1e-9
CONFIGS = [("moons", enc, k) for enc in ("angle", "reuploading") for k in (1, 2, 3, 4, 5)] + [
    ("circles", "angle", 4),
    ("circles", "angle", 5),
]
CHAVES = ["dataset", "encoding", "k"]
ROT = {
    "angle": "\\textit{Angle}",
    "reuploading": "\\textit{Re-upl.}",
    "moons": "\\textit{Moons}",
    "circles": "\\textit{Circles}",
}


def cfgs(sementes, etas, epocas) -> list[dict]:
    return [
        {"encoding": enc, "k": k, "dataset": ds, "seed": s, "eta": eta, "epocas": ep}
        for ds, enc, k in CONFIGS
        for eta in etas
        for ep in epocas
        for s in sementes
    ]


def escolher(por_treino: pd.DataFrame) -> pd.DataFrame:
    medias = por_treino.groupby([*CHAVES, "eta", "epocas"], as_index=False)["acc_val"].mean()
    escolhas = []
    for chave, g in medias.groupby(CHAVES):
        melhor = g["acc_val"].max()
        emp = g[np.abs(g["acc_val"] - melhor) <= TOL].sort_values(["epocas", "eta"])
        prot = emp[np.isclose(emp["eta"], PROTOCOLO[0]) & (emp["epocas"] == PROTOCOLO[1])]
        linha = (prot if len(prot) else emp).iloc[0]
        escolhas.append(
            dict(zip(CHAVES, chave), eta_escolhido=linha["eta"], epocas_escolhido=int(linha["epocas"]),
                 n_empatados=len(emp))
        )
    return pd.DataFrame(escolhas)


def resumir(por_treino: pd.DataFrame, escolhas: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for e in escolhas.itertuples(index=False):
        g = por_treino[
            (por_treino["dataset"] == e.dataset)
            & (por_treino["encoding"] == e.encoding)
            & (por_treino["k"] == e.k)
        ]
        linha = e._asdict()
        for rot, (eta, ep) in (("escolhido", (e.eta_escolhido, e.epocas_escolhido)), ("protocolo", PROTOCOLO)):
            x = g[np.isclose(g["eta"], eta) & (g["epocas"] == ep)]
            for m in ("acc_treino", "acc_val", "acc_teste", "epoca_escolhida"):
                linha[f"{m}_mean_{rot}"] = x[m].mean()
                linha[f"{m}_std_{rot}"] = x[m].std()
            linha[f"n_sementes_{rot}"] = x["seed"].nunique()
        linhas.append(linha)
    return pd.DataFrame(linhas)


def conferir(por_treino: pd.DataFrame) -> pd.DataFrame:
    """Combinações idênticas à v1.0.0 têm de dar as mesmas métricas."""
    colunas = ["acc_treino", "acc_val", "acc_teste", "epoca_escolhida"]
    antigos = []
    vr = pd.read_csv(RAIZ / "results/metrics/varredura_R.csv").assign(k=lambda d: d["R"])
    vl = pd.read_csv(RAIZ / "results/metrics/varredura_L_var.csv").assign(k=lambda d: d["L_var"])
    for v in (vr, vl):
        antigos.append(v.assign(eta=PROTOCOLO[0], epocas=PROTOCOLO[1], origem="varredura")[
            ["encoding", "dataset", "k", "seed", "eta", "epocas", "origem", *colunas]])
    se = pd.read_csv(RAIZ / "results/metrics/sensibilidade_eta.csv")
    se = se[se["encoding"].isin(["angle", "reuploading"])].assign(
        k=lambda d: np.where(d["encoding"] == "angle", PADRAO.L_var, PADRAO.R), epocas=PADRAO.epocas,
        origem="sensibilidade_eta",
    )
    antigos.append(se[["encoding", "dataset", "k", "seed", "eta", "epocas", "origem", *colunas]])
    antigos = pd.concat(antigos)
    antigos["k"] = antigos["k"].astype(int)
    junto = por_treino.merge(
        antigos, on=["encoding", "dataset", "k", "seed", "epocas"], suffixes=("", "_v100")
    )
    junto = junto[np.isclose(junto["eta"], junto["eta_v100"])]
    for c in colunas:
        junto[f"dif_{c}"] = junto[c] - junto[f"{c}_v100"]
    return junto[["encoding", "dataset", "k", "seed", "eta", "epocas", "origem",
                  *[f"dif_{c}" for c in colunas]]]


def tabela(resumo: pd.DataFrame) -> str:
    corpo, anterior = [], None
    for r in resumo.itertuples():
        if anterior is not None and r.dataset != anterior:
            corpo.append("\\midrule")
        anterior = r.dataset
        corpo.append(
            f"{ROT[r.dataset]} & {ROT[r.encoding]} & {r.k} & "
            f"{dec(r.eta_escolhido, 2)}; {r.epocas_escolhido} & "
            f"{md(r.acc_val_mean_escolhido, r.acc_val_std_escolhido)} & "
            f"{md(r.acc_teste_mean_protocolo, r.acc_teste_std_protocolo)} & "
            f"{md(r.acc_teste_mean_escolhido, r.acc_teste_std_escolhido)} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[htb]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4pt}",
            ("\\caption[Controle de otimização]{Controle de otimização: $\\eta \\in "
            "\\{0{,}01;\\ 0{,}03;\\ 0{,}1;\\ 0{,}3\\}$ e 30 ou 100 épocas, com a combinação "
            "escolhida pela maior acurácia média de validação (média $\\pm$ desvio padrão sobre "
            "as sementes 42 a 46).}"),
            "\\label{tab:controle-otimizacao}",
            "\\begin{tabular}{llcccccc}",
            "\\toprule",
            ("Conjunto & Codificação & $k$ & $\\eta$; épocas & Validação (escolhida) & "
            "Teste, protocolo & Teste, escolhida \\\\"),
            "\\midrule",
            *corpo,
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[2pt]",
            ("{\\footnotesize $k$: $L_\\text{var}$ no \\textit{angle} e $R$ no \\textit{re-uploading} "
            "(mesmo $p = 6k$). Protocolo: $\\eta = 0{,}1$ e 30 épocas. Mesmo orçamento de seleção "
            "para todas as configurações (8 combinações $\\times$ 5 sementes); o teste não participa "
            "da escolha. Empate na validação: fica o protocolo, se empatado; senão, menos épocas e "
            "menor $\\eta$.}"),
            "\\\\[2pt]",
            "{\\footnotesize Fonte: Elaborada pelo autor.}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    if args.rapido:
        rodar_paralelo(cfgs(SEMENTES[:1], (PADRAO.eta,), (3,)), rotulo="otim-rapido")
        print("modo rápido: nada gravado")
        return

    saidas = rodar_paralelo(cfgs(SEMENTES, ETAS, EPOCAS), rotulo="otim")
    por_treino = pd.DataFrame([s["resumo"] for s in saidas]).sort_values(
        [*CHAVES, "eta", "epocas", "seed"]
    )
    hist = pd.concat([s["hist"] for s in saidas], ignore_index=True).sort_values(
        [*CHAVES, "eta", "epocas", "seed", "epoca"]
    )
    escolhas = escolher(por_treino)
    resumo = resumir(por_treino, escolhas)
    validacao = por_treino.groupby([*CHAVES, "eta", "epocas"], as_index=False)[
        ["acc_val", "acc_treino"]
    ].mean()
    conf = conferir(por_treino)
    TABLES.mkdir(parents=True, exist_ok=True)
    tex = TABLES / "tab_controle_otimizacao.tex"
    tex.write_text(tabela(resumo), encoding="utf-8")
    arquivos = [
        gravar_csv(por_treino, METRICS / "controle_otimizacao.csv"),
        gravar_csv(hist, METRICS / "historico_controle_otimizacao.csv"),
        gravar_csv(resumo, METRICS / "resumo_controle_otimizacao.csv"),
        gravar_csv(validacao, METRICS / "validacao_controle_otimizacao.csv"),
        gravar_csv(conf, METRICS / "conferencia_otimizacao.csv"),
        tex,
    ]
    max_dif = float(conf.filter(like="dif_").abs().to_numpy().max())
    print(resumo.round(3).to_string(index=False))
    print(f"treinos conferidos com a v1.0.0: {len(conf)}; maior diferença: {max_dif:.3g}")
    registrar(__file__, arquivos, {"treinos": len(saidas), "max_dif_v100": max_dif})


if __name__ == "__main__":
    main()
