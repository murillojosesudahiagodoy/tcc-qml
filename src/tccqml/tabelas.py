"""Geração das tabelas do Capítulo 4 — a Etapa 10 do roteiro.

Cada tabela sai em dois formatos: `.csv` (para conferir e reprocessar) e
`.tex` (para colar no Overleaf sem editar nada). O `.tex` é um ambiente
`table` completo, com legenda acima e linha de fonte abaixo, no padrão ABNT
das Tabelas 2-5 que já estão no texto, e usa só `tabular` e `\\hline` — nenhum
pacote novo precisa ser carregado.

Quatro das cinco tabelas são DERIVADAS dos CSVs de `results/metrics/` e não
devem ser editadas à mão: se um número estiver estranho, o lugar de corrigir é
o experimento, não a tabela. A quinta (`tab_qualitativa`) é julgamento do autor
e tem uma coluna que só se sabe depois de implementar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tccqml.espectro import ENCODINGS_COM_ESPECTRO

FONTE = "Fonte: Elaborada pelo autor."

# Ordem fixa nas tabelas: a da Etapa 9 do roteiro, que é também a ordem em que
# as codificações aparecem no Capítulo 2.
ORDEM_ENCODINGS = ["angle", "amplitude", "reuploading", "zz"]
ORDEM_DATASETS = ["xor", "moons", "circles"]

ROTULOS = {
    "angle": "\\textit{Angle}",
    "amplitude": "\\textit{Amplitude}",
    "reuploading": "\\textit{Re-uploading}",
    "zz": "\\textit{Feature map} ZZ",
    "xor": "XOR",
    "moons": "\\textit{Moons}",
    "circles": "\\textit{Circles}",
}


def _rotulo(chave: str) -> str:
    return ROTULOS.get(chave, str(chave))


# Números pequenos vão por extenso na legenda, como manda a ABNT e como as
# Tabelas 2-5 já escritas fazem. Acima de dez, algarismo.
_POR_EXTENSO = {
    1: "uma", 2: "duas", 3: "três", 4: "quatro", 5: "cinco",
    6: "seis", 7: "sete", 8: "oito", 9: "nove", 10: "dez",
}


def _n_sementes(resumo: pd.DataFrame) -> int | None:
    """Quantas sementes entraram em cada média, lido de `resumo.csv`.

    `experiments.resumir` grava isso em `acc_teste_count`. Devolve None quando
    a coluna não existe ou quando as células discordam — a legenda então omite
    o número, em vez de afirmar um que não foi medido.
    """
    if "acc_teste_count" not in resumo.columns:
        return None
    valores = set(resumo["acc_teste_count"].dropna().astype(int))
    return valores.pop() if len(valores) == 1 else None


def _ordenar(df: pd.DataFrame, coluna: str, ordem: list[str]) -> pd.DataFrame:
    presentes = [v for v in ordem if v in set(df[coluna])]
    restantes = [v for v in df[coluna].unique() if v not in presentes]
    categorias = presentes + list(restantes)
    return df.assign(
        **{coluna: pd.Categorical(df[coluna], categories=categorias, ordered=True)}
    ).sort_values(coluna)


def to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    alinhamento: str | None = None,
    nota: str | None = None,
) -> str:
    """Um ambiente `table` completo, colável direto no Overleaf."""
    alinhamento = alinhamento or "l" + "c" * (df.shape[1] - 1)
    cabecalho = " & ".join(str(c) for c in df.columns) + " \\\\"
    corpo = [
        " & ".join("" if pd.isna(v) else str(v) for v in linha) + " \\\\"
        for linha in df.itertuples(index=False)
    ]
    partes = [
        "\\begin{table}[htb]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alinhamento}}}",
        "\\hline",
        cabecalho,
        "\\hline",
        *corpo,
        "\\hline",
        "\\end{tabular}",
    ]
    if nota:
        partes.append(f"\\\\[2pt]\n{{\\footnotesize {nota}}}")
    partes += [f"\\\\[2pt]\n{{\\footnotesize {FONTE}}}", "\\end{table}", ""]
    return "\n".join(partes)


def _escrever(df: pd.DataFrame, tex: str, nome: str, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino / f"{nome}.csv", index=False)
    (destino / f"{nome}.tex").write_text(tex, encoding="utf-8")
    return destino / f"{nome}.tex"


# --------------------------------------------------------------------------
# tab_acuracia — qual codificação acertou mais, em cada dataset
# --------------------------------------------------------------------------


def tab_acuracia(resumo: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Acurácia de teste, média +- desvio entre sementes.

    O desvio fica na tabela de propósito: com poucas sementes, diferença menor
    que o desvio não sustenta afirmação no Capítulo 4.
    """
    resumo = _ordenar(resumo, "encoding", ORDEM_ENCODINGS)
    tabela = pd.DataFrame({"Codificação": [_rotulo(e) for e in resumo["encoding"].unique()]})
    for dataset in [d for d in ORDEM_DATASETS if d in set(resumo["dataset"])]:
        fatia = resumo[resumo["dataset"] == dataset].set_index("encoding")
        tabela[_rotulo(dataset)] = [
            f"{fatia.loc[e, 'acc_teste_mean']:.3f} $\\pm$ {fatia.loc[e, 'acc_teste_std']:.3f}"
            if e in fatia.index
            else ""
            for e in resumo["encoding"].unique()
        ]
    n = _n_sementes(resumo)
    if n is None:
        sobre = "sobre as sementes"
    elif n == 1:
        sobre = "sobre uma semente"
    else:
        sobre = f"sobre {_POR_EXTENSO.get(n, n)} sementes"
    tex = to_latex(
        tabela,
        caption=(
            "Acurácia no conjunto de teste por codificação e conjunto de dados "
            f"(média $\\pm$ desvio padrão {sobre})."
        ),
        label="tab:acuracia",
        nota=(
            "O \\textit{amplitude} usa 1 qubit e 6 parâmetros, contra 2 qubits e "
            "12 das demais (ver Tabela~\\ref{tab:custo})."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_custo — quem é caro em quê
# --------------------------------------------------------------------------


def tab_custo(resumo: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Os itens que a Etapa 9 manda registrar, um por coluna."""
    colunas = [
        "n_qubits",
        "n_params_circuito",
        "depth",
        "depth_encoding",
        "gates_1q",
        "gates_2q",
        "avaliacoes_por_gradiente",
    ]
    custo = resumo.groupby("encoding", as_index=False)[colunas].first()
    custo = _ordenar(custo, "encoding", ORDEM_ENCODINGS)
    tabela = pd.DataFrame(
        {
            "Codificação": [_rotulo(e) for e in custo["encoding"]],
            "Qubits": custo["n_qubits"].astype(int).values,
            "$p$": custo["n_params_circuito"].astype(int).values,
            "Prof. total": custo["depth"].astype(int).values,
            "Prof. codif.": custo["depth_encoding"].astype(int).values,
            "Portas 1q": custo["gates_1q"].astype(int).values,
            "Portas 2q": custo["gates_2q"].astype(int).values,
            "Aval./grad.": custo["avaliacoes_por_gradiente"].astype(int).values,
        }
    )
    tex = to_latex(
        tabela,
        caption=(
            "Custo de circuito por codificação, com $n = 2$ atributos e o "
            "\\textit{ansatz} congelado. Contagens feitas sobre o circuito já "
            "decomposto nas portas básicas."
        ),
        label="tab:custo",
        nota=(
            "\\textit{Aval./grad.} é analítico (Eq.~2.47): o treino usa "
            "retropropagação sobre o simulador, não \\textit{parameter-shift}."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_convergencia — qual treinou mais rápido
# --------------------------------------------------------------------------


def tab_convergencia(resumo: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Época em que a acurácia de treino atinge 90% do seu valor final.

    Mede velocidade, não qualidade: um modelo ruim pode convergir depressa para
    o seu próprio teto baixo, e é por isso que a coluna de custo final vem ao
    lado.
    """
    resumo = _ordenar(resumo, "encoding", ORDEM_ENCODINGS)
    tabela = pd.DataFrame({"Codificação": [_rotulo(e) for e in resumo["encoding"].unique()]})
    for dataset in [d for d in ORDEM_DATASETS if d in set(resumo["dataset"])]:
        fatia = resumo[resumo["dataset"] == dataset].set_index("encoding")
        tabela[_rotulo(dataset)] = [
            f"{fatia.loc[e, 'epoca_90pct_mean']:.1f}" if e in fatia.index else ""
            for e in resumo["encoding"].unique()
        ]
    fatia_custo = resumo.groupby("encoding")["custo_final_mean"].mean()
    tabela["Custo final médio"] = [
        f"{fatia_custo.get(e, float('nan')):.3f}" for e in resumo["encoding"].unique()
    ]
    tex = to_latex(
        tabela,
        caption=(
            "Velocidade de convergência: primeira época em que a acurácia de "
            "treino atinge 90\\% do seu valor final (média sobre as sementes)."
        ),
        label="tab:convergencia",
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_espectro — frequências medidas contra as previstas
# --------------------------------------------------------------------------


def tab_espectro(espectro: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Omega medido por FFT contra o previsto na Tabela 5 (p. 52)."""
    tabela = pd.DataFrame(
        {
            "Codificação": [
                _rotulo(r.encoding) + ("" if pd.isna(r.L_reup) else f" ($L = {int(r.L_reup)}$)")
                for r in espectro.itertuples()
            ],
            "$\\omega_{\\max}$ previsto": espectro["omega_max_previsto"].astype(int).values,
            "$\\omega_{\\max}$ medido": espectro["omega_max_medido"].astype(int).values,
            "Termos previstos": espectro["n_termos_previsto"].astype(int).values,
            "Termos medidos": espectro["n_termos_medido"].astype(int).values,
            "$p$": espectro["n_params_circuito"].astype(int).values,
        }
    )
    tex = to_latex(
        tabela,
        caption=(
            "Espectro de Fourier medido por FFT contra o previsto na "
            "Tabela~\\ref{tab:espectro-previsto}, com $d = 2$ atributos."
        ),
        label="tab:espectro",
        nota=(
            "O número de termos cresce como $(2L+1)^d$ enquanto $p$ cresce "
            "linearmente: os coeficientes não podem ser escolhidos de forma "
            "independente. Codificações \\textit{amplitude} e ZZ não aparecem "
            "porque a saída não é uma série de Fourier nos atributos."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_qualitativa — a única preenchida à mão
# --------------------------------------------------------------------------

# A Tabela 4 (p. 46) do texto já traz forte/fraco. O que falta é a coluna de
# dificuldade de implementação, que só se sabe depois de implementar — e é
# julgamento do autor, não medição. Os valores abaixo registram a experiência
# desta implementação; revise-os antes de colar no texto.
QUALITATIVA = [
    {
        "Codificação": "angle",
        "Ponto forte": "circuito raso, um qubit por atributo, custo mínimo",
        "Ponto fraco": "espectro limitado a $\\Omega = \\{-1,0,1\\}$ por atributo",
        "Dificuldade": "baixa",
        "Observação": "primitiva pronta no PennyLane; nenhuma armadilha",
    },
    {
        "Codificação": "amplitude",
        "Ponto forte": "$\\lceil \\log_2 d \\rceil$ qubits: o menor número de qubits",
        "Ponto fraco": "descarta a norma do vetor; fronteira quádrica homogênea",
        "Dificuldade": "média",
        "Observação": (
            "a normalização dos dados interage com a codificação e precisa de "
            "diagnóstico próprio"
        ),
    },
    {
        "Codificação": "reuploading",
        "Ponto forte": "espectro cresce com $L$: $\\Omega = \\{-L,\\dots,L\\}$",
        "Ponto fraco": "profundidade e $p$ crescem linearmente com $L$",
        "Dificuldade": "média",
        "Observação": (
            "não cabe no formato \\emph{codifica e depois aplica o ansatz}; "
            "exigiu generalizar a montagem do circuito"
        ),
    },
    {
        "Codificação": "zz",
        "Ponto forte": "insere correlações entre atributos no bloco de dados",
        "Ponto fraco": "$r\\,d(d-1)$ CNOTs e profundidade $O(r d^2)$",
        "Dificuldade": "alta",
        "Observação": (
            "construção explícita; sem as Hadamards o operador é diagonal e o "
            "circuito fica inerte sem dar erro"
        ),
    },
]


def tab_qualitativa() -> tuple[pd.DataFrame, str]:
    """Tabela 4 do texto acrescida da coluna de dificuldade de implementação."""
    tabela = pd.DataFrame(QUALITATIVA)
    tabela["Codificação"] = tabela["Codificação"].map(_rotulo)
    tex = to_latex(
        tabela.drop(columns=["Observação"]),
        caption=(
            "Comparação qualitativa das codificações, com a dificuldade de "
            "implementação observada neste trabalho."
        ),
        label="tab:qualitativa",
        alinhamento="lp{4.2cm}p{4.2cm}c",
    )
    return tabela, tex


# --------------------------------------------------------------------------


def gerar_todas(out: str | Path = "results", verbose: bool = True) -> list[Path]:
    """Gera todas as tabelas a partir dos CSVs já produzidos pelos runners.

    Cada tabela é pulada (com aviso) se o CSV de origem ainda não existe, para
    que a geração parcial seja possível enquanto a grade ainda roda.
    """
    out = Path(out)
    metrics = out / "metrics"
    destino = out / "tables"
    gerados: list[Path] = []

    def _ler(nome: str) -> pd.DataFrame | None:
        caminho = metrics / nome
        if not caminho.exists():
            if verbose:
                print(f"[pulado] {caminho} não existe — rode o experimento primeiro")
            return None
        return pd.read_csv(caminho)

    resumo = _ler("resumo.csv")
    if resumo is not None:
        for nome, (df, tex) in {
            "tab_acuracia": tab_acuracia(resumo),
            "tab_custo": tab_custo(resumo),
            "tab_convergencia": tab_convergencia(resumo),
        }.items():
            gerados.append(_escrever(df, tex, nome, destino))

    espectro = _ler("espectro.csv")
    if espectro is not None:
        df, tex = tab_espectro(espectro)
        gerados.append(_escrever(df, tex, "tab_espectro", destino))

    df, tex = tab_qualitativa()
    gerados.append(_escrever(df, tex, "tab_qualitativa", destino))

    if verbose:
        for caminho in gerados:
            print(f"escrito {caminho}")
        print(f"\ncodificações com espectro aplicável: {', '.join(ENCODINGS_COM_ESPECTRO)}")
    return gerados
