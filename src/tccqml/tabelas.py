"""Geração das tabelas do Capítulo 4.

Cada tabela sai em dois formatos: `.csv` (para conferir e reprocessar) e
`.tex` (para colar no Overleaf sem editar nada). O `.tex` é um ambiente
`table` completo, com legenda acima (e uma legenda curta para a Lista de
Tabelas) e linha de fonte abaixo, no padrão ABNT das tabelas que já estão no
texto. Usa `tabular` com as regras do `booktabs` (`\\toprule`, `\\midrule`,
`\\bottomrule`), o mesmo estilo das tabelas do texto; o preâmbulo do TCC já
carrega o pacote, e o `.tex` gerado não carrega nenhum.

Oito das nove tabelas são DERIVADAS dos CSVs de `results/metrics/` e não
devem ser editadas à mão: se um número estiver estranho, o lugar de corrigir é
o experimento, não a tabela. A nona (`tab_qualitativa`) é julgamento do autor
e tem uma coluna que só se sabe depois de implementar.

Todas cabem na largura útil de uma página A4 com as margens do abnTeX2: as
que comparariam muitas colunas de "média ± desvio" lado a lado
(`tab_mesmo_p`, `tab_sensibilidade`) saem em formato longo, um grupo de
linhas por conjunto de dados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tccqml.config import PADRAO
from tccqml.espectro import ENCODINGS_COM_ESPECTRO

FONTE = "Fonte: Elaborada pelo autor."

# Ordem fixa nas tabelas: a ordem em que
# as codificações aparecem no Capítulo 2.
ORDEM_ENCODINGS = ["angle", "amplitude", "reuploading", "zz"]
ORDEM_DATASETS = ["xor", "moons", "circles"]

ROTULOS = {
    "angle": "\\textit{Angle}",
    "amplitude": "\\textit{Amplitude}",
    "reuploading": "\\textit{Re-uploading}",
    "zz": "\\textit{Feature map} ZZ",
    "xor": "\\textit{Xor}",
    "moons": "\\textit{Moons}",
    "circles": "\\textit{Circles}",
}


def _rotulo(chave: str) -> str:
    return ROTULOS.get(chave, str(chave))


def _dec(valor: float, casas: int = 3) -> str:
    """Número com vírgula decimal, como pede a ABNT e como o texto escreve."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _media_desvio(media: float, desvio: float) -> str:
    return f"{_dec(media)} $\\pm$ {_dec(desvio)}"


# Números pequenos vão por extenso na legenda, como manda a ABNT e como as
# tabelas já escritas no texto fazem. Acima de dez, algarismo.
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
    grupos: list[tuple[str, int]] | None = None,
    cabecalho: list[str] | None = None,
    preambulo: list[str] | None = None,
    divisorias: list[int] | None = None,
    caption_curta: str | None = None,
) -> str:
    """Um ambiente `table` completo, colável direto no Overleaf.

    `grupos` acrescenta uma linha acima do cabeçalho, como lista de
    (rótulo, número de colunas); rótulo vazio deixa as colunas sem grupo.
    `cabecalho` troca os nomes das colunas só no `.tex` — o `.csv` mantém os
    de `df`, que precisam ser únicos —, o que permite repetir "Prof." em dois
    grupos. `preambulo` entra logo depois do `\\centering` (tamanho da fonte,
    espaçamento entre colunas) e só vale dentro da tabela. `divisorias` lista
    os índices das linhas do corpo depois das quais entra um `\\hline`, para
    separar grupos numa tabela em formato longo. `caption_curta` é a legenda
    que entra na Lista de Tabelas, como nas tabelas escritas à mão no texto.
    """
    alinhamento = alinhamento or "l" + "c" * (df.shape[1] - 1)
    nomes = cabecalho or [str(c) for c in df.columns]
    if len(nomes) != df.shape[1]:
        raise ValueError("cabecalho precisa ter um nome por coluna")
    linhas_grupo: list[str] = []
    if grupos:
        if sum(n for _, n in grupos) != df.shape[1]:
            raise ValueError("os grupos precisam cobrir exatamente as colunas")
        celulas, regras, inicio = [], [], 1
        for rotulo, n in grupos:
            if rotulo:
                celulas.append(f"\\multicolumn{{{n}}}{{c}}{{{rotulo}}}")
                regras.append(f"\\cmidrule(lr){{{inicio}-{inicio + n - 1}}}")
            else:
                celulas.extend([""] * n)
            inicio += n
        linhas_grupo = [" & ".join(celulas) + " \\\\", " ".join(regras)]
    corpo = []
    for i, linha in enumerate(df.itertuples(index=False)):
        corpo.append(" & ".join("" if pd.isna(v) else str(v) for v in linha) + " \\\\")
        if divisorias and i in divisorias:
            corpo.append("\\midrule")
    partes = [
        "\\begin{table}[htb]",
        "\\centering",
        *(preambulo or []),
        f"\\caption[{caption_curta}]{{{caption}}}" if caption_curta else f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alinhamento}}}",
        "\\toprule",
        *linhas_grupo,
        " & ".join(nomes) + " \\\\",
        "\\midrule",
        *corpo,
        "\\bottomrule",
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

    Cada acurácia é a do modelo da época escolhida pela validação, medida uma
    única vez no teste. A `acc_val` fica só no `resumo.csv`: com três
    conjuntos, uma coluna de validação ao lado de cada um dobraria a largura
    da tabela além do que cabe na página.
    """
    resumo = _ordenar(resumo, "encoding", ORDEM_ENCODINGS)
    tabela = pd.DataFrame({"Codificação": [_rotulo(e) for e in resumo["encoding"].unique()]})
    for dataset in [d for d in ORDEM_DATASETS if d in set(resumo["dataset"])]:
        fatia = resumo[resumo["dataset"] == dataset].set_index("encoding")
        tabela[_rotulo(dataset)] = [
            _media_desvio(fatia.loc[e, "acc_teste_mean"], fatia.loc[e, "acc_teste_std"])
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
            "Acurácia no conjunto de teste por codificação e conjunto de dados, "
            "com o modelo da época de maior acurácia de validação "
            f"(média $\\pm$ desvio padrão {sobre})."
        ),
        label="tab:acuracia",
        caption_curta="Acurácia de teste por codificação e conjunto de dados",
        nota=_nota_recursos(resumo),
    )
    return tabela, tex


def _nota_recursos(resumo: pd.DataFrame) -> str | None:
    """Qubits e `p` de cada codificação, lidos de `resumo.csv`.

    A comparação de acurácia não é entre modelos de mesmo tamanho, e a nota
    diz isso com os números de cada linha, em vez de uma frase fixa que envelhece
    quando uma codificação muda (foi o que aconteceu com o re-uploading, p = 18).
    """
    colunas = {"n_qubits", "n_params_circuito"}
    if not colunas <= set(resumo.columns):
        return None
    recursos = resumo.groupby("encoding", observed=True)[sorted(colunas)].first()
    partes = [
        f"{_rotulo(e)}, {int(recursos.loc[e, 'n_qubits'])} e {int(recursos.loc[e, 'n_params_circuito'])}"
        for e in [e for e in ORDEM_ENCODINGS if e in recursos.index]
    ]
    return (
        "Qubits e parâmetros do circuito ($p$) de cada codificação: "
        + "; ".join(partes)
        + " (ver Tabela~\\ref{tab:custo})."
    )


# --------------------------------------------------------------------------
# tab_custo — quem é caro em quê
# --------------------------------------------------------------------------


def _colunas_de_custo_por_passo(custo: pd.DataFrame) -> pd.DataFrame:
    """Garante as contagens de avaliações e shots, derivadas de `p`.

    São analíticas: dependem só de `n_params_circuito`, de |B| e de S. Um
    `resumo.csv` gerado antes da correção (2p + 1) não tem essas colunas; elas
    são então recalculadas com as mesmas funções de `circuit_stats` e o |B| e o
    S do protocolo, em vez de exigir a grade inteira de novo para uma conta.
    """
    from tccqml.circuit_stats import (
        aval_derivadas_amostra,
        aval_gradiente_amostra,
        n_aval_passo,
        n_shots_passo,
    )

    p = custo["n_params_circuito"].astype(int)
    derivadas = {
        "aval_derivadas_amostra": p.map(aval_derivadas_amostra),
        "aval_gradiente_amostra": p.map(aval_gradiente_amostra),
        "n_aval_passo": p.map(lambda v: n_aval_passo(v, PADRAO.batch_efetivo)),
        "n_shots_passo": p.map(lambda v: n_shots_passo(v, PADRAO.batch_efetivo, PADRAO.shots)),
    }
    for coluna, valores in derivadas.items():
        if coluna not in custo.columns:
            custo = custo.assign(**{coluna: valores})
    return custo


def _milhar(valor: int) -> str:
    """Inteiro com espaço fino de milhar no LaTeX, como o texto escreve 504 000."""
    return f"{int(valor):,}".replace(",", "\\,")


# Contagens de portas que o Cap. 3 ("Métricas") promete, gravadas por `comparar` desde que
# `COLUNAS_CUSTO` passou a incluí-las. Um `resumo.csv` anterior não as tem.
_PORTAS = ("gates_total", "gates_1q_encoding", "gates_2q_encoding", "gates_total_encoding")


def _colunas_de_portas(custo: pd.DataFrame) -> pd.DataFrame:
    """Garante as contagens de portas, medindo o circuito quando faltam.

    São estáticas: dependem só da arquitetura, não do treino. Quando o
    `resumo.csv` é anterior a essas colunas, cada circuito é montado de novo
    com o protocolo da grade (`L_var`, `R`) e medido pela mesma
    `stats()` que os runners usam — sem treino e sem número escrito à mão.
    """
    faltando = [c for c in _PORTAS if c not in custo.columns]
    if not faltando:
        return custo
    from tccqml import model
    from tccqml.circuit_stats import stats
    from tccqml.config import PADRAO

    medidas = {}
    for encoding in custo["encoding"]:
        clf = model.build(
            encoding,
            n_features=2,
            L_var=PADRAO.L_var,
            enc_kwargs={"R": PADRAO.R} if encoding == "reuploading" else None,
        )
        # A estrutura do circuito não depende do valor do dado, só da forma.
        medidas[encoding] = stats(clf, [0.3, 1.2])
    return custo.assign(
        **{c: [medidas[e][c] for e in custo["encoding"]] for c in faltando}
    )


def tab_custo(resumo: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Recursos do bloco de codificação e do circuito completo, e custo por passo.

    O bloco de codificação e o circuito completo vêm em grupos separados
    (Cap. 3, "Métricas"): profundidade, portas totais e portas de dois qubits de cada
    um, mais os qubits do circuito. As contagens de hardware vêm em colunas
    separadas: avaliações do circuito (quantos circuitos distintos por passo)
    e shots (quantas execuções). As 2p avaliações deslocadas ficam ao lado
    para comparação com a regra de deslocamento, mas o custo do gradiente é 2p + 1 por
    amostra, por causa do resíduo.
    """
    colunas = ["n_qubits", "n_params_circuito", "depth", "depth_encoding", "gates_2q"]
    extras = [
        c
        for c in (
            *_PORTAS,
            "aval_derivadas_amostra",
            "aval_gradiente_amostra",
            "n_aval_passo",
            "n_shots_passo",
        )
        if c in resumo.columns
    ]
    custo = resumo.groupby("encoding", as_index=False)[colunas + extras].first()
    custo = _colunas_de_portas(custo)
    custo = _colunas_de_custo_por_passo(custo)
    custo = _ordenar(custo, "encoding", ORDEM_ENCODINGS)

    def inteiros(coluna: str):
        return custo[coluna].astype(int).values

    tabela = pd.DataFrame(
        {
            "Codificação": [_rotulo(e) for e in custo["encoding"]],
            "$p$": inteiros("n_params_circuito"),
            "Codif.: prof.": inteiros("depth_encoding"),
            "Codif.: portas": inteiros("gates_total_encoding"),
            "Codif.: CNOTs": inteiros("gates_2q_encoding"),
            "Circuito: qubits": inteiros("n_qubits"),
            "Circuito: prof.": inteiros("depth"),
            "Circuito: portas": inteiros("gates_total"),
            "Circuito: CNOTs": inteiros("gates_2q"),
            "Desloc. ($2p$)": inteiros("aval_derivadas_amostra"),
            "Aval./amostra": inteiros("aval_gradiente_amostra"),
            "Aval./passo": inteiros("n_aval_passo"),
            "\\textit{Shots}/passo": [_milhar(v) for v in custo["n_shots_passo"]],
        }
    )
    tex = to_latex(
        tabela,
        caption=(
            "Custo de circuito por codificação, com $d = 2$ atributos e o "
            "\\textit{ansatz} congelado. Contagens feitas sobre o circuito já "
            "decomposto nas portas básicas."
        ),
        label="tab:custo",
        caption_curta="Custo de circuito por codificação",
        grupos=[
            ("", 2),
            ("Bloco de codificação", 3),
            ("Circuito completo", 4),
            ("Custo por passo", 4),
        ],
        # Cabeçalhos longos em duas linhas (`\shortstack`, LaTeX básico): com
        # 13 colunas, numa linha só a tabela passa da margem da página A4.
        cabecalho=[
            "Codificação", "$p$",
            "Prof.", "Portas", "CNOTs",
            "Qubits", "Prof.", "Portas", "CNOTs",
            "\\shortstack{Desloc.\\\\($2p$)}",
            "\\shortstack{Aval./\\\\amostra}",
            "\\shortstack{Aval./\\\\passo}",
            "\\shortstack{\\textit{Shots}/\\\\passo}",
        ],
        preambulo=["\\footnotesize", "\\setlength{\\tabcolsep}{2pt}"],
        nota=(
            "Portas: total de portas básicas (rotações de um eixo, Hadamards e CNOTs), "
            "sem contar fases globais; "
            "CNOTs: portas de dois qubits. O bloco de codificação usa os mesmos "
            "qubits do circuito completo. "
            "Custo por passo: o que um passo do otimizador custaria em "
            "\\textit{hardware}. Contagens de gradiente analíticas: o treino usa retropropagação "
            "sobre o simulador, não \\textit{parameter-shift}. "
            "\\textit{Desloc.}: só as $2p$ avaliações deslocadas que dão as "
            "derivadas (Eq.~\\eqref{eq:parameter-shift}). \\textit{Aval./amostra}: $2p + 1$, com a "
            "avaliação sem deslocamento que dá o resíduo $f(x) + b - y$ do custo "
            "quadrático. \\textit{Aval./passo}: $(2p + 1)|B|$. "
            "\\textit{Shots/passo}: $(2p + 1)|B|S$, com $S$ \\textit{shots} por avaliação. "
            "No \\textit{amplitude}, o bloco de codificação inclui uma rotação $R_z$ de "
            "ângulo nulo emitida pela decomposição de preparação de estados do PennyLane."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_espectro — frequências medidas contra as previstas
# --------------------------------------------------------------------------


def tab_espectro(espectro: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Omega medido por FFT 2D contra o suporte previsto (tab:espectros, Cap. 2).

    O omega vai POR ATRIBUTO: o máximo entre os dois coincidiria com o previsto
    e esconderia que x_1 para em R - 1. Os termos são CONTADOS na FFT 2D, e
    não calculados como (2 omega_max + 1)^d.
    """
    tabela = pd.DataFrame(
        {
            "Codificação": [
                _rotulo(r.encoding) + ("" if pd.isna(r.R) else f" ($R = {int(r.R)}$)")
                for r in espectro.itertuples()
            ],
            "$\\omega_{\\max}$ previsto": espectro["omega_max_previsto"].astype(int).values,
            "$\\omega_{\\max}$ em $x_1$": espectro["omega_max_x1"].astype(int).values,
            "$\\omega_{\\max}$ em $x_2$": espectro["omega_max_x2"].astype(int).values,
            "Termos previstos": espectro["n_termos_previsto"].astype(int).values,
            "Termos medidos": espectro["n_termos_medido"].astype(int).values,
            "Cruzados": espectro["n_termos_cruzados"].astype(int).values,
            "$p$": espectro["n_params_circuito"].astype(int).values,
        }
    )
    tex = to_latex(
        tabela,
        caption=(
            "Espectro de Fourier medido por FFT 2D contra o limite da "
            "Tabela~\\ref{tab:espectros}, com $d = 2$ atributos."
        ),
        label="tab:espectro",
        caption_curta="Espectro de Fourier medido",
        cabecalho=[
            "Codificação",
            "\\shortstack{$\\omega_{\\max}$\\\\previsto}",
            "\\shortstack{$\\omega_{\\max}$\\\\em $x_1$}",
            "\\shortstack{$\\omega_{\\max}$\\\\em $x_2$}",
            "\\shortstack{Termos\\\\previstos}",
            "\\shortstack{Termos\\\\medidos}",
            "Cruzados",
            "$p$",
        ],
        preambulo=["\\small", "\\setlength{\\tabcolsep}{4pt}"],
        nota=(
            "A teoria dá um limite superior: nenhuma energia aparece fora de "
            "$\\{-R, \\dots, R\\}^d$, mas nem todos os termos permitidos aparecem. "
            "\\textit{Termos medidos}: pares $(\\omega_1, \\omega_2)$ com energia "
            "na FFT 2D; \\textit{Cruzados}: os que dependem dos dois atributos. "
            "Codificações \\textit{amplitude} e ZZ não aparecem porque a saída não "
            "é uma série de Fourier nos atributos. Em $x_1$, a frequência máxima é $R - 1$: "
            "a última inserção de $x_1$ atua só no qubit 0 e não afeta a leitura de $Z_0$."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_ablacao — o que some quando o ansatz perde as CNOTs
# --------------------------------------------------------------------------


def tab_ablacao(ablacao: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Acurácia de teste com e sem CNOTs no ansatz, por conjunto de dados.

    Os três conjuntos aparecem porque a queda não é específica do xor: sem
    CNOTs, <Z_0> só enxerga x_1 (Cap. 2, "Entrelaçamento e termos cruzados"), e
    qualquer conjunto que precise de
    x_2 cai.
    """
    agregado = ablacao.groupby(["dataset", "ansatz"])["acc_teste"].agg(["mean", "std"])
    datasets = [d for d in ORDEM_DATASETS if d in set(ablacao["dataset"])]
    nomes = {"strongly_entangling": "Com CNOTs", "local": "Sem CNOTs"}
    tabela = pd.DataFrame({"\\textit{Ansatz}": [nomes[a] for a in nomes]})
    for dataset in datasets:
        tabela[_rotulo(dataset)] = [
            _media_desvio(*agregado.loc[(dataset, a)]) if (dataset, a) in agregado.index else ""
            for a in nomes
        ]
    tex = to_latex(
        tabela,
        caption=(
            "Acurácia de teste do \\textit{angle} com e sem portas de dois qubits "
            "no \\textit{ansatz} (mesmo $p$)."
        ),
        label="tab:ablacao",
        caption_curta="Ablação do entrelaçamento no \\textit{angle}",
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_diagnostico — o amplitude com as duas normalizações
# --------------------------------------------------------------------------

# Rótulo de `diagnostico_amplitude.csv` (coluna `normalizacao`) -> linha da tabela.
# A do protocolo vem primeiro e é marcada como a usada no trabalho.
NORMALIZACOES = {
    "[0,pi]": "$[0, \\pi]$ (usada no trabalho)",
    "[-1,1]": "$[-1, 1]$",
}


def tab_diagnostico(diagnostico: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Acurácia de teste do amplitude em [0, pi] contra [-1, 1], por conjunto.

    O amplitude só vê a direção de x, e suas fronteiras são retas pela origem:
    a normalização decide onde a origem cai em relação aos dados. O trabalho
    mantém [0, pi] para todas as codificações; esta tabela mostra o preço
    dessa escolha para o amplitude.
    """
    agregado = diagnostico.groupby(["normalizacao", "dataset"])["acc_teste"].agg(["mean", "std"])
    datasets = [d for d in ORDEM_DATASETS if d in set(diagnostico["dataset"])]
    presentes = [n for n in NORMALIZACOES if n in set(diagnostico["normalizacao"])]
    tabela = pd.DataFrame({"Normalização": [NORMALIZACOES[n] for n in presentes]})
    for dataset in datasets:
        tabela[_rotulo(dataset)] = [
            _media_desvio(*agregado.loc[(n, dataset)]) if (n, dataset) in agregado.index else ""
            for n in presentes
        ]
    tex = to_latex(
        tabela,
        caption="Acurácia de teste do \\textit{amplitude} com duas normalizações dos dados.",
        label="tab:diagnostico",
        caption_curta="Normalização dos dados no \\textit{amplitude}",
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_mesmo_p — frequência contra número de parâmetros
# --------------------------------------------------------------------------


def tab_mesmo_p(
    varredura_R: pd.DataFrame, varredura_L_var: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Re-uploading com R = k repetições contra angle com L_var = k camadas: mesmo p.

    Com dois qubits os dois modelos têm p = 6k. O que muda é o suporte: o
    angle fica em Omega = {-1, 0, 1} por atributo, o re-uploading vai até R.
    Formato longo, um grupo de linhas por conjunto: seis colunas de
    "média ± desvio" lado a lado não cabem na página.
    """
    reup = varredura_R.groupby(["R", "dataset"])["acc_teste"].agg(["mean", "std"])
    angle = varredura_L_var.groupby(["L_var", "dataset"])["acc_teste"].agg(["mean", "std"])
    p_reup = varredura_R.groupby("R")["n_params_circuito"].first()
    p_angle = varredura_L_var.groupby("L_var")["n_params_circuito"].first()
    valores = sorted(set(p_reup.index.astype(int)) & set(p_angle.index.astype(int)))
    datasets = [d for d in ORDEM_DATASETS if d in set(varredura_R["dataset"])]
    for k in valores:
        if int(p_reup.loc[k]) != int(p_angle.loc[k]):
            raise ValueError(f"p diferente para k = {k}: a comparação deixa de ser de mesmo p")

    linhas, divisorias = [], []
    for dataset in datasets:
        for posicao, k in enumerate(valores):
            linhas.append(
                {
                    "Conjunto": _rotulo(dataset) if posicao == 0 else "",
                    "$k$": k,
                    "$p$": int(p_reup.loc[k]),
                    "\\textit{Angle} ($L_\\text{var} = k$)": _media_desvio(
                        *angle.loc[(k, dataset)]
                    ),
                    "\\textit{Re-uploading} ($R = k$)": _media_desvio(*reup.loc[(k, dataset)]),
                }
            )
        divisorias.append(len(linhas) - 1)
    tabela = pd.DataFrame(linhas)
    tex = to_latex(
        tabela,
        caption=(
            "Acurácia de teste do \\textit{angle} com $L_\\text{var} = k$ camadas no "
            "\\textit{ansatz} contra o \\textit{re-uploading} com $R = k$ repetições: "
            "mesmo $p$, espectros diferentes."
        ),
        label="tab:mesmo-p",
        caption_curta="\\textit{Angle} e \\textit{re-uploading} com o mesmo número de parâmetros",
        alinhamento="lcccc",
        divisorias=divisorias[:-1],
        nota=(
            "Com $k = 1$, os dois modelos são o mesmo circuito (uma codificação seguida "
            "de uma camada treinável), e as colunas coincidem."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_sensibilidade — a codificação estava bem treinada com o eta comum?
# --------------------------------------------------------------------------


def _eta(valor: float) -> str:
    return f"{float(valor):g}".replace(".", ",")


def tab_sensibilidade(
    resumo: pd.DataFrame, candidatos: list[float] | None = None
) -> tuple[pd.DataFrame, str]:
    """Acurácia de teste com o eta do protocolo e com o eta escolhido.

    Formato longo: um grupo de linhas por conjunto, uma linha por codificação.
    O eta escolhido sai de `resumo_sensibilidade_eta.csv`, onde foi escolhido
    pela VALIDAÇÃO: o teste aqui só mede o eta já escolhido. Se as duas
    colunas diferem menos que o desvio, a codificação não estava limitada pela
    taxa de aprendizado.
    """
    resumo = _ordenar(resumo, "encoding", ORDEM_ENCODINGS)
    encodings = list(resumo["encoding"].unique())
    eta_protocolo = _eta(resumo["eta_protocolo"].iloc[0])
    linhas, divisorias = [], []
    for dataset in [d for d in ORDEM_DATASETS if d in set(resumo["dataset"])]:
        fatia = resumo[resumo["dataset"] == dataset].set_index("encoding")
        presentes = [e for e in encodings if e in fatia.index]
        for posicao, e in enumerate(presentes):
            linha = fatia.loc[e]
            linhas.append(
                {
                    "Conjunto": _rotulo(dataset) if posicao == 0 else "",
                    "Codificação": _rotulo(e),
                    "$\\eta$ escolhido": _eta(linha["eta_escolhido"]),
                    f"Teste, $\\eta = {eta_protocolo.replace(',', '{,}')}$": _media_desvio(
                        linha["acc_teste_mean_protocolo"], linha["acc_teste_std_protocolo"]
                    ),
                    "Teste, $\\eta$ escolhido": _media_desvio(
                        linha["acc_teste_mean_escolhido"], linha["acc_teste_std_escolhido"]
                    ),
                }
            )
        divisorias.append(len(linhas) - 1)
    tabela = pd.DataFrame(linhas)

    n = _n_sementes(resumo.rename(columns={"n_sementes": "acc_teste_count"}))
    if n is None:
        sobre = "sobre as sementes"
    elif n == 1:
        sobre = "sobre uma semente"
    else:
        sobre = f"sobre {_POR_EXTENSO.get(n, n)} sementes"
    lista = " entre " + ", ".join(_eta(c) for c in sorted(candidatos)) if candidatos else ""
    tex = to_latex(
        tabela,
        caption=(
            "Sensibilidade à taxa de aprendizado: acurácia de teste com o $\\eta$ do "
            f"protocolo ({eta_protocolo}) e com o $\\eta$ escolhido{lista} "
            "pela maior acurácia média de validação "
            f"(média $\\pm$ desvio padrão {sobre})."
        ),
        label="tab:sensibilidade",
        caption_curta="Sensibilidade à taxa de aprendizado",
        alinhamento="llccc",
        divisorias=divisorias[:-1],
        nota=(
            "A escolha de $\\eta$ usa só a validação; o conjunto de teste não "
            "participa dela. Empate na validação: fica o $\\eta$ do protocolo, se "
            "empatado; senão, o menor. A comparação principal "
            "(Tabela~\\ref{tab:acuracia}) continua com o $\\eta$ do protocolo para "
            "todas as codificações."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_verificacoes — o que os dados permitem, independente do circuito
# --------------------------------------------------------------------------


def tab_verificacoes(controle: pd.DataFrame, limiar: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Controle linear por conjunto e regra g > t no circles, acurácia de teste.

    Lê os CSVs POR TREINO (`controle_linear.csv`, `limiar_circles.csv`) e
    agrega aqui, como a `tab_ablacao`. As duas linhas da regra g > t só têm a
    coluna do circles: a função g foi pensada para a geometria dele.
    """
    datasets = [d for d in ORDEM_DATASETS if d in set(controle["dataset"])]
    agregado = controle.groupby("dataset")["acc_teste"].agg(["mean", "std"])
    linhas = [
        {
            "Verificação": "Controle linear (9 funções)",
            **{_rotulo(d): _media_desvio(*agregado.loc[d]) for d in datasets},
        }
    ]
    por_versao = limiar.groupby("versao")
    versoes = (("sem_ruido", "$g/2 > t$, sem ruído"), ("com_ruido", "$g/2 > t$, com ruído"))
    for versao, rotulo in versoes:
        if versao not in por_versao.groups:
            continue
        fatia = por_versao.get_group(versao)
        if versao == "com_ruido" and "ruido" in fatia.columns:
            rotulo = f"$g/2 > t$, ruído {_dec(float(fatia['ruido'].iloc[0]), 2)}"
        linha = {"Verificação": rotulo, **{_rotulo(d): "--" for d in datasets}}
        linha[_rotulo("circles")] = _media_desvio(fatia["acc_teste"].mean(), fatia["acc_teste"].std())
        linhas.append(linha)
    tabela = pd.DataFrame(linhas)

    n = controle["seed"].nunique()
    sobre = "sobre uma semente" if n == 1 else f"sobre {_POR_EXTENSO.get(n, n)} sementes"
    tex = to_latex(
        tabela,
        caption=(
            "Verificações sobre os dados: acurácia de teste da regressão logística "
            "nas nove funções $\\{1, \\cos x_j, \\sin x_j\\}$ e seus produtos, e da regra "
            "$g(x)/2 > t$, com $g(x) = \\sin x_1 + \\sin x_2$, no \\textit{circles} "
            f"(média $\\pm$ desvio padrão {sobre})."
        ),
        label="tab:verificacoes",
        caption_curta="Verificações independentes do treinamento quântico",
        nota=(
            "Coordenadas já normalizadas para $[0, \\pi]$, com as mesmas partições e "
            "sementes da comparação principal. A regressão logística tem "
            f"hiperparâmetros fixados antes de rodar ($C = {_dec(PADRAO.controle_linear_C, 1)}$); "
            "o limiar $t$ e o sentido da desigualdade são escolhidos no treino. "
            "Um separador nesse espaço de funções não prova que o circuito "
            "quântico realize ou aprenda os mesmos coeficientes."
        ),
    )
    return tabela, tex


# --------------------------------------------------------------------------
# tab_qualitativa — a única preenchida à mão
# --------------------------------------------------------------------------

# Pontos fortes e fracos de cada codificação implementada, mais a coluna de
# dificuldade de implementação, que só se sabe depois de implementar — e é
# julgamento do autor, não medição. Os valores abaixo registram a experiência
# desta implementação; revise-os antes de colar no texto.
QUALITATIVA = [
    {
        "Codificação": "angle",
        "Ponto forte": "bloco de codificação com profundidade 1 e sem CNOTs",
        "Ponto fraco": "um qubit por atributo; suporte $\\Omega = \\{-1,0,1\\}$ por atributo",
        "Dificuldade": "baixa",
        "Observação": "primitiva pronta no PennyLane; nenhuma armadilha",
    },
    {
        "Codificação": "amplitude",
        "Ponto forte": "$\\lceil \\log_2 d \\rceil$ qubits para $d$ atributos",
        "Ponto fraco": "descarta a norma do vetor; fronteira formada por direções que partem da origem; preparação genérica com $O(d)$ CNOTs",
        "Dificuldade": "média",
        "Observação": (
            "a normalização dos dados interage com a codificação e precisa de "
            "diagnóstico próprio"
        ),
    },
    {
        "Codificação": "reuploading",
        "Ponto forte": "suporte cresce com $R$: $\\Omega = \\{-R,\\dots,R\\}$ por atributo",
        "Ponto fraco": "profundidade e $p$ crescem linearmente com $R$",
        "Dificuldade": "média",
        "Observação": (
            "não cabe no formato \\emph{codifica e depois aplica o ansatz}; "
            "exigiu generalizar a montagem do circuito"
        ),
    },
    {
        "Codificação": "zz",
        "Ponto forte": "termos que dependem de pares de atributos já no bloco de dados",
        "Ponto fraco": "$r_{\\mathrm{ZZ}}\\,d(d-1)$ CNOTs e profundidade $O(r_{\\mathrm{ZZ}}\\,d^2)$",
        "Dificuldade": "alta",
        "Observação": (
            "construção explícita; sem as Hadamards o operador é diagonal e o "
            "circuito fica inerte sem dar erro"
        ),
    },
]


def tab_qualitativa() -> tuple[pd.DataFrame, str]:
    """Pontos fortes e fracos, com a dificuldade de implementação observada."""
    tabela = pd.DataFrame(QUALITATIVA)
    tabela["Codificação"] = tabela["Codificação"].map(_rotulo)
    tex = to_latex(
        tabela.drop(columns=["Observação"]),
        caption=(
            "Comparação qualitativa das codificações implementadas, para as "
            "construções descritas no Capítulo~\\ref{cap:fundamentos}, com a "
            "dificuldade de implementação observada neste trabalho."
        ),
        label="tab:qualitativa",
        caption_curta="Comparação qualitativa das codificações implementadas",
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
        }.items():
            gerados.append(_escrever(df, tex, nome, destino))

    espectro = _ler("espectro.csv")
    if espectro is not None:
        df, tex = tab_espectro(espectro)
        gerados.append(_escrever(df, tex, "tab_espectro", destino))

    ablacao = _ler("ablacao_entrelacamento.csv")
    if ablacao is not None:
        df, tex = tab_ablacao(ablacao)
        gerados.append(_escrever(df, tex, "tab_ablacao", destino))

    diagnostico = _ler("diagnostico_amplitude.csv")
    if diagnostico is not None:
        df, tex = tab_diagnostico(diagnostico)
        gerados.append(_escrever(df, tex, "tab_diagnostico", destino))

    varredura_R = _ler("varredura_R.csv")
    varredura_L_var = _ler("varredura_L_var.csv")
    if varredura_R is not None and varredura_L_var is not None:
        df, tex = tab_mesmo_p(varredura_R, varredura_L_var)
        gerados.append(_escrever(df, tex, "tab_mesmo_p", destino))

    sensibilidade = _ler("resumo_sensibilidade_eta.csv")
    if sensibilidade is not None:
        por_treino = metrics / "sensibilidade_eta.csv"
        candidatos = (
            sorted(pd.read_csv(por_treino)["eta"].unique()) if por_treino.exists() else None
        )
        df, tex = tab_sensibilidade(sensibilidade, candidatos)
        gerados.append(_escrever(df, tex, "tab_sensibilidade", destino))

    controle = _ler("controle_linear.csv")
    limiar = _ler("limiar_circles.csv")
    if controle is not None and limiar is not None:
        df, tex = tab_verificacoes(controle, limiar)
        gerados.append(_escrever(df, tex, "tab_verificacoes", destino))

    df, tex = tab_qualitativa()
    gerados.append(_escrever(df, tex, "tab_qualitativa", destino))

    if verbose:
        for caminho in gerados:
            print(f"escrito {caminho}")
        print(f"\ncodificações com espectro aplicável: {', '.join(ENCODINGS_COM_ESPECTRO)}")
    return gerados
