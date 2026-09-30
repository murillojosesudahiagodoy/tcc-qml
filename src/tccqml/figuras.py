"""Geração das figuras do TCC — a Etapa 10 do roteiro.

Sete figuras em PDF vetorial, com os nomes exatos que o LaTeX referencia.

Convenções, todas deliberadas:

- **PDF vetorial**, não PNG: o texto vai ser lido em tela e impresso, e curva
  rasterizada em 300 dpi fica visivelmente pior que vetor no papel;
- **sem título embutido** na imagem: a legenda é responsabilidade do
  `\\caption` do LaTeX, e título duplicado é erro de diagramação;
- **fonte de 10 pt** no tamanho final da página, para não haver rótulo
  ilegível depois do `\\includegraphics`;
- **paleta que funciona em preto e branco**: cor E marcador E traço distintos
  por série, porque a versão impressa do TCC pode ser monocromática.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from tccqml import model
from tccqml.config import PADRAO
from tccqml.data import load_dataset
from tccqml.espectro import espectro_por_atributo
from tccqml.model import pesos_iniciais

# Backend sem janela: a geração das figuras precisa rodar em terminal, em CI e
# dentro do pytest, onde não há display nenhum para abrir.
matplotlib.use("Agg")

ORDEM_ENCODINGS = ["angle", "amplitude", "reuploading", "zz"]
ORDEM_DATASETS = ["xor", "moons", "circles"]

NOMES = {
    "angle": "Angle",
    "amplitude": "Amplitude",
    "reuploading": "Re-uploading",
    "zz": "Feature map ZZ",
    "xor": "XOR",
    "moons": "Moons",
    "circles": "Circles",
}

# Versões curtas para eixos apertados, onde o nome completo quebraria em três
# linhas e ficaria ilegível a 10 pt na página.
NOMES_CURTOS = {**NOMES, "zz": "ZZ", "reuploading": "Re-upl."}

# Uma cor, um marcador e um traço por codificação: legível em cores e em P&B.
ESTILO = {
    "angle": {"color": "#1b1b1b", "marker": "o", "linestyle": "-"},
    "amplitude": {"color": "#4c72b0", "marker": "s", "linestyle": "--"},
    "reuploading": {"color": "#c44e52", "marker": "^", "linestyle": "-."},
    "zz": {"color": "#55a868", "marker": "D", "linestyle": ":"},
}

RC = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
}


def _estilo(encoding: str) -> dict:
    return ESTILO.get(encoding, {"color": "#7f7f7f", "marker": "v", "linestyle": "-"})


def _salvar(fig, nome: str, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / nome
    fig.savefig(caminho, format="pdf")
    plt.close(fig)
    return caminho


def _presentes(valores, ordem: list[str]) -> list[str]:
    conjunto = set(valores)
    return [v for v in ordem if v in conjunto]


# --------------------------------------------------------------------------


def fig_datasets(destino: Path) -> Path:
    """Os três conjuntos sintéticos, já normalizados em [0, pi]."""
    datasets = ORDEM_DATASETS
    fig, eixos = plt.subplots(1, len(datasets), figsize=(6.5, 2.3), sharey=True)
    for eixo, nome in zip(np.atleast_1d(eixos), datasets):
        ds = load_dataset(nome)
        for classe, marcador, cor in ((0, "o", "#1b1b1b"), (1, "^", "#c44e52")):
            m = ds.y_train == classe
            eixo.scatter(
                ds.X_train[m, 0],
                ds.X_train[m, 1],
                s=12,
                marker=marcador,
                c=cor,
                alpha=0.75,
                label=f"classe {classe}",
                linewidths=0,
            )
        eixo.set_xlabel(f"{NOMES[nome]}\n$x_1$")
        eixo.set_xlim(-0.2, np.pi + 0.2)
        eixo.set_ylim(-0.2, np.pi + 0.2)
    np.atleast_1d(eixos)[0].set_ylabel("$x_2$")
    np.atleast_1d(eixos)[-1].legend(loc="upper right", framealpha=0.9)
    return _salvar(fig, "datasets.pdf", destino)


def fig_curvas_treinamento(comparacao: pd.DataFrame, destino: Path) -> Path:
    """Custo e acurácia de teste por época, uma coluna por dataset."""
    datasets = _presentes(comparacao["dataset"], ORDEM_DATASETS)
    encodings = _presentes(comparacao["encoding"], ORDEM_ENCODINGS)
    fig, eixos = plt.subplots(
        2, len(datasets), figsize=(6.5, 4.2), sharex=True, sharey="row", squeeze=False
    )
    for coluna, dataset in enumerate(datasets):
        for encoding in encodings:
            fatia = comparacao[
                (comparacao["dataset"] == dataset) & (comparacao["encoding"] == encoding)
            ]
            # Média entre sementes: a curva de uma semente só é ruído.
            media = fatia.groupby("epoca")[["custo", "acc_teste"]].mean()
            estilo = _estilo(encoding)
            for linha, coluna_dado in enumerate(("custo", "acc_teste")):
                eixos[linha][coluna].plot(
                    media.index,
                    media[coluna_dado],
                    label=NOMES.get(encoding, encoding),
                    linewidth=1.3,
                    markevery=6,
                    markersize=4,
                    **estilo,
                )
        eixos[1][coluna].set_xlabel(f"{NOMES.get(dataset, dataset)}\népoca")
    eixos[0][0].set_ylabel("custo quadrático")
    eixos[1][0].set_ylabel("acurácia de teste")
    # Acima da grade, numa linha: dentro de um painel ela cobria as curvas.
    alcas, rotulos = eixos[0][0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="outside upper center", ncol=len(rotulos), frameon=False)
    return _salvar(fig, "curvas-treinamento.pdf", destino)


def fig_fronteiras(
    resumo: pd.DataFrame, pesos_dir: Path, destino: Path, seed: int | None = None
) -> Path:
    """Fronteiras aprendidas, grade codificação x dataset.

    Usa os pesos gravados pela grade — nada é retreinado aqui, então a figura é
    literalmente o modelo que gerou os números das tabelas.
    """
    seed = seed or PADRAO.seed
    datasets = _presentes(resumo["dataset"], ORDEM_DATASETS)
    encodings = _presentes(resumo["encoding"], ORDEM_ENCODINGS)
    fig, eixos = plt.subplots(
        len(encodings),
        len(datasets),
        figsize=(6.5, 2.1 * len(encodings)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    grade = np.linspace(-0.1, np.pi + 0.1, 60)
    G1, G2 = np.meshgrid(grade, grade)
    pontos = np.column_stack([G1.ravel(), G2.ravel()])

    for i, encoding in enumerate(encodings):
        for j, dataset in enumerate(datasets):
            eixo = eixos[i][j]
            ds = load_dataset(dataset, seed=seed)
            arquivo = pesos_dir / f"{encoding}_{dataset}_{seed}.npz"
            if arquivo.exists():
                guardado = np.load(arquivo)
                clf = model.build(
                    encoding,
                    n_features=ds.n_features,
                    n_layers=PADRAO.n_layers,
                    enc_kwargs=(
                        {"L_reup": PADRAO.L_reup} if encoding == "reuploading" else None
                    ),
                )
                alpha = guardado["alpha"] if "alpha" in guardado.files else None
                z = np.asarray(
                    model.saida_continua(
                        clf, guardado["weights"], alpha, float(guardado["bias"]), pontos
                    ),
                    dtype=float,
                ).reshape(G1.shape)
                # `extend` preenche além de [-1, 1]: a saída é <Z_0> + b, e o
                # viés pode empurrá-la para fora do intervalo do observável.
                eixo.contourf(
                    G1,
                    G2,
                    z,
                    levels=np.linspace(-1, 1, 21),
                    cmap="RdBu",
                    alpha=0.55,
                    extend="both",
                )
                eixo.contour(G1, G2, z, levels=[0.0], colors="k", linewidths=1.1)
            for classe, marcador, cor in ((0, "o", "#1b1b1b"), (1, "^", "#7a0f16")):
                m = ds.y_test == classe
                eixo.scatter(
                    ds.X_test[m, 0],
                    ds.X_test[m, 1],
                    s=9,
                    marker=marcador,
                    c=cor,
                    linewidths=0,
                )
            if i == 0:
                eixo.set_title(NOMES.get(dataset, dataset))
            if j == 0:
                eixo.set_ylabel(f"{NOMES.get(encoding, encoding)}\n$x_2$")
            if i == len(encodings) - 1:
                eixo.set_xlabel("$x_1$")

    # Uma legenda só, abaixo da grade: o marcador é a classe VERDADEIRA e a cor
    # do fundo é a PREVISÃO. Um erro é um marcador sobre o fundo da outra
    # classe. O "RdBu" leva saída negativa (prevê 0) ao vermelho e positiva
    # (prevê 1) ao azul, como `model.prever`.
    mapa = plt.get_cmap("RdBu")
    itens = [
        Line2D([], [], marker="o", color="#1b1b1b", linestyle="", markersize=5, label="classe 0 (teste)"),
        Line2D([], [], marker="^", color="#7a0f16", linestyle="", markersize=5, label="classe 1 (teste)"),
        Patch(facecolor=mapa(0.2), alpha=0.55, label="prevê classe 0"),
        Patch(facecolor=mapa(0.8), alpha=0.55, label="prevê classe 1"),
        Line2D([], [], color="k", linewidth=1.1, label="fronteira (saída = 0)"),
    ]
    fig.legend(handles=itens, loc="outside lower center", ncol=3, framealpha=0.9)
    return _salvar(fig, "fronteiras-aprendidas.pdf", destino)


def fig_acuracia_vs_L(varredura: pd.DataFrame, destino: Path) -> Path:
    """Acurácia contra número de blocos do re-uploading (Previsões 1 e 5)."""
    fig, eixo = plt.subplots(figsize=(4.2, 2.8))
    datasets = _presentes(varredura["dataset"], ORDEM_DATASETS)
    estilos = [
        {"color": "#1b1b1b", "marker": "o", "linestyle": "-"},
        {"color": "#c44e52", "marker": "^", "linestyle": "--"},
        {"color": "#4c72b0", "marker": "s", "linestyle": "-."},
    ]
    for dataset, estilo in zip(datasets, estilos):
        fatia = varredura[varredura["dataset"] == dataset]
        agregado = fatia.groupby("L_reup")["acc_teste"].agg(["mean", "std"])
        eixo.errorbar(
            agregado.index,
            agregado["mean"],
            yerr=agregado["std"],
            capsize=3,
            linewidth=1.3,
            markersize=5,
            label=NOMES.get(dataset, dataset),
            **estilo,
        )
    eixo.set_xlabel("blocos de re-uploading $L$")
    eixo.set_ylabel("acurácia de teste")
    eixo.set_xticks(sorted(varredura["L_reup"].dropna().unique()))
    eixo.legend(framealpha=0.9)
    return _salvar(fig, "acuracia-vs-L.pdf", destino)


def fig_acuracia_vs_custo(resumo: pd.DataFrame, destino: Path) -> Path:
    """A pergunta final da Etapa 10: a melhor em acurácia foi a mais viável?

    Um ponto por (codificação, dataset). O eixo horizontal é a profundidade
    total do circuito, que é o custo que mais pesa em hardware ruidoso; o
    tamanho do marcador mostra as portas de dois qubits, que é onde o erro se
    concentra.
    """
    fig, eixo = plt.subplots(figsize=(5.6, 3.0))
    for encoding in _presentes(resumo["encoding"], ORDEM_ENCODINGS):
        fatia = resumo[resumo["encoding"] == encoding]
        estilo = _estilo(encoding)
        eixo.scatter(
            fatia["depth"],
            fatia["acc_teste_mean"],
            s=30 + 12 * fatia["gates_2q"],
            marker=estilo["marker"],
            facecolors="none",
            edgecolors=estilo["color"],
            linewidths=1.4,
            # O número de CNOTs é fixo por codificação, então ele vai no próprio
            # rótulo: explica o tamanho do marcador sem uma segunda legenda.
            label=f"{NOMES.get(encoding, encoding)} ({int(fatia['gates_2q'].iloc[0])} CNOTs)",
        )
        for linha in fatia.itertuples():
            eixo.annotate(
                NOMES.get(linha.dataset, linha.dataset),
                (linha.depth, linha.acc_teste_mean),
                textcoords="offset points",
                xytext=(6, -3),
                fontsize=7,
                color=estilo["color"],
            )
    eixo.set_xlabel("profundidade total do circuito")
    eixo.set_ylabel("acurácia de teste")
    # Profundidade é contagem de portas: marcas só em inteiros.
    eixo.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Folga à direita para o rótulo do ponto mais à direita não sair do quadro.
    eixo.margins(x=0.18)
    # A legenda vai ACIMA dos eixos: os pontos ocupam quase todo o quadro e
    # qualquer canto interno cobriria um rótulo de dataset.
    eixo.legend(
        framealpha=0.9,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        borderaxespad=0.0,
    )
    return _salvar(fig, "acuracia-vs-custo.pdf", destino)


def fig_espectro(destino: Path, L_reups: tuple[int, ...] = (1, 2, 3)) -> Path:
    """FFT de f(x) para L crescente: Omega_L abrindo com L (Tabela 5).

    O angle encoding entra como piso de comparação, com Omega = {-1, 0, 1}.
    """
    configuracoes = [("angle", None, None)] + [
        ("reuploading", {"L_reup": L}, L) for L in L_reups
    ]
    fig, eixos = plt.subplots(
        1, len(configuracoes), figsize=(6.5, 2.2), sharey=True, squeeze=False
    )
    for eixo, (encoding, kwargs, L) in zip(eixos[0], configuracoes):
        clf = model.build(encoding, n_features=2, n_layers=PADRAO.n_layers, enc_kwargs=kwargs)
        w, alpha, _ = pesos_iniciais(clf, seed=PADRAO.seed)
        # Escala maior nos pesos excita todo o espectro acessível (mesmo
        # motivo de `espectro.tabela_espectro`).
        espectros = espectro_por_atributo(clf, w * 3.0, alpha)
        melhor = max(espectros, key=lambda e: e.omega_max)
        n = min(len(melhor.amplitudes), 8)
        eixo.bar(range(n), melhor.amplitudes[:n], color="#4c72b0", width=0.6)
        limite = L if L is not None else 1
        eixo.axvline(limite + 0.5, color="#c44e52", linestyle="--", linewidth=1.1)
        rotulo = NOMES.get(encoding, encoding) + (f", $L = {L}$" if L else "")
        eixo.set_xlabel(f"{rotulo}\n$\\omega$")
        eixo.set_xticks(range(n))
    eixos[0][0].set_ylabel("$|c_\\omega|$")
    # As barras já são explicadas pelo eixo; só a linha tracejada precisa de
    # legenda. Fica fora dos painéis: eles são estreitos demais e qualquer
    # caixa interna encosta na própria linha que ela explica.
    limite = Line2D([], [], color="#c44e52", linestyle="--", linewidth=1.1, label="limite teórico")
    fig.legend(handles=[limite], loc="outside upper right", frameon=False)
    return _salvar(fig, "espectro-reuploading.pdf", destino)


def fig_custo_por_codificacao(resumo: pd.DataFrame, destino: Path) -> Path:
    """Barras de qubits, profundidade e portas de dois qubits por codificação."""
    colunas = ["n_qubits", "depth", "depth_encoding", "gates_2q", "n_params_circuito"]
    custo = resumo.groupby("encoding", as_index=False)[colunas].first()
    encodings = _presentes(custo["encoding"], ORDEM_ENCODINGS)
    custo = custo.set_index("encoding").loc[encodings]

    metricas = [
        ("n_qubits", "qubits"),
        ("depth", "profundidade total"),
        ("depth_encoding", "prof. da codificação"),
        ("gates_2q", "portas de 2 qubits"),
        ("n_params_circuito", "parâmetros $p$"),
    ]
    fig, eixos = plt.subplots(1, len(metricas), figsize=(6.8, 2.2), squeeze=False)
    posicoes = np.arange(len(encodings))
    for eixo, (coluna, rotulo) in zip(eixos[0], metricas):
        eixo.bar(
            posicoes,
            custo[coluna].values,
            color=[_estilo(e)["color"] for e in encodings],
            width=0.65,
        )
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels(
            [NOMES_CURTOS.get(e, e) for e in encodings], rotation=90, fontsize=8
        )
        eixo.set_xlabel(rotulo, fontsize=8)
        for pos, valor in zip(posicoes, custo[coluna].values):
            eixo.text(pos, valor, f"{int(valor)}", ha="center", va="bottom", fontsize=7)
        eixo.margins(y=0.18)
    itens = [Patch(facecolor=_estilo(e)["color"], label=NOMES.get(e, e)) for e in encodings]
    fig.legend(handles=itens, loc="outside upper center", ncol=len(itens), framealpha=0.9)
    return _salvar(fig, "custo-por-codificacao.pdf", destino)


# --------------------------------------------------------------------------


def gerar_todas(out: str | Path = "results", verbose: bool = True) -> list[Path]:
    """Gera todas as figuras a partir dos CSVs já produzidos pelos runners.

    Cada figura é pulada (com aviso) se o CSV de origem ainda não existe.
    """
    out = Path(out)
    metrics = out / "metrics"
    destino = out / "figures"
    gerados: list[Path] = []

    def _ler(nome: str) -> pd.DataFrame | None:
        caminho = metrics / nome
        if not caminho.exists():
            if verbose:
                print(f"[pulado] {caminho} não existe — rode o experimento primeiro")
            return None
        return pd.read_csv(caminho)

    with plt.rc_context(RC):
        gerados.append(fig_datasets(destino))
        gerados.append(fig_espectro(destino))

        comparacao = _ler("comparacao.csv")
        if comparacao is not None:
            gerados.append(fig_curvas_treinamento(comparacao, destino))

        resumo = _ler("resumo.csv")
        if resumo is not None:
            gerados.append(fig_acuracia_vs_custo(resumo, destino))
            gerados.append(fig_custo_por_codificacao(resumo, destino))
            gerados.append(fig_fronteiras(resumo, out / "weights", destino))

        varredura = _ler("varredura_L.csv")
        if varredura is not None:
            gerados.append(fig_acuracia_vs_L(varredura, destino))

    if verbose:
        for caminho in gerados:
            print(f"escrito {caminho}")
    return gerados
