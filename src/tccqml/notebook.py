"""Apoio ao notebook `notebooks/tcc_qml.ipynb`.

O notebook não reimplementa treino nem análise: todo experimento é uma chamada
a `python -m tccqml <subcomando>` com o mesmo Python do kernel, e cada chamada
imprime o comando equivalente numa linha que começa com `$`. Este módulo guarda
o código de exibição, para que cada célula do notebook tenha uma ou duas linhas.

Nada daqui é usado pela linha de comando nem pelos experimentos. IPython,
ipywidgets e pymupdf (extra `notebooks`) só são importados dentro das funções
que exibem algo, e o módulo pode ser importado sem eles.

Uso, no notebook:

    from tccqml import notebook as nb
    nb.iniciar(rodar=False)
    nb.experimento("comparar")
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tccqml.config import PADRAO

# --------------------------------------------------------------------------
# Configuração da sessão
# --------------------------------------------------------------------------


@dataclass
class _Sessao:
    raiz: Path | None = None
    out: Path = Path(PADRAO.out)
    out_treino: Path = Path(PADRAO.out) / "notebook"
    rodar: bool = False


_SESSAO = _Sessao()


def _achar_raiz() -> Path:
    # Sobe a partir da pasta atual; fora do repo, usa a instalação editável do pacote.
    for pasta in (Path.cwd(), *Path.cwd().parents):
        if (pasta / "pyproject.toml").exists() and (pasta / "src" / "tccqml").is_dir():
            return pasta
    pasta = Path(__file__).resolve().parents[2]
    if (pasta / "pyproject.toml").exists():
        return pasta
    raise RuntimeError(
        "Não achei a raiz do repositório. Abra o notebook de dentro dele "
        "ou instale o pacote com `pip install -e .`."
    )


def iniciar(rodar: bool = False, out: str = PADRAO.out, out_treino: str | None = None) -> None:
    """Fixa a pasta de trabalho na raiz do repositório e as opções do notebook.

    `rodar=True` roda os experimentos cujos arquivos faltarem; `False` só lê os
    resultados versionados em `out`. O treino isolado grava em `out_treino`
    (padrão `<out>/notebook`), separado dos resultados oficiais.
    """
    import pennylane as qml

    _SESSAO.raiz = _achar_raiz()
    os.chdir(_SESSAO.raiz)  # `out` e os demais caminhos relativos partem da raiz
    _SESSAO.out = Path(out)
    _SESSAO.out_treino = Path(out_treino) if out_treino else Path(out) / "notebook"
    _SESSAO.rodar = bool(rodar)

    try:  # caminho relativo à raiz, para a saída não expor a pasta pessoal de quem roda
        executavel = Path(sys.executable).relative_to(_SESSAO.raiz).as_posix()
    except ValueError:
        executavel = Path(sys.executable).name
    print(f"raiz do repositório   {_SESSAO.raiz.name}/")
    print(f"resultados oficiais   {_SESSAO.out.as_posix()}/   (rodar = {_SESSAO.rodar})")
    print(f"treino isolado        {_SESSAO.out_treino.as_posix()}/")
    print(f"Python {platform.python_version()} ({executavel}) · PennyLane {qml.__version__} · "
          f"{platform.system()} {platform.release()}")


def _pasta(nome: str) -> Path:
    return _SESSAO.out / nome


# --------------------------------------------------------------------------
# Chamadas à linha de comando
# --------------------------------------------------------------------------


def _cita(arg) -> str:
    arg = str(arg)
    return f'"{arg}"' if " " in arg else arg


def _como_no_terminal(cmd: list) -> str:
    # O comando que se digitaria no terminal: `python` no lugar do caminho do executável.
    return " ".join(["python", *(_cita(a) for a in cmd[1:])])


def _executar(cmd: list) -> str:
    # Roda um processo mostrando a saída ao vivo; falha com erro claro. Usa `Popen`
    # em vez de `run` porque `run` só devolve a saída no fim, e a grade leva minutos.
    cmd = [str(a) for a in cmd]
    print(f"$ {_como_no_terminal(cmd)}", flush=True)
    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    inicio = time.perf_counter()
    linhas = []
    with subprocess.Popen(
        cmd, cwd=_SESSAO.raiz, env=ambiente, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    ) as proc:
        for linha in proc.stdout:
            print(linha, end="", flush=True)
            linhas.append(linha)
    if proc.returncode != 0:
        erro = subprocess.CalledProcessError(proc.returncode, cmd, output="".join(linhas))
        raise RuntimeError(
            f"`{_como_no_terminal(cmd)}` terminou com código {proc.returncode}. A saída "
            "completa está acima; o comando pode ser repetido no terminal, na raiz do repo."
        ) from erro
    print(f"[ok em {time.perf_counter() - inicio:.0f} s]")
    return "".join(linhas)


def _comando(*args, out=None) -> list:
    destino = Path(_SESSAO.out if out is None else out).as_posix()
    return [sys.executable, "-m", "tccqml", "--out", destino, *args]


def tccqml(*args, out=None) -> str:
    """`python -m tccqml --out <out> <args>`, com o mesmo Python do kernel."""
    return _executar(_comando(*args, out=out))


def testes() -> None:
    """Roda a suíte inteira (cerca de 2 minutos). Se ela falhar, nenhum número vale."""
    _executar([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])


def _faltando(arquivos) -> list[Path]:
    return [Path(a) for a in arquivos if not Path(a).exists()]


def _rodar_se_preciso(args, esperados, out=None) -> bool:
    # Roda o subcomando só se faltar arquivo e rodar = True. Devolve True se rodou.
    falta = _faltando(esperados)
    if not falta:
        return False
    texto = _como_no_terminal(_comando(*args, out=out))
    if not _SESSAO.rodar:
        print(f"[pulado] rodar = False e faltam {len(falta)} arquivo(s), por exemplo "
              f"{falta[0].as_posix()}.\n         Para gerá-los: $ {texto}")
        return False
    tccqml(*args, out=out)
    return True


# --------------------------------------------------------------------------
# Exibição de tabelas e figuras gravadas pela CLI
# --------------------------------------------------------------------------

TABELAS = ["tab_acuracia", "tab_custo", "tab_espectro", "tab_ablacao", "tab_diagnostico",
           "tab_mesmo_p", "tab_sensibilidade", "tab_verificacoes", "tab_qualitativa"]
FIGURAS = ["datasets.pdf", "curvas-treinamento.pdf", "fronteiras-aprendidas.pdf",
           "acuracia-vs-custo.pdf", "custo-por-codificacao.pdf", "acuracia-vs-R.pdf",
           "espectro-reuploading.pdf", "sensibilidade-eta.pdf", "limiar-circles.pdf"]


def _garantir_saidas() -> None:
    # `tabelas` e `figuras` só leem CSV (segundos): se faltar alguma saída, regenera.
    tabelas = [_pasta("tables") / f"{t}.{ext}" for t in TABELAS for ext in ("tex", "csv")]
    _rodar_se_preciso(["tabelas"], tabelas)
    _rodar_se_preciso(["figuras"], [_pasta("figures") / f for f in FIGURAS])


def _display(*objetos) -> None:
    from IPython.display import display

    display(*objetos)


def _markdown(texto: str) -> None:
    from IPython.display import Markdown

    _display(Markdown(texto))


def mostrar_figura(nome: str, dpi: int = 140) -> None:
    """Exibe `<out>/figures/<nome>`, um PDF gravado por `figuras` (ou um PNG)."""
    from IPython.display import Image

    caminho = Path(nome) if Path(nome).parent != Path(".") else _pasta("figures") / nome
    if not caminho.exists():
        _markdown(f"> ⚠️ **`{caminho.as_posix()}` ainda não existe.** Rode o experimento "
                  f"correspondente e depois `python -m tccqml --out {_SESSAO.out.as_posix()} figuras`.")
        return
    _markdown(f"**`{caminho.name}`**")
    if caminho.suffix.lower() != ".pdf":
        _display(Image(filename=str(caminho)))
        return
    try:
        import pymupdf
    except ImportError:
        _markdown(f"> O PDF existe (`{caminho.as_posix()}`), mas exibi-lo aqui requer o "
                  "`pymupdf` (extra `notebooks`).")
        return
    with pymupdf.open(caminho) as documento:
        _display(Image(data=documento[0].get_pixmap(dpi=dpi).tobytes("png")))


_LATEX = [("$\\pm$", "±"), ("{,}", ","), ("\\,", " "), ("\\omega", "ω"), ("\\Omega", "Ω"),
          ("\\eta", "η"), ("_{\\max}", "_max"), ("_{\\mathrm{ZZ}}", "_ZZ"), ("\\text{var}", "var"),
          ("\\dots", "…"), ("\\lceil", "⌈"), ("\\rceil", "⌉"), ("\\log_2", "log₂"),
          ("\\{", "{"), ("\\}", "}"), ("--", "—")]


def sem_latex(valor):
    """Tira a marcação LaTeX de uma célula de tabela, para exibir no notebook."""
    if not isinstance(valor, str):
        return valor
    for de, para in _LATEX:
        valor = valor.replace(de, para)
    valor = re.sub(r"\\(?:textit|textbf|emph|texttt|boldsymbol)\{([^{}]*)\}", r"\1", valor)
    return valor.replace("$", "")


def _exibir_df(df: pd.DataFrame, uuid: str) -> None:
    # set_uuid: id fixo no HTML, para a cópia executada não mudar a cada geração.
    _display(df.style.set_uuid(uuid).hide(axis="index"))


def mostrar_tabela(nome: str) -> None:
    """Exibe `<out>/tables/<nome>.csv`, a mesma tabela do `.tex`, sem a marcação LaTeX."""
    caminho = _pasta("tables") / f"{nome}.csv"
    if not caminho.exists():
        print(f"⚠️ {caminho.as_posix()} não existe: rode o experimento e `python -m tccqml tabelas`.")
        return
    tabela = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    _markdown(f"**`{nome}`** (a mesma tabela de `{caminho.with_suffix('.tex').as_posix()}`)")
    _exibir_df(tabela.map(sem_latex).rename(columns=sem_latex), nome)


def ler(nome: str) -> pd.DataFrame | None:
    """Lê um CSV de `<out>/metrics`, ou avisa que ele ainda não existe."""
    caminho = _pasta("metrics") / nome
    if not caminho.exists():
        print(f"⚠️ {caminho.as_posix()} não existe: rode o experimento correspondente.")
        return None
    return pd.read_csv(caminho)


# --------------------------------------------------------------------------
# Os experimentos do TCC, um por subcomando
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Experimento:
    """Um subcomando da CLI e o que o notebook mostra dele."""

    args: tuple[str, ...]
    gravados: tuple[str, ...]  # relativos a <out>; o tempo de treino sai do primeiro
    tabelas: tuple[str, ...] = ()
    figuras: tuple[str, ...] = ()
    extra: object = field(default=None, compare=False)  # função sem argumentos, opcional


def _extra_comparar() -> None:
    por_treino = ler("por_treino.csv")
    if por_treino is not None:
        print(f"{por_treino['encoding'].nunique()} codificações × {por_treino['dataset'].nunique()} "
              f"conjuntos × {por_treino['seed'].nunique()} sementes")


def _extra_varredura_R() -> None:
    varredura = ler("varredura_R.csv")
    if varredura is not None:
        _markdown("**Custo por $R$** (é o mesmo nos três conjuntos)")
        custo = varredura.groupby("R")[["n_params_circuito", "depth", "gates_2q", "n_shots_passo"]]
        _exibir_df(custo.first().reset_index(), "custo_por_R")


def _extra_espectro() -> None:
    espectro = ler("espectro.csv")
    if espectro is not None:
        _markdown("**`espectro.csv`** (com as colunas `dentro_do_limite` e `atinge_limite`)")
        _exibir_df(espectro, "espectro_csv")


_POR_SEMENTE = tuple(
    f"weights/{e}_{d}_{s}.npz"
    for e in PADRAO.encodings_grade for d in PADRAO.datasets_grade for s in PADRAO.sementes
)

EXPERIMENTOS: dict[str, Experimento] = {
    "comparar": Experimento(
        ("comparar",),
        ("metrics/por_treino.csv", "metrics/comparacao.csv", "metrics/resumo.csv", *_POR_SEMENTE),
        tabelas=("tab_acuracia", "tab_custo"),
        figuras=("fronteiras-aprendidas.pdf", "curvas-treinamento.pdf",
                 "acuracia-vs-custo.pdf", "custo-por-codificacao.pdf"),
        extra=_extra_comparar,
    ),
    "sensibilidade": Experimento(
        ("sensibilidade",),
        ("metrics/sensibilidade_eta.csv", "metrics/resumo_sensibilidade_eta.csv"),
        tabelas=("tab_sensibilidade",),
        figuras=("sensibilidade-eta.pdf",),
    ),
    "varredura_R": Experimento(
        ("varredura", "--parametro", "R"),
        ("metrics/varredura_R.csv",),
        figuras=("acuracia-vs-R.pdf",),
        extra=_extra_varredura_R,
    ),
    "varredura_L_var": Experimento(
        ("varredura", "--parametro", "L_var"),
        ("metrics/varredura_L_var.csv",),
        tabelas=("tab_mesmo_p",),
    ),
    "espectro": Experimento(
        ("espectro",),
        ("metrics/espectro.csv",),
        tabelas=("tab_espectro",),
        figuras=("espectro-reuploading.pdf",),
        extra=_extra_espectro,
    ),
    "ablacao": Experimento(
        ("ablacao",), ("metrics/ablacao_entrelacamento.csv",), tabelas=("tab_ablacao",)
    ),
    "diagnostico": Experimento(
        ("diagnostico",), ("metrics/diagnostico_amplitude.csv",), tabelas=("tab_diagnostico",)
    ),
    "verificacoes": Experimento(
        ("verificacoes",),
        ("metrics/controle_linear.csv", "metrics/resumo_controle_linear.csv",
         "metrics/limiar_circles.csv", "metrics/resumo_limiar_circles.csv"),
        tabelas=("tab_verificacoes",),
        figuras=("limiar-circles.pdf",),
    ),
}


def experimento(nome: str) -> None:
    """Roda o experimento `nome` se preciso e mostra as tabelas e figuras dele.

    Só roda se faltar algum arquivo que ele grava e se `rodar=True`; nesse caso
    regenera também as tabelas e as figuras com os CSVs novos.
    """
    exp = EXPERIMENTOS[nome]
    if _rodar_se_preciso(exp.args, [_SESSAO.out / g for g in exp.gravados]):
        print("\nAtualizando tabelas e figuras com os CSVs novos:")
        tccqml("tabelas")
        tccqml("figuras")
    _garantir_saidas()

    principal = _SESSAO.out / exp.gravados[0]
    if principal.exists():
        df = pd.read_csv(principal)
        if "segundos" in df:
            print(f"{len(df)} treinos, {df['segundos'].sum() / 60:.1f} min de treino somados "
                  "(coluna `segundos`)")
    if exp.extra is not None:
        exp.extra()
    for tabela in exp.tabelas:
        mostrar_tabela(tabela)
    for figura in exp.figuras:
        mostrar_figura(figura)


ONDE_NO_TCC = {
    "tab_acuracia": "Cap. 4 › Comparação principal (tab:acuracia)",
    "tab_custo": "Cap. 4 › Verificações dos circuitos e desempenho × recursos (tab:custo)",
    "tab_espectro": "Cap. 4 › Verificações dos circuitos: espectro medido (tab:espectro)",
    "tab_ablacao": "Cap. 4 › Controles: ablação do entrelaçamento (tab:ablacao)",
    "tab_diagnostico": "Cap. 4 › Controles: normalização no amplitude (tab:diagnostico)",
    "tab_mesmo_p": "Cap. 4 › Controles: varredura de R com o mesmo p (tab:mesmo-p)",
    "tab_sensibilidade": "Cap. 4 › Controles: sensibilidade à taxa de aprendizado (tab:sensibilidade)",
    "tab_verificacoes": "Cap. 4 › Controles: verificações independentes do treino (tab:verificacoes)",
    "tab_qualitativa": "opcional no Cap. 4 (o texto vive em tabelas.QUALITATIVA)",
    "datasets.pdf": "Cap. 3 › Conjuntos de dados (sugestão: fig:datasets)",
    "fronteiras-aprendidas.pdf": "Cap. 4 › Comparação principal, semente 42 (sugestão: fig:fronteiras)",
    "curvas-treinamento.pdf": "Cap. 4 › Comparação principal (sugestão: fig:curvas)",
    "acuracia-vs-custo.pdf": "Cap. 4 › Desempenho × recursos (sugestão: fig:acc-custo)",
    "custo-por-codificacao.pdf": "opcional; repete a tab:custo em barras",
    "acuracia-vs-R.pdf": "Cap. 4 › Controles: varredura de R (sugestão: fig:acc-R)",
    "espectro-reuploading.pdf": "Cap. 4 › Verificações dos circuitos: espectro (sugestão: fig:espectro)",
    "sensibilidade-eta.pdf": "Cap. 4 › Controles: sensibilidade a η (sugestão: fig:sensibilidade)",
    "limiar-circles.pdf": "Cap. 4 › Controles: limiar de g no circles, semente 42 (sugestão: fig:limiar)",
}


def indice() -> None:
    """Cada tabela e figura gravada, se existe e onde entra no TCC."""
    _garantir_saidas()
    linhas = [
        {"arquivo": f"tables/{t}.tex", "existe": (_pasta("tables") / f"{t}.tex").exists(),
         "no TCC": ONDE_NO_TCC[t]}
        for t in TABELAS
    ] + [
        {"arquivo": f"figures/{f}", "existe": (_pasta("figures") / f).exists(),
         "no TCC": ONDE_NO_TCC[f]}
        for f in FIGURAS
    ]
    _exibir_df(pd.DataFrame(linhas), "indice_tcc")


# --------------------------------------------------------------------------
# Seção 1: os dados
# --------------------------------------------------------------------------


def bruto_vs_normalizado(dataset: str = "moons") -> None:
    """Os pontos do gerador e o treino depois do `MinMaxScaler`, lado a lado."""
    import matplotlib.pyplot as plt

    from tccqml.data import _GENERATORS, load_dataset

    # O mesmo gerador que `load_dataset` usa, antes da divisão e da normalização.
    X_bruto, y_bruto = _GENERATORS[dataset](PADRAO.n_samples, PADRAO.noise, PADRAO.seed)
    ds = load_dataset(dataset, seed=PADRAO.seed)

    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    for classe, marcador in ((0, "o"), (1, "^")):
        eixos[0].scatter(*X_bruto[y_bruto == classe].T, marker=marcador, s=22, alpha=0.8,
                         label=f"classe {classe}")
        eixos[1].scatter(*ds.X_train[ds.y_train == classe].T, marker=marcador, s=22, alpha=0.8)
    eixos[0].set_title(f"bruto ({len(X_bruto)} amostras)")
    eixos[0].legend(loc="lower left", fontsize=8)
    for v in (0, np.pi):
        eixos[1].axhline(v, ls=":", lw=1, color="gray")
        eixos[1].axvline(v, ls=":", lw=1, color="gray")
    eixos[1].set_title(rf"treino normalizado para $[0, \pi]$ ({len(ds.X_train)} amostras)")
    eixos[0].set(xlabel="$u_1$", ylabel="$u_2$")
    eixos[1].set(xlabel="$x_1$", ylabel="$x_2$")
    fig.suptitle(f"{dataset}: cada eixo vai para [0, π] separadamente")
    fig.tight_layout()
    plt.show()


def particoes() -> pd.DataFrame:
    """Tamanho, fração da classe 1 e mínimo/máximo de cada partição, por conjunto."""
    from tccqml.data import load_dataset

    linhas = []
    for nome in PADRAO.datasets_grade:
        d = load_dataset(nome, seed=PADRAO.seed)
        linha = {"conjunto": nome}
        for parte, X, y in (("treino", d.X_train, d.y_train), ("val", d.X_val, d.y_val),
                            ("teste", d.X_test, d.y_test)):
            linha[f"n_{parte}"] = len(X)
            linha[f"frac1_{parte}"] = round(float(y.mean()), 3)
            linha[f"min_{parte}"] = round(float(X.min()), 3)
            linha[f"max_{parte}"] = round(float(X.max()), 3)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("conjunto")


# --------------------------------------------------------------------------
# Seção 2: anatomia do classificador
# --------------------------------------------------------------------------

X_EXEMPLO = (0.5, 2.0)


def _montar(encoding, L_var=PADRAO.L_var, ansatz=PADRAO.ansatz, R=PADRAO.R):
    # Mesma chamada de `python -m tccqml treinar` (cli._treinar).
    from tccqml import model

    enc_kwargs = {"R": R} if encoding == "reuploading" else None
    return model.build(encoding, n_features=2, L_var=L_var, ansatz=ansatz,
                       enc_kwargs=enc_kwargs)


def _estado_codificado(clf, x=X_EXEMPLO) -> np.ndarray:
    # psi(x): só o bloco de dados, sem camadas treináveis (no re-uploading, os R blocos S(x)).
    import pennylane as qml
    from pennylane import numpy as pnp

    from tccqml.embeddings import get_encoding

    enc = get_encoding(clf.encoding, **(clf.enc_kwargs or {}))

    @qml.qnode(qml.device("default.qubit", wires=clf.n_qubits))
    def so_codificacao(x):
        enc.bloco_de_dados()(x, wires=range(clf.n_qubits))
        return qml.state()

    return np.asarray(so_codificacao(pnp.array(x, requires_grad=False)))


def anatomia(encoding: str = "angle", R: int = PADRAO.R) -> None:
    """Custo, circuito e estado psi(x) de uma codificação, com os pesos iniciais."""
    import pennylane as qml
    from pennylane import numpy as pnp

    from tccqml import model
    from tccqml.circuit_stats import formatar, stats

    clf = _montar(encoding, R=R)
    weights, alpha, _ = model.pesos_iniciais(clf, seed=PADRAO.seed)
    x = pnp.array(X_EXEMPLO, requires_grad=False)

    print(f"codificação: {encoding}   qubits: {clf.n_qubits}   p (circuito): "
          f"{clf.n_params_circuito}   treináveis com o viés: {clf.n_params}\n")
    print(formatar(stats(clf, list(X_EXEMPLO))))
    print("\ncircuito em blocos:")
    print(qml.draw(clf.circuit, max_length=120, show_matrices=False)(x, weights, alpha))
    print('\ncircuito no nível do dispositivo (level="device"):')
    print(qml.draw(clf.circuit, level="device", max_length=140)(x, weights, alpha))

    psi = _estado_codificado(clf)
    print(f"\npsi(x) do bloco de dados, x = {list(X_EXEMPLO)}   "
          f"(norma {np.sum(np.abs(psi) ** 2):.8f})")
    for i, amp in enumerate(psi):
        base = format(i, f"0{clf.n_qubits}b")
        print(f"  |{base}>  amplitude {amp.real:+.4f}{amp.imag:+.4f}i   "
              f"probabilidade {abs(amp) ** 2:.4f}")
    if clf.n_qubits == 2:
        # Concorrência de um estado puro de 2 qubits: 0 para estado produto, 1 para
        # maximamente entrelaçado.
        concorrencia = 2 * abs(psi[0] * psi[3] - psi[1] * psi[2])
        tipo = "produto (sem entrelaçamento)" if concorrencia < 1e-9 else "ENTRELAÇADO"
        print(f"  concorrência C = 2|a00 a11 - a01 a10| = {concorrencia:.4f}  ->  estado {tipo}")
    else:
        print("  um só qubit: não há par para entrelaçar")


# --------------------------------------------------------------------------
# Seção 3: um treino isolado
# --------------------------------------------------------------------------


def _args_treinar(encoding, dataset, ansatz, R, L_var, epocas, eta, seed) -> list[str]:
    args = ["treinar", "--encoding", encoding, "--dataset", dataset, "--ansatz", ansatz]
    if encoding == "reuploading":
        args += ["--R", R]
    args += ["--L-var", L_var, "--epocas", epocas, "--eta", eta, "--seed", seed, "--salvar"]
    return [str(a) for a in args]


def treinar(
    encoding: str = "angle",
    dataset: str = "moons",
    *,
    ansatz: str = PADRAO.ansatz,
    R: int = PADRAO.R,
    L_var: int = PADRAO.L_var,
    epocas: int = PADRAO.epocas,
    eta: float = PADRAO.eta,
    seed: int = PADRAO.seed,
) -> None:
    """Roda `python -m tccqml treinar ... --salvar` no `out_treino` e mostra o resultado.

    Treina sempre, com qualquer valor de `rodar`: é uma ação explícita e leva
    segundos. Mostra o custo, o circuito com os pesos da época escolhida, as
    curvas, a fronteira de decisão e a comparação com a grade oficial.
    """
    from tccqml.cli import nome_treino

    cfg = {"encoding": encoding, "dataset": dataset, "ansatz": ansatz, "R": R,
           "L_var": L_var, "epocas": epocas, "eta": eta, "seed": seed}
    tccqml(*_args_treinar(**cfg), out=_SESSAO.out_treino)
    _mostrar_treino(nome_treino(SimpleNamespace(**cfg, n_samples=PADRAO.n_samples)), **cfg)


def _mostrar_treino(nome, encoding, dataset, ansatz, R, L_var, epocas, eta, seed) -> None:
    import matplotlib.pyplot as plt
    import pennylane as qml
    from pennylane import numpy as pnp

    from tccqml import model
    from tccqml.data import load_dataset

    base = _SESSAO.out_treino
    hist = pd.read_csv(base / "metrics" / f"{nome}.csv")
    res = pd.read_csv(base / "metrics" / f"{nome}_resumo.csv").iloc[0]
    pesos = np.load(base / "weights" / f"{nome}.npz")
    alpha = pesos["alpha"] if "alpha" in pesos.files else None
    weights, bias = pesos["weights"], float(pesos["bias"])
    clf = _montar(encoding, L_var=L_var, ansatz=ansatz, R=R)
    ds = load_dataset(dataset, seed=seed)
    escolhida = int(res["epoca_escolhida"])

    _markdown(f"#### Resultado: `{nome}`")
    previsto = model.prever(clf, weights, alpha, bias, ds.X_test)
    print(f"conferência: acurácia de teste recalculada dos pesos gravados = "
          f"{np.mean(np.asarray(previsto) == ds.y_test):.3f}")
    print("\ncircuito com os pesos da época escolhida, no ponto de exemplo (nível do dispositivo):")
    x = pnp.array(X_EXEMPLO, requires_grad=False)
    print(qml.draw(clf.circuit, level="device", max_length=140)(x, weights, alpha))

    fig, eixos = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    eixos[0].plot(hist["epoca"], hist["custo"], marker="o", ms=3)
    eixos[0].set(xlabel="época", ylabel="custo quadrático de treino", title="Convergência")
    eixos[1].plot(hist["epoca"], hist["acc_treino"], marker="o", ms=3, label="treino")
    eixos[1].plot(hist["epoca"], hist["acc_val"], marker="s", ms=3, label="validação")
    eixos[1].axhline(0.5, ls="--", lw=1, color="gray", label="acaso (0,5)")
    eixos[1].set(xlabel="época", ylabel="acurácia", ylim=(0.4, 1.02),
                 title="Acurácia (o teste não entra aqui)")
    for eixo in eixos[:2]:
        eixo.axvline(escolhida, ls=":", lw=1.2, color="k", label="época escolhida")
    eixos[1].legend(loc="lower right", fontsize=8)

    grade = np.linspace(-0.1, np.pi + 0.1, 60)
    G1, G2 = np.meshgrid(grade, grade)
    pontos = np.column_stack([G1.ravel(), G2.ravel()])
    z = np.asarray(model.saida_continua(clf, weights, alpha, bias, pontos), dtype=float)
    mapa = eixos[2].contourf(G1, G2, z.reshape(G1.shape), levels=np.linspace(-1, 1, 21),
                             cmap="RdBu_r", alpha=0.6, extend="both")
    eixos[2].contour(G1, G2, z.reshape(G1.shape), levels=[0.0], colors="k", linewidths=1.5)
    for classe, marcador, cor in ((0, "o", "#1b1b1b"), (1, "^", "#7a0f16")):
        m = ds.y_test == classe
        eixos[2].scatter(ds.X_test[m, 0], ds.X_test[m, 1], marker=marcador, c=cor, s=34,
                         edgecolors="white", linewidths=0.6, label=f"classe {classe} (teste)")
    eixos[2].set(xlabel="$x_1$", ylabel="$x_2$", title="Fronteira aprendida")
    eixos[2].legend(loc="upper right", fontsize=8)
    fig.colorbar(mapa, ax=eixos[2], label=r"$\langle Z_0\rangle + b$  (negativo: classe 0)")
    fig.suptitle(f"{encoding} · {dataset} · semente {seed}")
    fig.tight_layout()
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"{nome}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"figura salva em {(figs / f'{nome}.png').as_posix()}")

    _comparar_com_a_grade(res, escolhida, encoding, dataset, ansatz, R, L_var, epocas,
                          eta, seed)


def _comparar_com_a_grade(res, escolhida, encoding, dataset, ansatz, R, L_var, epocas,
                          eta, seed) -> None:
    # Esta semente foi típica? Compara com a média e o desvio da grade oficial.
    resumo = ler("resumo.csv")
    if resumo is None:
        return
    linha = resumo[(resumo["encoding"] == encoding) & (resumo["dataset"] == dataset)]
    if linha.empty:
        print(f"\n{encoding} × {dataset} não faz parte da grade.")
        return
    linha = linha.iloc[0]
    media, desvio = linha["acc_teste_mean"], linha["acc_teste_std"]
    print(f"\ngrade oficial (`comparar`, {int(linha['acc_teste_count'])} sementes): "
          f"teste {media:.3f} ± {desvio:.3f}")
    distancia = (res["acc_teste"] - media) / desvio if desvio > 0 else 0.0
    print(f"este treino: {res['acc_teste']:.3f}, a {distancia:+.1f} desvio(s) da média "
          f"({'típico' if abs(distancia) <= 1 else 'fora de 1 desvio'})")

    protocolo = (ansatz == PADRAO.ansatz and L_var == PADRAO.L_var and eta == PADRAO.eta
                 and epocas == PADRAO.epocas
                 and (encoding != "reuploading" or R == PADRAO.R))
    if not protocolo:
        print("atenção: esta configuração foge do protocolo da grade; a comparação é só indicativa.")
        return
    por_treino = ler("por_treino.csv") if seed in PADRAO.sementes else None
    if por_treino is None:
        return
    mesma = por_treino[(por_treino["encoding"] == encoding) & (por_treino["dataset"] == dataset)
                       & (por_treino["seed"] == seed)]
    if len(mesma):
        g = mesma.iloc[0]
        igual = all(np.isclose(g[c], res[c]) for c in ("acc_treino", "acc_val", "acc_teste")) \
            and int(g["epoca_escolhida"]) == escolhida
        print(f"mesma semente na grade: treino {g['acc_treino']:.3f}, val {g['acc_val']:.3f}, "
              f"teste {g['acc_teste']:.3f}, época {int(g['epoca_escolhida'])} -> "
              f"{'idêntico ao terminal ✓' if igual else 'DIFERENTE ✗ (investigar)'}")


# --------------------------------------------------------------------------
# Painéis opcionais (ipywidgets; só no Jupyter ou no VS Code)
# --------------------------------------------------------------------------


def painel_anatomia() -> None:
    """Seletor de codificação que redesenha a `anatomia`."""
    import ipywidgets as widgets

    seletor = widgets.ToggleButtons(options=list(PADRAO.encodings_grade), value="angle",
                                    description="codificação")
    saida = widgets.Output()

    def _redesenhar(_=None):
        saida.clear_output(wait=True)
        with saida:
            anatomia(seletor.value)

    seletor.observe(_redesenhar, names="value")
    _redesenhar()
    _display(widgets.VBox([seletor, saida]))


def painel_treino() -> None:
    """Painel com as opções de `treinar` e um botão; mostra o comando equivalente."""
    import ipywidgets as widgets

    from tccqml.ansatz import ANSATZE
    from tccqml.data import _GENERATORS
    from tccqml.embeddings import ENCODINGS

    estilo = {"description_width": "70px"}
    campos = {
        "encoding": widgets.Dropdown(options=sorted(ENCODINGS), value="angle",
                                     description="encoding", style=estilo),
        "dataset": widgets.Dropdown(options=sorted(_GENERATORS), value="moons",
                                    description="dataset", style=estilo),
        "ansatz": widgets.Dropdown(options=sorted(ANSATZE), value=PADRAO.ansatz,
                                   description="ansatz", style=estilo),
        "R": widgets.BoundedIntText(value=PADRAO.R, min=1, max=10,
                                         description="R (R)", style=estilo),
        "L_var": widgets.BoundedIntText(value=PADRAO.L_var, min=1, max=10,
                                           description="L_var", style=estilo),
        "epocas": widgets.BoundedIntText(value=PADRAO.epocas, min=1, max=500,
                                         description="epocas", style=estilo),
        "eta": widgets.BoundedFloatText(value=PADRAO.eta, min=1e-4, max=10.0, step=0.01,
                                       description="eta", style=estilo),
        "seed": widgets.IntText(value=PADRAO.seed, description="seed", style=estilo),
    }
    botao = widgets.Button(description="Treinar", button_style="primary", icon="play")
    comando_html = widgets.HTML()
    saida = widgets.Output()

    def _cfg():
        return {chave: campo.value for chave, campo in campos.items()}

    def _atualizar_comando(_=None):
        campos["R"].disabled = campos["encoding"].value != "reuploading"
        texto = _como_no_terminal(_comando(*_args_treinar(**_cfg()), out=_SESSAO.out_treino))
        comando_html.value = f"comando equivalente: <code>{texto}</code>"

    def _treinar(_):
        botao.disabled = True
        saida.clear_output()
        try:
            with saida:
                cfg = _cfg()
                treinar(cfg.pop("encoding"), cfg.pop("dataset"), **cfg)
        finally:
            botao.disabled = False

    for campo in campos.values():
        campo.observe(_atualizar_comando, names="value")
    botao.on_click(_treinar)
    _atualizar_comando()
    _display(widgets.VBox([
        widgets.HBox([campos["encoding"], campos["dataset"], campos["ansatz"]]),
        widgets.HBox([campos["R"], campos["L_var"], campos["epocas"]]),
        widgets.HBox([campos["eta"], campos["seed"], botao]),
        comando_html,
        saida,
    ]))
