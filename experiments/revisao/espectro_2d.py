"""Espectro 2D para k e R = 1..5, na inicialização x3 e nos modelos treinados.

Mesma medição de `tccqml.espectro.espectro_2d` (grade uniforme 32 x 32 em
[0, 2pi)^2, FFT 2D dividida por 32^2, coeficiente "não nulo" se |c_w| > 1e-6,
`PADRAO.espectro_limiar`). A função avaliada é a saída do circuito f_theta(x) =
<Z_0>, SEM o viés b: o viés só somaria uma constante ao coeficiente c_(0,0).

Configurações: angle com L_var = k e re-uploading com R = k, k = 1..5.
    - inicialização x3: parâmetros iniciais da semente 42 multiplicados por 3,
      como na tab:espectro (mostra o suporte que a arquitetura alcança);
    - treinados: parâmetros da época escolhida, gravados por
      `historicos_varredura.py`, nos três conjuntos e nas cinco sementes.

Três contagens por configuração, separadas de propósito:
    permitidas   |Omega|^2 da tab:espectros: 9 no angle, (2R + 1)^2 no re-uploading;
    alcançáveis  termos acima de 1e-6 com a inicialização x3 (suporte da arquitetura);
    usadas       termos acima de 1e-6 após o treino e, como leitura complementar,
                 termos com |c_w| >= 1% do maior |c_w| com w != (0, 0) (limiar relativo).

Saídas (results/revisao/):
    metrics/espectro_2d.csv              contagens por configuração e modelo
    metrics/resumo_espectro_2d.csv       permitidas / alcançáveis / usadas (média nas sementes)
    metrics/coeficientes_espectro_2d.csv |c_w| de cada mapa da figura
    figures/espectro-2d.pdf              mapas de |c_w|, escala log
    tables/tab_espectro_2d.tex

Uso: python experiments/revisao/espectro_2d.py
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
from _comum import (
    FIGURES,
    METRICS,
    TABLES,
    WEIGHTS,
    dec,
    gravar_csv,
    registrar,
    salvar_figura,
)
from matplotlib.colors import LogNorm
from pennylane import numpy as pnp

from tccqml import model
from tccqml.config import PADRAO
from tccqml.espectro import espectro_2d
from tccqml.figuras import NOMES, RC
from tccqml.model import pesos_iniciais

N = PADRAO.espectro_n_pontos_2d
LIMIAR = PADRAO.espectro_limiar
LIMIAR_RELATIVO = 0.01
KS = (1, 2, 3, 4, 5)
W_FIG = 6  # |omega| máximo mostrado nos mapas


def construir(enc: str, k: int):
    if enc == "angle":
        return model.build("angle", n_features=2, L_var=k)
    return model.build("reuploading", n_features=2, enc_kwargs={"R": k})


def coeficientes(clf, w) -> np.ndarray:
    """|c_w| em [-15, 15]^2 (fftshift), mesma grade e normalização de `espectro_2d`."""
    eixo = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    g1, g2 = np.meshgrid(eixo, eixo, indexing="ij")
    X = np.column_stack([g1.ravel(), g2.ravel()])
    f = np.asarray(clf.circuit(pnp.array(X, requires_grad=False), w, None), dtype=float)
    return np.fft.fftshift(np.abs(np.fft.fft2(f.reshape(N, N))) / N**2)


def contar(c: np.ndarray) -> dict:
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / N)).astype(int)
    w1, w2 = np.meshgrid(freqs, freqs, indexing="ij")
    centro = (w1 == 0) & (w2 == 0)
    acima = c > LIMIAR
    maior = c[~centro].max()
    relativo = c >= LIMIAR_RELATIVO * maior
    return {
        "n_termos": int(acima.sum()),
        "n_cruzados": int((acima & (w1 != 0) & (w2 != 0)).sum()),
        "omega_max_x1": int(np.abs(w1[acima]).max()) if acima.any() else 0,
        "omega_max_x2": int(np.abs(w2[acima]).max()) if acima.any() else 0,
        "n_termos_rel1pct": int(relativo.sum()),
        "c_max_sem_00": float(maior),
        "c_00": float(c[centro][0]),
    }


def main() -> None:
    linhas, mapas = [], {}
    for enc in ("angle", "reuploading"):
        for k in KS:
            clf = construir(enc, k)
            permitidas = 9 if enc == "angle" else (2 * k + 1) ** 2
            w0, _, _ = pesos_iniciais(clf, seed=PADRAO.seed)
            c = coeficientes(clf, w0 * 3.0)
            # Conferência com a função do núcleo: o mesmo conjunto de termos.
            assert contar(c)["n_termos"] == espectro_2d(clf, w0 * 3.0).n_termos
            mapas[(enc, k, "init")] = c
            linhas.append({"encoding": enc, "k": k, "modelo": "inicializacao_x3", "dataset": "",
                           "seed": PADRAO.seed, "n_permitidas": permitidas, **contar(c)})
            for dataset in PADRAO.datasets_grade:
                for seed in PADRAO.sementes:
                    pesos = np.load(WEIGHTS / "varredura" / f"{enc}_k{k}_{dataset}_{seed}.npz")
                    c = coeficientes(clf, pnp.array(pesos["weights"], requires_grad=False))
                    if dataset == "moons" and seed == PADRAO.seed:
                        mapas[(enc, k, "treinado")] = c
                    linhas.append({"encoding": enc, "k": k, "modelo": "treinado", "dataset": dataset,
                                   "seed": seed, "n_permitidas": permitidas, **contar(c)})
    df = pd.DataFrame(linhas)

    resumo = []
    for (enc, k), g in df.groupby(["encoding", "k"], sort=False):
        ini = g[g["modelo"] == "inicializacao_x3"].iloc[0]
        tr = g[g["modelo"] == "treinado"]
        resumo.append({
            "encoding": enc, "k": k, "p": 6 * k, "n_permitidas": ini["n_permitidas"],
            "n_alcancaveis": ini["n_termos"], "omega_max_x1": ini["omega_max_x1"],
            "omega_max_x2": ini["omega_max_x2"],
            "n_usadas_min": tr["n_termos"].min(), "n_usadas_max": tr["n_termos"].max(),
            "n_usadas_rel1pct_mean": tr["n_termos_rel1pct"].mean(),
            "n_usadas_rel1pct_min": tr["n_termos_rel1pct"].min(),
            "n_usadas_rel1pct_max": tr["n_termos_rel1pct"].max(),
            "n_usadas_rel1pct_mean_moons": tr[tr["dataset"] == "moons"]["n_termos_rel1pct"].mean(),
            "fora_do_permitido": bool(max(tr["omega_max_x1"].max(), tr["omega_max_x2"].max())
                                      > (1 if enc == "angle" else k)),
        })
    resumo = pd.DataFrame(resumo)

    # Mapas da figura em formato longo, para conferência.
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / N)).astype(int)
    longos = []
    for (enc, k, modelo), c in mapas.items():
        for i, a in enumerate(freqs):
            for j, b in enumerate(freqs):
                if abs(a) <= W_FIG and abs(b) <= W_FIG:
                    longos.append({"encoding": enc, "k": k, "modelo": modelo, "omega_1": a,
                                   "omega_2": b, "abs_c": c[i, j]})

    # Figura: linhas = (angle, re-uploading) x (inicialização, treinado), colunas = k.
    sel = (freqs >= -W_FIG) & (freqs <= W_FIG)
    norma = LogNorm(vmin=LIMIAR, vmax=1.0)
    linhas_fig = [("angle", "init"), ("reuploading", "init"), ("angle", "treinado"), ("reuploading", "treinado")]
    rot_linha = {"init": "inic. $\\times 3$", "treinado": "treinado (moons, 42)"}
    with plt.rc_context(RC | {"axes.grid": False, "figure.constrained_layout.use": True}):
        fig, eixos = plt.subplots(len(linhas_fig), len(KS), figsize=(6.5, 6.0), sharex=True, sharey=True)
        for i, (enc, modelo) in enumerate(linhas_fig):
            for j, k in enumerate(KS):
                eixo = eixos[i][j]
                c = mapas[(enc, k, modelo)][np.ix_(sel, sel)]
                im = eixo.imshow(np.ma.masked_less_equal(c.T, LIMIAR), origin="lower", norm=norma,
                                 cmap="viridis", extent=(-W_FIG - 0.5, W_FIG + 0.5, -W_FIG - 0.5, W_FIG + 0.5))
                lim = 1 if enc == "angle" else k
                eixo.add_patch(plt.Rectangle((-lim - 0.5, -lim - 0.5), 2 * lim + 1, 2 * lim + 1,
                                             fill=False, edgecolor="#c44e52", linewidth=0.9, linestyle="--"))
                eixo.set_xticks([-5, 0, 5])
                eixo.set_yticks([-5, 0, 5])
                if i == 0:
                    eixo.set_title(f"$k = {k}$")
                if j == 0:
                    eixo.set_ylabel(f"{NOMES[enc]}\n{rot_linha[modelo]}\n$\\omega_2$", fontsize=8)
                if i == len(linhas_fig) - 1:
                    eixo.set_xlabel("$\\omega_1$")
        barra = fig.colorbar(im, ax=eixos, shrink=0.6, location="right")
        barra.set_label("$|c_\\omega|$ (branco: $\\leq 10^{-6}$)")
        caminho_fig = salvar_figura(fig, FIGURES / "espectro-2d.pdf")

    corpo = []
    for r in resumo.itertuples():
        if r.encoding == "reuploading" and r.k == 1:
            corpo.append("\\midrule")
        nome = NOMES[r.encoding] if r.k == 1 else ""
        nome = f"\\textit{{{nome}}}" if nome else ""
        usadas = (f"{r.n_usadas_min}" if r.n_usadas_min == r.n_usadas_max
                  else f"{r.n_usadas_min}--{r.n_usadas_max}")
        corpo.append(
            f"{nome} & {r.k} & {r.p} & {r.n_permitidas} & {r.n_alcancaveis} & "
            f"{r.omega_max_x1}; {r.omega_max_x2} & {usadas} & "
            f"{dec(r.n_usadas_rel1pct_mean, 1)} ({r.n_usadas_rel1pct_min}--{r.n_usadas_rel1pct_max}) \\\\"
        )
    tex = "\n".join([
        "\\begin{table}[htb]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        ("\\caption[Frequências permitidas, alcançáveis e usadas]{Termos $(\\omega_1, \\omega_2)$ "
        "do espectro de $f_\\theta$: permitidos pela codificação (Tabela~\\ref{tab:espectros}), "
        "alcançados pela arquitetura (inicialização $\\times 3$) e presentes nos modelos treinados.}"),
        "\\label{tab:espectro-2d}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        " & & & & & & \\multicolumn{2}{c}{Treinados (15 modelos)} \\\\",
        "\\cmidrule(lr){7-8}",
        ("Codificação & $k$ & $p$ & Permitidos & Alcançáveis & $\\omega_{\\max}$ ($x_1$; $x_2$) & "
        "$> 10^{-6}$ & $\\geq 1\\%$ do máx. \\\\"),
        "\\midrule",
        *corpo,
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[2pt]",
        ("{\\footnotesize $k$: $L_\\text{var}$ no \\textit{angle} e $R$ no \\textit{re-uploading}. "
        "FFT 2D de $f_\\theta = \\expval{Z_0}$, sem o viés $b$, numa grade de $32 \\times 32$ pontos "
        "em $[0, 2\\pi)^2$, dividida por $32^2$; um termo conta como presente se $|c_\\omega| > 10^{-6}$. "
        "Alcançáveis: parâmetros iniciais da semente 42 multiplicados por 3. Treinados: três "
        "conjuntos $\\times$ cinco sementes, parâmetros da época escolhida; última coluna: média "
        "(mínimo--máximo) dos termos com $|c_\\omega|$ de pelo menos 1\\% do maior coeficiente "
        "com $\\omega \\neq (0, 0)$.}"),
        "\\\\[2pt]",
        "{\\footnotesize Fonte: Elaborada pelo autor.}",
        "\\end{table}",
        "",
    ])
    TABLES.mkdir(parents=True, exist_ok=True)
    caminho_tex = TABLES / "tab_espectro_2d.tex"
    caminho_tex.write_text(tex, encoding="utf-8")
    arquivos = [
        gravar_csv(df, METRICS / "espectro_2d.csv"),
        gravar_csv(resumo, METRICS / "resumo_espectro_2d.csv"),
        gravar_csv(pd.DataFrame(longos), METRICS / "coeficientes_espectro_2d.csv"),
        caminho_fig, caminho_tex,
    ]
    print(resumo.round(2).to_string(index=False))
    registrar(__file__, arquivos, {"limiar": LIMIAR, "limiar_relativo": LIMIAR_RELATIVO, "n_grade": N})


if __name__ == "__main__":
    main()
