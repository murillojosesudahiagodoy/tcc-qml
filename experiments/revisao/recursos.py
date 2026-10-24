"""Recursos por codificação e por profundidade, k e R = 1..5.

Separa quatro grandezas que a Tabela tab:custo mostra juntas: profundidade do
circuito compilado, número de CNOTs, avaliações do circuito por passo do
otimizador e shots por passo. Tudo sai de `circuit_stats.stats`, a mesma
função da tab:custo (circuito decomposto em RY, RZ, Hadamard e CNOT; custo
por passo (2p + 1)|B| e (2p + 1)|B|S, analítico). Nada é contado à mão.

Saídas (results/revisao/):
    metrics/recursos_profundidade.csv
    tables/tab_recursos_profundidade.tex

Uso: python experiments/revisao/recursos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _comum import METRICS, TABLES, gravar_csv, registrar

from tccqml import model
from tccqml.circuit_stats import stats
from tccqml.config import PADRAO

ROT = {
    "angle": "\\textit{Angle}",
    "reuploading": "\\textit{Re-uploading}",
    "amplitude": "\\textit{Amplitude}",
    "zz": "\\textit{Feature map} ZZ",
}


def configuracoes():
    for k in (1, 2, 3, 4, 5):
        yield "angle", k, {"L_var": k}, None
    for k in (1, 2, 3, 4, 5):
        yield "reuploading", k, {"L_var": PADRAO.L_var}, {"R": k}
    yield "amplitude", PADRAO.L_var, {"L_var": PADRAO.L_var}, None
    yield "zz", PADRAO.L_var, {"L_var": PADRAO.L_var}, None


def main() -> None:
    x = [0.3, 1.1]  # a estrutura do circuito não depende do valor do dado
    linhas = []
    for enc, k, kw, enc_kwargs in configuracoes():
        clf = model.build(enc, n_features=2, enc_kwargs=enc_kwargs, **kw)
        s = stats(clf, x, batch_size=PADRAO.batch_efetivo, shots=PADRAO.shots)
        linhas.append({"encoding": enc, "k": k, **{c: s[c] for c in (
            "n_qubits", "depth", "depth_encoding", "gates_total", "gates_2q", "gates_2q_encoding",
            "n_params_circuito", "aval_gradiente_amostra", "n_aval_passo", "n_shots_passo")}})
    df = pd.DataFrame(linhas)

    corpo, anterior = [], None
    for r in df.itertuples():
        if anterior is not None and r.encoding != anterior:
            corpo.append("\\midrule")
        nome = ROT[r.encoding] if r.encoding != anterior else ""
        anterior = r.encoding
        kk = "--" if r.encoding in ("amplitude", "zz") else str(r.k)
        corpo.append(
            f"{nome} & {kk} & {r.n_params_circuito} & {r.depth} & {r.gates_2q} & "
            f"{r.n_aval_passo} & {r.n_shots_passo:,}".replace(",", "\\,") + " \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{table}[htb]",
            "\\centering",
            "\\small",
            ("\\caption[Recursos por codificação e profundidade]{Recursos do circuito compilado e "
            "custo de um passo de otimização em \\textit{hardware}, por codificação e por "
            "profundidade ($k = L_\\text{var}$ no \\textit{angle}, $k = R$ no \\textit{re-uploading}).}"),
            "\\label{tab:recursos-profundidade}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Codificação & $k$ & $p$ & Profundidade & CNOTs & Aval./passo & \\textit{Shots}/passo \\\\",
            "\\midrule",
            *corpo,
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[2pt]",
            ("{\\footnotesize Profundidade e CNOTs: circuito completo, decomposto em portas básicas "
            "como na Tabela~\\ref{tab:custo}. Aval./passo: $(2p + 1)|B|$; \\textit{Shots}/passo: "
            "$(2p + 1)|B|S$, com $|B| = 20$ e $S = 1000$. \\textit{Amplitude} e ZZ com o \\textit{ansatz} "
            "de referência ($L_\\text{var} = 2$). Gerada por código (\\texttt{circuit\\_stats.stats}).}"),
            "\\\\[2pt]",
            "{\\footnotesize Fonte: Elaborada pelo autor.}",
            "\\end{table}",
            "",
        ]
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    caminho = TABLES / "tab_recursos_profundidade.tex"
    caminho.write_text(tex, encoding="utf-8")
    arquivos = [gravar_csv(df, METRICS / "recursos_profundidade.csv"), caminho]
    print(df.to_string(index=False))
    registrar(__file__, arquivos)


if __name__ == "__main__":
    main()
