"""Utilidades comuns aos scripts de `experiments/revisao/`.

Os scripts desta pasta só IMPORTAM o código de `tccqml` (treino, dados,
espectro, contagem de recursos); nada do núcleo é alterado. Todas as saídas
vão para `results/revisao/`, sem tocar nos CSVs, pesos e figuras da versão
1.0.0.

Toda saída registra a versão do código: CSVs ganham a coluna `commit`; PDFs
levam o hash nos metadados; cada script grava um manifesto
`results/revisao/versoes/<script>.json` com o hash, se `src/` ou
`experiments/` tinham alterações não commitadas e a lista de arquivos
gerados. O hash é lido com `git rev-parse HEAD` (só leitura).
"""

from __future__ import annotations

import os

# Um fio de BLAS por processo: os treinos rodam em paralelo, um por núcleo.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "results" / "revisao"
METRICS = SAIDA / "metrics"
TABLES = SAIDA / "tables"
FIGURES = SAIDA / "figures"
WEIGHTS = SAIDA / "weights"
VERSOES = SAIDA / "versoes"

# Sementes que nenhum experimento anterior usou (`sementes_novas.py`).
SEMENTES_NOVAS = (47, 48, 49, 50, 51)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def versao_codigo() -> dict:
    """Hash do commit atual e se o código tinha alterações locais (só leitura).

    Só `src/` e `experiments/` contam: as saídas em `results/` mudam a cada
    execução e não dizem nada sobre o código que as gerou.
    """
    return {
        "commit": _git("rev-parse", "HEAD") or "desconhecido",
        "arvore_com_alteracoes": bool(_git("status", "--porcelain", "--", "src", "experiments")),
    }


COMMIT = versao_codigo()["commit"]


def gravar_csv(df: pd.DataFrame, caminho: Path) -> Path:
    """CSV com a coluna `commit` (versão do código que gerou a linha)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.assign(commit=COMMIT).to_csv(caminho, index=False)
    return caminho


def salvar_figura(fig, caminho: Path) -> Path:
    import matplotlib.pyplot as plt

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        caminho,
        format="pdf",
        metadata={"CreationDate": None, "Subject": f"tcc-qml commit {COMMIT}"},
    )
    plt.close(fig)
    return caminho


def registrar(script: str, saidas: list[Path], extra: dict | None = None) -> Path:
    """Manifesto do script: versão do código, comando e arquivos gerados."""
    VERSOES.mkdir(parents=True, exist_ok=True)
    manifesto = {
        **versao_codigo(),
        "script": Path(script).resolve().relative_to(RAIZ).as_posix(),
        "comando": " ".join([Path(sys.executable).name, *sys.argv]),
        "data": datetime.now(UTC).isoformat(timespec="seconds"),
        "saidas": sorted(str(Path(s).resolve().relative_to(RAIZ)).replace("\\", "/") for s in saidas),
        **(extra or {}),
    }
    destino = VERSOES / f"{Path(script).stem}.json"
    destino.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


# --------------------------------------------------------------------------
# Treino de uma configuração, executável em outro processo
# --------------------------------------------------------------------------


def treinar_config(cfg: dict) -> dict:
    """Um treino com `experiments.rodar_um`, variando só o que `cfg` pede.

    `cfg`: encoding ("angle" ou "reuploading"), dataset, seed, k (L_var do
    angle ou R do re-uploading), eta e epocas. O resto vem de `PADRAO`.
    Devolve histórico, resumo e os parâmetros da época escolhida como arrays
    NumPy, para atravessar a fronteira entre processos.
    """
    from tccqml.config import PADRAO
    from tccqml.experiments import rodar_um

    enc, k = cfg["encoding"], int(cfg["k"])
    protocolo = replace(
        PADRAO,
        eta=float(cfg.get("eta", PADRAO.eta)),
        epocas=int(cfg.get("epocas", PADRAO.epocas)),
        L_var=k if enc == "angle" else PADRAO.L_var,
    )
    hist, resumo, r = rodar_um(enc, cfg["dataset"], int(cfg["seed"]), protocolo, R=k)
    extra = {"k": k, "eta": protocolo.eta, "epocas": protocolo.epocas}
    return {
        "cfg": cfg,
        "hist": hist.assign(**extra),
        "resumo": {**resumo, **extra},
        "weights": np.asarray(r.weights, dtype=float),
        "bias": float(r.bias),
    }


def rodar_paralelo(cfgs: list[dict], n_proc: int | None = None, rotulo: str = "") -> list[dict]:
    """Roda `treinar_config` em paralelo e imprime o progresso com estimativa."""
    n_proc = n_proc or max(1, (os.cpu_count() or 2) - 1)
    saidas, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futuros = [pool.submit(treinar_config, c) for c in cfgs]
        for i, fut in enumerate(as_completed(futuros), 1):
            s = fut.result()
            saidas.append(s)
            dec = time.perf_counter() - t0
            c = s["cfg"]
            print(
                f"[{rotulo} {i:4d}/{len(cfgs)}] {c['encoding']:11s} k={c['k']} "
                f"{c['dataset']:7s} s={c['seed']} eta={c.get('eta', '-')} ep={c.get('epocas', '-')} "
                f"teste={s['resumo']['acc_teste']:.3f}  "
                f"(decorrido {dec / 60:.1f} min, faltam ~{dec / i * (len(cfgs) - i) / 60:.1f} min)",
                flush=True,
            )
    return saidas


def salvar_pesos(saida: dict, pasta: Path, nome: str) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{nome}.npz"
    np.savez(caminho, weights=saida["weights"], bias=saida["bias"], commit=COMMIT)
    return caminho


def media_desvio(df: pd.DataFrame, chaves: list[str], colunas: list[str]) -> pd.DataFrame:
    """Média, desvio (ddof = 1) e número de sementes por grupo."""
    agg = df.groupby(chaves, as_index=False)[colunas].agg(["mean", "std"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    n = df.groupby(chaves, as_index=False)["seed"].nunique().rename(columns={"seed": "n_sementes"})
    return agg.merge(n, on=chaves)


def dec(valor: float, casas: int = 3) -> str:
    """Número com vírgula decimal, como no texto."""
    return f"{valor:.{casas}f}".replace(".", ",")


def md(media: float, desvio: float, casas: int = 3) -> str:
    return f"{dec(media, casas)} $\\pm$ {dec(desvio, casas)}"
