"""Partições salvas e verificações do pré-processamento.

1. Para cada conjunto e semente (42-46 da grade e 47-51 das sementes novas),
   grava os pontos gerados antes da normalização, os rótulos, os índices de
   treino, validação e teste e os extremos do MinMaxScaler ajustado no
   treino. Os índices são obtidos repetindo as duas chamadas de
   `train_test_split` de `data.load_dataset` sobre `arange(N)`: a divisão
   estratificada só depende dos rótulos, de N e da semente. O script confere
   que os índices reproduzem exatamente os arrays de `load_dataset`.
2. Verifica se alguma entrada do amplitude encoding tem norma zero, nas duas
   normalizações usadas ([0, pi] e [-1, 1]). O código não trata esse caso:
   `AmplitudeEmbedding(normalize=True)` devolveria NaN, e `model.prever`
   atribuiria a classe 0. A verificação mede se o caso ocorre.
3. Mede onde a normalização em [-1, 1] leva o centro geométrico dos dados
   (média dos pontos de treino).

Saídas (results/revisao/):
    particoes/<dataset>_<seed>.npz
    metrics/verificacao_particoes.csv

Uso: python experiments/revisao/particoes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _comum import COMMIT, METRICS, SAIDA, SEMENTES_NOVAS, gravar_csv, registrar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tccqml.config import PADRAO
from tccqml.data import _GENERATORS, load_dataset


def indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As mesmas duas divisões de `load_dataset`, aplicadas aos índices."""
    idx = np.arange(len(y))
    resto, teste, y_resto, _ = train_test_split(
        idx, y, test_size=PADRAO.test_size, random_state=seed, stratify=y
    )
    treino, val = train_test_split(
        resto, test_size=PADRAO.val_size / (1 - PADRAO.test_size), random_state=seed,
        stratify=y_resto,
    )
    return treino, val, teste


def main() -> None:
    pasta = SAIDA / "particoes"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos, linhas = [], []
    for dataset in PADRAO.datasets_grade:
        for seed in (*PADRAO.sementes, *SEMENTES_NOVAS):
            X, y = _GENERATORS[dataset](PADRAO.n_samples, PADRAO.noise, seed)
            tr, va, te = indices(y, seed)
            scaler = MinMaxScaler(feature_range=PADRAO.feature_range).fit(X[tr])
            ds = load_dataset(dataset, seed=seed)
            confere = all(
                np.array_equal(scaler.transform(X[i]), Xd) and np.array_equal(y[i], yd)
                for i, Xd, yd in ((tr, ds.X_train, ds.y_train), (va, ds.X_val, ds.y_val),
                                  (te, ds.X_test, ds.y_test))
            )
            caminho = pasta / f"{dataset}_{seed}.npz"
            np.savez(caminho, X_bruto=X, y=y, idx_treino=tr, idx_val=va, idx_teste=te,
                     data_min=scaler.data_min_, data_max=scaler.data_max_, commit=COMMIT)
            arquivos.append(caminho)

            ds_m1 = load_dataset(dataset, seed=seed, feature_range=(-1.0, 1.0))
            normas_pi = np.linalg.norm(np.vstack([ds.X_train, ds.X_val, ds.X_test]), axis=1)
            normas_m1 = np.linalg.norm(np.vstack([ds_m1.X_train, ds_m1.X_val, ds_m1.X_test]), axis=1)
            centro_m1 = ds_m1.X_train.mean(axis=0)
            linhas.append({
                "dataset": dataset, "seed": seed,
                "n_treino": len(tr), "n_val": len(va), "n_teste": len(te),
                "indices_reproduzem_load_dataset": confere,
                "fracao_classe1_treino": y[tr].mean(), "fracao_classe1_val": y[va].mean(),
                "fracao_classe1_teste": y[te].mean(), "fracao_classe1_total": y.mean(),
                "norma_min_0_pi": normas_pi.min(), "norma_min_m1_1": normas_m1.min(),
                "n_norma_zero_0_pi": int((normas_pi == 0).sum()),
                "n_norma_zero_m1_1": int((normas_m1 == 0).sum()),
                "centro_treino_m1_1_x1": centro_m1[0], "centro_treino_m1_1_x2": centro_m1[1],
                "fora_0_pi_val_teste": int(((np.vstack([ds.X_val, ds.X_test]) < 0)
                                            | (np.vstack([ds.X_val, ds.X_test]) > np.pi)).any(axis=1).sum()),
            })
    df = pd.DataFrame(linhas)
    arquivos.append(gravar_csv(df, METRICS / "verificacao_particoes.csv"))
    print(df.round(4).to_string(index=False))
    print("todas as partições reproduzem load_dataset:", bool(df["indices_reproduzem_load_dataset"].all()))
    registrar(__file__, arquivos)


if __name__ == "__main__":
    main()
