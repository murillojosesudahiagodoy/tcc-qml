"""Treinamento do classificador variacional (Etapa 8)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from tccqml.data import Dataset
from tccqml.metrics import accuracy, square_loss
from tccqml.model import Classificador, pesos_iniciais, prever, saida_continua


@dataclass
class Resultado:
    """Saída de um treino: parâmetros finais e histórico por época."""

    weights: object
    bias: object
    historico: pd.DataFrame
    acc_treino: float
    acc_teste: float
    meta: dict = field(default_factory=dict)


def treinar(
    clf: Classificador,
    ds: Dataset,
    epocas: int = 30,
    batch_size: int = 20,
    lr: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> Resultado:
    """Otimiza (weights, bias) minimizando o custo quadrático.

    O gradiente do circuito vem do parameter-shift do PennyLane; o passo
    de atualização é clássico. É isso que torna o modelo híbrido.
    """
    weights, bias = pesos_iniciais(clf, seed=seed)
    opt = qml.AdamOptimizer(stepsize=lr)
    rng = np.random.default_rng(seed)

    X_tr = pnp.array(ds.X_train, requires_grad=False)
    y_tr = np.asarray(ds.y_train_pm1, dtype=float)
    n = len(X_tr)

    def custo(w, b, Xb, yb):
        return square_loss(yb, saida_continua(clf, w, b, Xb))

    linhas = []
    for epoca in range(1, epocas + 1):
        for idx in np.array_split(rng.permutation(n), max(1, n // batch_size)):
            weights, bias, _, _ = opt.step(custo, weights, bias, X_tr[idx], y_tr[idx])

        custo_epoca = float(custo(weights, bias, X_tr, y_tr))
        acc_tr = accuracy(ds.y_train, prever(clf, weights, bias, ds.X_train))
        acc_te = accuracy(ds.y_test, prever(clf, weights, bias, ds.X_test))
        linhas.append(
            {"epoca": epoca, "custo": custo_epoca, "acc_treino": acc_tr, "acc_teste": acc_te}
        )

        if verbose and (epoca % 5 == 0 or epoca == 1):
            print(f"época {epoca:3d}  custo={custo_epoca:.4f}  treino={acc_tr:.3f}  teste={acc_te:.3f}")

    hist = pd.DataFrame(linhas)
    return Resultado(
        weights=weights,
        bias=bias,
        historico=hist,
        acc_treino=float(hist["acc_treino"].iloc[-1]),
        acc_teste=float(hist["acc_teste"].iloc[-1]),
        meta={
            "encoding": clf.encoding,
            "dataset": ds.name,
            "n_qubits": clf.n_qubits,
            "n_layers": clf.n_layers,
            "n_params": clf.n_params,
            "epocas": epocas,
            "lr": lr,
        },
    )