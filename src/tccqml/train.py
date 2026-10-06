"""Treinamento do classificador variacional (Etapa 8)."""

import copy
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
    """Saída de um treino: parâmetros da época escolhida e histórico por época.

    `weights`, `alpha`, `bias` e as três acurácias são as do modelo da época
    `epoca_escolhida` (a de maior acurácia de validação), não as da última
    época. `acc_teste` é a única medida feita no conjunto de teste.
    """

    weights: object
    bias: object
    historico: pd.DataFrame
    acc_treino: float
    acc_teste: float
    acc_val: float = float("nan")
    epoca_escolhida: int = 0
    alpha: object = None
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
    """Otimiza (weights, alpha, bias) minimizando o custo quadrático.

    O gradiente do circuito é obtido por retropropagação sobre a simulação:
    o dispositivo `default.qubit` diferencia o próprio simulador, que é o
    caminho mais barato em software (Seção 2.3.5.1). O parameter-shift NÃO é
    executado aqui — ele é contabilizado ANALITICAMENTE em `circuit_stats.py`
    (Eqs. 2.47 e 2.48), porque é o custo que o mesmo treino teria em hardware
    real. O passo de atualização é clássico, e é isso que torna o modelo
    híbrido.

    A cada época o histórico registra custo e acurácia de treino e a acurácia
    de VALIDAÇÃO — nunca a de teste. A validação escolhe a época de parada: os
    parâmetros da época com maior `acc_val` são guardados (no empate fica a
    mais antiga, que é o modelo mais simples de alcançar) e restaurados no
    fim. Só então o teste é usado, UMA única vez, com esses parâmetros. Se a
    acurácia de teste fosse olhada época a época, escolher a melhor curva seria
    ajustar ao teste, e o número reportado deixaria de ser uma estimativa
    honesta da generalização.
    """
    weights, alpha, bias = pesos_iniciais(clf, seed=seed)
    opt = qml.AdamOptimizer(stepsize=lr)
    rng = np.random.default_rng(seed)

    X_tr = pnp.array(ds.X_train, requires_grad=False)
    y_tr = np.asarray(ds.y_train_pm1, dtype=float)
    n = len(X_tr)

    def custo(w, a, b, Xb, yb):
        return square_loss(yb, saida_continua(clf, w, a, b, Xb))

    # O otimizador do PennyLane não aceita None entre os argumentos treináveis,
    # então `alpha` só entra no passo quando a codificação de fato tem
    # parâmetros próprios (nenhuma das quatro do núcleo tem).
    def custo_sem_alpha(w, b, Xb, yb):
        return custo(w, None, b, Xb, yb)

    linhas = []
    melhor_acc_val = -np.inf
    melhor_epoca = 0
    melhores = None
    for epoca in range(1, epocas + 1):
        for idx in np.array_split(rng.permutation(n), max(1, n // batch_size)):
            if alpha is None:
                weights, bias, _, _ = opt.step(custo_sem_alpha, weights, bias, X_tr[idx], y_tr[idx])
            else:
                weights, alpha, bias, _, _ = opt.step(
                    custo, weights, alpha, bias, X_tr[idx], y_tr[idx]
                )

        custo_epoca = float(custo(weights, alpha, bias, X_tr, y_tr))
        acc_tr = accuracy(ds.y_train, prever(clf, weights, alpha, bias, ds.X_train))
        acc_va = accuracy(ds.y_val, prever(clf, weights, alpha, bias, ds.X_val))
        linhas.append(
            {"epoca": epoca, "custo": custo_epoca, "acc_treino": acc_tr, "acc_val": acc_va}
        )

        # Estritamente maior: no empate fica a época mais antiga.
        if acc_va > melhor_acc_val:
            melhor_acc_val = acc_va
            melhor_epoca = epoca
            melhores = copy.deepcopy((weights, alpha, bias))

        if verbose and (epoca % 5 == 0 or epoca == 1):
            print(f"época {epoca:3d}  custo={custo_epoca:.4f}  treino={acc_tr:.3f}  val={acc_va:.3f}")

    hist = pd.DataFrame(linhas)
    weights, alpha, bias = melhores
    escolhida = hist[hist["epoca"] == melhor_epoca].iloc[0]
    # A única vez em que o conjunto de teste é tocado.
    acc_teste = accuracy(ds.y_test, prever(clf, weights, alpha, bias, ds.X_test))

    if verbose:
        print(
            f"época escolhida pela validação: {melhor_epoca}  "
            f"(treino={escolhida['acc_treino']:.3f}  val={escolhida['acc_val']:.3f})"
        )

    return Resultado(
        weights=weights,
        alpha=alpha,
        bias=bias,
        historico=hist,
        acc_treino=float(escolhida["acc_treino"]),
        acc_teste=float(acc_teste),
        acc_val=float(escolhida["acc_val"]),
        epoca_escolhida=int(melhor_epoca),
        meta={
            "encoding": clf.encoding,
            "dataset": ds.name,
            "ansatz": clf.ansatz,
            "n_qubits": clf.n_qubits,
            "n_layers": clf.n_layers,
            "n_params_circuito": clf.n_params_circuito,
            "n_params_encoding": clf.n_params_encoding,
            "n_params": clf.n_params,
            "epocas": epocas,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
        },
    )
