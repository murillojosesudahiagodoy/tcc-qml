"""Classificador quântico variacional (Etapas 3 e 8).

Monta a cadeia do roteiro:

    x -> psi(x) -> U(theta) psi(x) -> <Z_0> -> y_chapeu

A montagem é genérica o bastante para as quatro codificações da Etapa 9: as
que preparam o estado e depois entregam ao ansatz (angle, amplitude, zz) e a
que intercala dados e camadas treináveis (re-uploading, `interleaved=True`).
"""

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tccqml.ansatz import get_ansatz, init_weights
from tccqml.embeddings import get_encoding


@dataclass
class Classificador:
    """Circuito + metadados necessários para treinar e para medir custo."""

    circuit: object
    encoding: str
    n_qubits: int
    n_layers: int
    weights_shape: tuple
    ansatz: str = "strongly_entangling"
    alpha_shape: tuple | None = None
    interleaved: bool = False
    enc_kwargs: dict | None = None

    @property
    def n_params_ansatz(self) -> int:
        """Parâmetros das camadas treináveis do circuito."""
        return int(np.prod(self.weights_shape))

    @property
    def n_params_encoding(self) -> int:
        """Parâmetros próprios da codificação. Zero nas quatro do núcleo.

        No re-uploading os pesos intercalados são do ansatz, não da
        codificação: a codificação apenas escolhe ONDE eles entram.
        """
        return 0 if self.alpha_shape is None else int(np.prod(self.alpha_shape))

    @property
    def n_params_circuito(self) -> int:
        """O `p` das Tabelas 2 e 3 e das Eqs. 2.47/2.48: só o circuito."""
        return self.n_params_ansatz + self.n_params_encoding

    @property
    def n_params(self) -> int:
        """Parâmetros treináveis do modelo, incluindo o viés clássico.

        NÃO é o `p` das fórmulas de custo. O viés é somado depois da medição e
        sua derivada é clássica, sem executar o circuito (Seção 2.2.3.3), então
        ele não entra nem na Eq. 2.47 nem na Eq. 2.48. Para isso use
        `n_params_circuito`.
        """
        return self.n_params_circuito + 1


def build(
    encoding: str,
    n_features: int,
    n_layers: int = 2,
    ansatz: str = "strongly_entangling",
    device: str = "default.qubit",
    enc_kwargs: dict | None = None,
) -> Classificador:
    """Constrói o QNode que devolve <Z_0> em [-1, 1].

    `ansatz` só deve sair do padrão na ablação da T5 — nos experimentos
    principais ele fica congelado em `strongly_entangling`.
    """
    enc_kwargs = dict(enc_kwargs or {})
    enc = get_encoding(encoding, **enc_kwargs)
    ans = get_ansatz(ansatz)
    n_qubits = enc.n_qubits(n_features)
    dev = qml.device(device, wires=n_qubits)

    if enc.interleaved:
        # A codificação monta o circuito inteiro; o ansatz não é aplicado
        # depois e é ela quem dita a forma dos pesos.
        w_shape = tuple(enc.params_shape(n_features))
        alpha_shape = None
    else:
        w_shape = tuple(ans.weights_shape(n_layers, n_qubits))
        alpha_shape = (
            tuple(enc.params_shape(n_features)) if enc.params_shape is not None else None
        )

    @qml.qnode(dev)
    def circuit(x, weights, alpha=None):
        if enc.interleaved:
            enc.apply(x, wires=range(n_qubits), params=weights)
        else:
            enc.apply(x, wires=range(n_qubits), params=alpha)
            ans.apply(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return Classificador(
        circuit=circuit,
        encoding=encoding,
        n_qubits=n_qubits,
        n_layers=n_layers,
        weights_shape=w_shape,
        ansatz=ansatz,
        alpha_shape=alpha_shape,
        interleaved=enc.interleaved,
        enc_kwargs=enc_kwargs,
    )


def pesos_iniciais(clf: Classificador, seed: int = 42):
    """Devolve (weights, alpha, bias) prontos para o otimizador.

    `alpha` é None quando a codificação não tem parâmetros próprios, que é o
    caso das quatro codificações do núcleo.
    """
    w = init_weights(clf.n_layers, clf.n_qubits, seed=seed, forma=clf.weights_shape)
    alpha = (
        None
        if clf.alpha_shape is None
        else init_weights(clf.n_layers, clf.n_qubits, seed=seed + 1, forma=clf.alpha_shape)
    )
    b = pnp.array(0.0, requires_grad=True)
    return w, alpha, b


def saida_continua(clf: Classificador, weights, alpha, bias, X):
    """<Z_0> + bias para um lote de amostras. Imagem aproximada: [-1, 1]."""
    return clf.circuit(X, weights, alpha) + bias


def prever(clf: Classificador, weights, alpha, bias, X):
    """Converte a saída contínua em rótulo {0, 1}."""
    z = np.asarray(saida_continua(clf, weights, alpha, bias, X), dtype=float)
    return (z > 0).astype(int)
