"""Métricas de desempenho (Etapa 8).

Aqui ficam só as métricas que dependem do TREINO. As métricas de custo do
circuito (profundidade, portas, qubits) são estáticas e vivem em circuit_stats.py.
"""

import numpy as np
from pennylane import numpy as pnp


def square_loss(labels, predictions):
    """Custo quadrático sobre rótulos em {-1, +1}."""
    return pnp.mean((pnp.asarray(predictions) - np.asarray(labels)) ** 2)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))