"""Testes do classificador variacional (Etapas 3 e 8)."""

import numpy as np
from pennylane import numpy as pnp

from tccqml import model
from tccqml.ansatz import init_weights, weights_shape
from tccqml.data import load_dataset


def _clf(n_layers=2):
    return model.build("angle", n_features=2, n_layers=n_layers)


def test_numero_de_qubits_segue_a_codificacao():
    assert _clf().n_qubits == 2


def test_saida_esta_no_intervalo_de_um_observavel():
    """<Z> vive em [-1, 1]. Sair disso significa circuito ou medição errados."""
    clf = _clf()
    w, b = model.pesos_iniciais(clf)
    ds = load_dataset("moons", n_samples=40)

    z = np.asarray(model.saida_continua(clf, w, pnp.array(0.0), ds.X_train))

    assert np.all(z >= -1.0 - 1e-9) and np.all(z <= 1.0 + 1e-9)


def test_predicao_e_binaria_e_tem_o_tamanho_certo():
    clf = _clf()
    w, b = model.pesos_iniciais(clf)
    ds = load_dataset("moons", n_samples=40)

    y = model.prever(clf, w, b, ds.X_train)

    assert y.shape == ds.y_train.shape
    assert set(np.unique(y)) <= {0, 1}


def test_contagem_de_parametros_bate_com_o_ansatz():
    clf = _clf(n_layers=3)
    esperado = int(np.prod(weights_shape(3, 2))) + 1  # +1 do bias
    assert clf.n_params == esperado


def test_pesos_iniciais_sao_pequenos():
    """Inicialização grande em circuito profundo cai em barren plateau."""
    w = init_weights(n_layers=4, n_qubits=2, seed=1)
    assert float(pnp.max(pnp.abs(w))) < 1.0


def test_pesos_iniciais_sao_reprodutiveis():
    assert np.allclose(init_weights(2, 2, seed=7), init_weights(2, 2, seed=7))