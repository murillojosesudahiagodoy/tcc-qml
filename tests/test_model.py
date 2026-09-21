"""Testes do classificador variacional (Etapas 3 e 8)."""

import numpy as np
import pytest
from pennylane import numpy as pnp

from tccqml import model
from tccqml.ansatz import init_weights, weights_shape
from tccqml.data import load_dataset


def _clf(n_layers=2, encoding="angle", **kwargs):
    return model.build(encoding, n_features=2, n_layers=n_layers, **kwargs)


def test_numero_de_qubits_segue_a_codificacao():
    assert _clf().n_qubits == 2


def test_amplitude_usa_menos_qubits_que_as_demais():
    """Não é bug: d atributos cabem em ceil(log2 d) qubits (Seção 2.4.4)."""
    assert _clf(encoding="amplitude").n_qubits == 1


def test_saida_esta_no_intervalo_de_um_observavel():
    """<Z> vive em [-1, 1]. Sair disso significa circuito ou medição errados."""
    clf = _clf()
    w, alpha, _ = model.pesos_iniciais(clf)
    ds = load_dataset("moons", n_samples=40)

    z = np.asarray(model.saida_continua(clf, w, alpha, pnp.array(0.0), ds.X_train))

    assert np.all(z >= -1.0 - 1e-9) and np.all(z <= 1.0 + 1e-9)


def test_predicao_e_binaria_e_tem_o_tamanho_certo():
    clf = _clf()
    w, alpha, b = model.pesos_iniciais(clf)
    ds = load_dataset("moons", n_samples=40)

    y = model.prever(clf, w, alpha, b, ds.X_train)

    assert y.shape == ds.y_train.shape
    assert set(np.unique(y)) <= {0, 1}


def test_contagem_de_parametros_bate_com_o_ansatz():
    clf = _clf(n_layers=3)
    esperado = int(np.prod(weights_shape(3, 2)))

    assert clf.n_params_circuito == esperado
    assert clf.n_params == esperado + 1  # +1 do viés clássico


def test_p_do_circuito_nao_inclui_o_vies():
    """A ambiguidade que a T0 resolveu.

    O `p` das Tabelas 2 e 3 e das Eqs. 2.47/2.48 é 12 na configuração de
    referência: só o circuito. O viés é somado depois da medição e sua derivada
    é clássica, então ele não entra em nenhuma fórmula de custo — mas continua
    sendo um parâmetro treinável do modelo, e daí os 13.
    """
    clf = _clf()

    assert clf.n_params_circuito == 12
    assert clf.n_params == 13
    assert clf.n_params_encoding == 0


@pytest.mark.parametrize("encoding", ["angle", "amplitude", "reuploading", "zz"])
def test_nenhuma_codificacao_do_nucleo_tem_parametros_proprios(encoding):
    """No re-uploading os pesos intercalados são do ansatz, não da codificação."""
    assert _clf(encoding=encoding).n_params_encoding == 0


def test_reuploading_tem_p_ditado_pelo_numero_de_blocos():
    """L_reup é independente do n_layers do ansatz (a codificação é interleaved)."""
    clf = _clf(n_layers=2, encoding="reuploading", enc_kwargs={"L_reup": 4})

    assert clf.interleaved
    assert clf.weights_shape == (4, 2, 3)
    assert clf.n_params_circuito == 24


def test_pesos_iniciais_sao_pequenos():
    """Inicialização grande em circuito profundo cai em barren plateau."""
    w = init_weights(n_layers=4, n_qubits=2, seed=1)
    assert float(pnp.max(pnp.abs(w))) < 1.0


def test_pesos_iniciais_sao_reprodutiveis():
    assert np.allclose(init_weights(2, 2, seed=7), init_weights(2, 2, seed=7))
