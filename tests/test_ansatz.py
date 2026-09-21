"""Testes do ansatz e da ablação de entrelaçamento (T5).

O teste `test_ablacao_cai_para_o_acaso_no_xor` é o mais informativo do
repositório: ele não checa uma peça, checa a implementação INTEIRA contra uma
previsão fechada do texto. Se ele passar, a cadeia
dados -> codificação -> ansatz -> medição -> treino está coerente.
"""

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from tccqml import model
from tccqml.ansatz import ANSATZE, get_ansatz, init_weights, weights_shape_local
from tccqml.data import load_dataset
from tccqml.train import treinar


def test_registro_de_ansatze():
    for nome, ans in ANSATZE.items():
        assert ans.name == nome
        assert callable(ans.apply)
        assert ans.descricao


def test_ansatz_desconhecido_falha():
    with pytest.raises(ValueError, match="desconhecido"):
        get_ansatz("mágico")


def test_local_tem_a_mesma_forma_de_pesos_que_o_padrao():
    """Mesmo p nos dois: a ablação isola o entrelaçamento, não o tamanho."""
    from tccqml.ansatz import weights_shape

    assert tuple(weights_shape_local(2, 2)) == tuple(weights_shape(2, 2))


def test_local_nao_emite_nenhuma_porta_de_dois_qubits():
    dev = qml.device("default.qubit", wires=2)
    ans = get_ansatz("local")

    @qml.qnode(dev)
    def circuito(w):
        ans.apply(w, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    pesos = init_weights(2, 2, forma=weights_shape_local(2, 2))
    contagem = qml.specs(circuito)(pesos)["resources"].gate_types

    assert sum(v for k, v in contagem.items() if k in {"CNOT", "CZ"}) == 0


def test_local_ignora_o_segundo_atributo():
    """Eq. 2.83, na forma mais direta possível.

    Com estado produto e U = u_0 (x) u_1, o valor de <Z_0> depende só de x_1:
    o segundo atributo é codificado, ocupa um qubit e é completamente ignorado
    pela medição. Mudar x_2 não pode mexer na saída.
    """
    clf = model.build("angle", n_features=2, n_layers=2, ansatz="local")
    w, alpha, b = model.pesos_iniciais(clf)

    saidas = [
        float(model.saida_continua(clf, w, alpha, b, pnp.array([0.7, x2], requires_grad=False)))
        for x2 in (0.0, 1.0, 2.0, 3.0)
    ]

    assert np.allclose(saidas, saidas[0])


def test_ablacao_cai_para_o_acaso_no_xor():
    """Previsão 3: sem entrelaçamento no ansatz, o XOR vira moeda (~0,50).

    O par de controle com o ansatz congelado precisa ficar bem acima disso —
    senão o que o teste mostra é que o treino não funciona, não que o
    entrelaçamento importa.
    """
    ds = load_dataset("xor")

    com = treinar(
        model.build("angle", n_features=2, n_layers=2),
        ds,
        epocas=30,
        verbose=False,
    )
    sem = treinar(
        model.build("angle", n_features=2, n_layers=2, ansatz="local"),
        ds,
        epocas=30,
        verbose=False,
    )

    assert 0.40 <= sem.acc_teste <= 0.60, f"esperado ~0,50, obtido {sem.acc_teste:.3f}"
    assert com.acc_teste > 0.75, f"o controle precisa aprender: {com.acc_teste:.3f}"
