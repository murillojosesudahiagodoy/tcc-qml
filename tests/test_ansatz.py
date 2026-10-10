"""Testes do ansatz e da ablação de entrelaçamento.

`test_ablacao_cai_para_o_acaso_no_xor` checa a cadeia inteira
(dados -> codificação -> ansatz -> medição -> treino) contra a hipótese H3,
em vez de checar uma peça isolada.
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
    """A saída sem entrelaçamento, na forma mais direta possível.

    Cap. 2, "Entrelaçamento e termos cruzados".

    Com estado produto e U = u_0 (x) u_1, o valor de <Z_0> depende só de x_1:
    o segundo atributo é codificado, ocupa um qubit e é completamente ignorado
    pela medição. Mudar x_2 não pode mexer na saída.
    """
    clf = model.build("angle", n_features=2, L_var=2, ansatz="local")
    w, alpha, b = model.pesos_iniciais(clf)

    saidas = [
        float(model.saida_continua(clf, w, alpha, b, pnp.array([0.7, x2], requires_grad=False)))
        for x2 in (0.0, 1.0, 2.0, 3.0)
    ]

    assert np.allclose(saidas, saidas[0])


def test_ablacao_cai_para_o_acaso_no_xor():
    """H3: sem entrelaçamento no ansatz, o xor vira moeda (~0,50).

    O par de controle com o ansatz congelado precisa ficar bem acima disso —
    senão o que o teste mostra é que o treino não funciona, não que o
    entrelaçamento importa.
    """
    ds = load_dataset("xor")

    com = treinar(
        model.build("angle", n_features=2, L_var=2),
        ds,
        epocas=30,
        verbose=False,
    )
    sem = treinar(
        model.build("angle", n_features=2, L_var=2, ansatz="local"),
        ds,
        epocas=30,
        verbose=False,
    )

    assert 0.40 <= sem.acc_teste <= 0.60, f"esperado ~0,50, obtido {sem.acc_teste:.3f}"
    assert com.acc_teste > 0.75, f"o controle precisa aprender: {com.acc_teste:.3f}"


# --------------------------------------------------------------------------
# Parâmetros inertes (Cap. 2, parágrafo sobre U_ent^dagger Z_0 U_ent = Z_1)
# --------------------------------------------------------------------------


def test_par_de_cnots_leva_z0_a_z1():
    """CNOT_{1->0} CNOT_{0->1}, conjugando Z_0, dá Z_1 (imagem de Heisenberg).

    É a identidade que torna inerte a Rot final do qubit 0 e que limita a
    frequência em x_1 do re-uploading a R - 1.
    """
    u_ent = qml.matrix(
        qml.prod(qml.CNOT(wires=[1, 0]), qml.CNOT(wires=[0, 1])), wire_order=[0, 1]
    )
    z0 = qml.matrix(qml.PauliZ(0), wire_order=[0, 1])
    z1 = qml.matrix(qml.PauliZ(1), wire_order=[0, 1])
    assert np.allclose(u_ent.conj().T @ z0 @ u_ent, z1)


@pytest.mark.parametrize(
    "encoding, kwargs",
    [("angle", None), ("reuploading", {"R": 2}), ("reuploading", {"R": 3})],
)
def test_rot_final_do_qubit_0_e_inerte(encoding, kwargs):
    """Os 3 parâmetros da última Rot no qubit 0 têm derivada identicamente nula.

    Com pesos ALEATÓRIOS e vários pontos, para não depender de um caso
    particular; e mudar esses parâmetros não muda a saída. Os demais
    parâmetros da última camada, no qubit 1, não são inertes.
    """
    clf = model.build(encoding, n_features=2, L_var=2, enc_kwargs=kwargs)
    rng = np.random.default_rng(7)
    X = pnp.array(rng.uniform(0, np.pi, size=(20, 2)), requires_grad=False)
    w = pnp.array(rng.normal(0.0, 1.0, size=clf.weights_shape), requires_grad=True)

    gradiente = qml.grad(lambda pesos: pnp.sum(clf.circuit(X, pesos, None) ** 2))(w)

    assert np.allclose(gradiente[-1, 0, :], 0.0, atol=1e-12)
    assert np.abs(gradiente[-1, 1, :]).max() > 1e-3

    alterado = w.copy()
    alterado[-1, 0, :] += 1.234
    assert np.allclose(clf.circuit(X, w, None), clf.circuit(X, alterado, None), atol=1e-12)


def test_uma_camada_ignora_o_primeiro_atributo():
    """Com L_var = 1 (ou R = 1), a saída não depende de x_1: só do qubit 1.

    É por isso que, na tab_mesmo_p, k = 1 fica no acaso no xor.
    """
    clf = model.build("angle", n_features=2, L_var=1)
    rng = np.random.default_rng(3)
    w = pnp.array(rng.normal(0.0, 1.0, size=clf.weights_shape), requires_grad=False)
    x2 = rng.uniform(0, np.pi, size=10)
    for x1 in (0.1, 1.3, 2.9):
        X = pnp.array(np.column_stack([np.full(10, x1), x2]), requires_grad=False)
        X_ref = pnp.array(np.column_stack([np.zeros(10), x2]), requires_grad=False)
        assert np.allclose(clf.circuit(X, w, None), clf.circuit(X_ref, w, None), atol=1e-12)
