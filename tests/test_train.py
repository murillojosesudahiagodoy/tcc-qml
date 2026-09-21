"""Testes do laço de treinamento (Etapa 8)."""

import pytest

from tccqml import model
from tccqml.data import load_dataset
from tccqml.metrics import accuracy, square_loss
from tccqml.train import treinar


def test_square_loss_zera_na_predicao_perfeita():
    assert float(square_loss([1, -1, 1], [1.0, -1.0, 1.0])) == 0.0


def test_accuracy_conta_acertos():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_treino_reduz_o_custo():
    """O teste que importa: se o custo não cai, o modelo não está aprendendo."""
    ds = load_dataset("moons", n_samples=80, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=8, batch_size=20, lr=0.2, seed=42, verbose=False)

    assert r.historico["custo"].iloc[-1] < r.historico["custo"].iloc[0]


def test_treino_supera_o_acaso():
    ds = load_dataset("moons", n_samples=80, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=10, batch_size=20, lr=0.2, seed=42, verbose=False)

    assert r.acc_treino > 0.6


@pytest.mark.parametrize("encoding", ["angle", "amplitude", "reuploading", "zz"])
def test_todas_as_codificacoes_treinam(encoding):
    """A Etapa 9 exige que trocar a codificação seja trocar uma string."""
    ds = load_dataset("moons", n_samples=80, seed=42)
    clf = model.build(encoding, n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=8, batch_size=20, lr=0.2, seed=42, verbose=False)

    assert r.historico["custo"].iloc[-1] < r.historico["custo"].iloc[0]
    assert r.meta["encoding"] == encoding


def test_historico_e_metadados_completos():
    ds = load_dataset("moons", n_samples=60, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=3, batch_size=20, seed=42, verbose=False)

    assert list(r.historico.columns) == ["epoca", "custo", "acc_treino", "acc_teste"]
    assert len(r.historico) == 3
    assert r.meta["encoding"] == "angle"
    assert r.meta["dataset"] == "moons"
    assert r.meta["n_params_circuito"] == 12
    assert r.meta["n_params"] == 13


def test_meta_separa_p_do_circuito_dos_parametros_do_modelo():
    """Os CSVs antigos gravavam n_params=13 onde o texto fala em p=12."""
    ds = load_dataset("moons", n_samples=60, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=1, seed=42, verbose=False)

    assert r.meta["n_params_circuito"] + 1 == r.meta["n_params"]
    assert r.meta["n_params_encoding"] == 0
