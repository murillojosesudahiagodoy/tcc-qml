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
    """Se o custo não cai ao longo das épocas, o modelo não está aprendendo."""
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

    assert list(r.historico.columns) == ["epoca", "custo", "acc_treino", "acc_val"]
    assert len(r.historico) == 3
    assert r.meta["encoding"] == "angle"
    assert r.meta["dataset"] == "moons"
    assert r.meta["n_params_circuito"] == 12
    assert r.meta["n_params"] == 13


def test_meta_separa_p_do_circuito_dos_parametros_do_modelo():
    """O CSV precisa gravar os dois contadores: o texto fala em p = 12, não 13."""
    ds = load_dataset("moons", n_samples=60, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=1, seed=42, verbose=False)

    assert r.meta["n_params_circuito"] + 1 == r.meta["n_params"]
    assert r.meta["n_params_encoding"] == 0


def test_historico_nao_tem_coluna_de_teste():
    """O teste é usado uma vez, no fim — nunca época a época."""
    ds = load_dataset("moons", n_samples=60, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=3, batch_size=20, seed=42, verbose=False)

    assert not [c for c in r.historico.columns if "test" in c]


def test_epoca_escolhida_e_a_de_maior_acc_val():
    """A validação escolhe a época; no empate fica a mais antiga."""
    ds = load_dataset("moons", n_samples=80, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=8, batch_size=20, lr=0.2, seed=42, verbose=False)
    hist = r.historico
    melhor = hist.loc[hist["acc_val"].idxmax()]  # idxmax devolve o primeiro máximo

    assert r.epoca_escolhida == int(melhor["epoca"])
    assert r.acc_val == pytest.approx(hist["acc_val"].max())
    assert r.acc_treino == pytest.approx(melhor["acc_treino"])


def test_parametros_devolvidos_sao_os_da_epoca_escolhida():
    """acc_treino, acc_val e acc_teste saem todos do mesmo modelo restaurado."""
    ds = load_dataset("moons", n_samples=80, seed=42)
    clf = model.build("angle", n_features=2, n_layers=2)

    r = treinar(clf, ds, epocas=8, batch_size=20, lr=0.2, seed=42, verbose=False)

    def acc(X, y):
        return accuracy(y, model.prever(clf, r.weights, r.alpha, r.bias, X))

    assert acc(ds.X_train, ds.y_train) == pytest.approx(r.acc_treino)
    assert acc(ds.X_val, ds.y_val) == pytest.approx(r.acc_val)
    assert acc(ds.X_test, ds.y_test) == pytest.approx(r.acc_teste)
