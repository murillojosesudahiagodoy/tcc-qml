"""Testes do módulo de dados."""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from tccqml.config import PADRAO
from tccqml.data import load_dataset

NOMES = ["xor", "moons", "circles"]


@pytest.mark.parametrize("nome", NOMES)
def test_formato_e_divisao(nome):
    ds = load_dataset(nome, n_samples=200, val_size=0.2, test_size=0.2)

    assert ds.n_features == 2
    assert len(ds.X_train) == 120
    assert len(ds.X_val) == 40
    assert len(ds.X_test) == 40
    assert len(ds.y_train) == len(ds.X_train)
    assert len(ds.y_val) == len(ds.X_val)
    assert len(ds.y_test) == len(ds.X_test)


@pytest.mark.parametrize("nome", NOMES)
def test_split_padrao_e_180_60_60(nome):
    """60/20/20 do total com N = 300, como o Capítulo 3 declara."""
    ds = load_dataset(nome)

    assert (len(ds.X_train), len(ds.X_val), len(ds.X_test)) == (180, 60, 60)


@pytest.mark.parametrize("nome", NOMES)
def test_conjuntos_disjuntos_e_completos(nome):
    """Nenhuma amostra aparece em dois conjuntos, e nenhuma se perde.

    Comparado ANTES da normalização: depois dela os três conjuntos passam pelo
    mesmo scaler, que é injetivo, então a disjunção também vale nos dados
    normalizados — mas é nos dados brutos que a pergunta faz sentido.
    """
    ds = load_dataset(nome, feature_range=(0.0, 1.0))
    linhas = [tuple(np.round(x, 12)) for x in (*ds.X_train, *ds.X_val, *ds.X_test)]

    assert len(linhas) == PADRAO.n_samples
    assert len(set(linhas)) == PADRAO.n_samples


@pytest.mark.parametrize("nome", NOMES)
def test_estratificacao_preserva_as_classes(nome):
    """A proporção de classes dos três conjuntos acompanha a do total."""
    ds = load_dataset(nome)
    total = np.concatenate([ds.y_train, ds.y_val, ds.y_test]).mean()

    for y in (ds.y_train, ds.y_val, ds.y_test):
        # Com 60 amostras, uma amostra de diferença é 1/60 da fração.
        assert abs(y.mean() - total) <= 1 / len(y) + 1e-12


def test_scaler_ajustado_so_no_treino(monkeypatch):
    """Ajustar o scaler em validação ou teste vazaria informação deles."""
    from tccqml import data

    vistos = []

    class ScalerEspiao(MinMaxScaler):
        def fit(self, X, y=None):
            vistos.append(len(X))
            return super().fit(X, y)

    monkeypatch.setattr(data, "MinMaxScaler", ScalerEspiao)
    ds = data.load_dataset("moons")

    assert vistos == [len(ds.X_train)]
    # Ajustado no treino: só ele toca exatamente as bordas do intervalo.
    assert np.allclose(ds.X_train.min(axis=0), 0.0)
    assert np.allclose(ds.X_train.max(axis=0), np.pi)


@pytest.mark.parametrize("nome", NOMES)
def test_treino_normalizado_no_intervalo(nome):
    ds = load_dataset(nome, feature_range=(0.0, np.pi))

    assert np.isclose(ds.X_train.min(), 0.0)
    assert np.isclose(ds.X_train.max(), np.pi)


@pytest.mark.parametrize("nome", NOMES)
def test_classes_balanceadas(nome):
    """Classe desbalanceada faz acurácia mentir: 90% acertando só a maioria."""
    ds = load_dataset(nome, n_samples=200)
    fracao = ds.y_train.mean()

    assert 0.4 <= fracao <= 0.6


@pytest.mark.parametrize("nome", NOMES)
def test_reprodutibilidade(nome):
    a = load_dataset(nome, seed=7)
    b = load_dataset(nome, seed=7)

    assert np.allclose(a.X_train, b.X_train)
    assert np.array_equal(a.y_train, b.y_train)


def test_rotulos_pm1():
    ds = load_dataset("moons")

    assert set(np.unique(ds.y_train)) == {0, 1}
    assert set(np.unique(ds.y_train_pm1)) == {-1, 1}
    assert set(np.unique(ds.y_val_pm1)) == {-1, 1}


def test_dataset_desconhecido_falha():
    with pytest.raises(ValueError, match="desconhecido"):
        load_dataset("iris")