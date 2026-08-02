"""Testes do módulo de dados."""

import numpy as np
import pytest

from tccqml.data import load_dataset

NOMES = ["xor", "moons", "circles"]


@pytest.mark.parametrize("nome", NOMES)
def test_formato_e_divisao(nome):
    ds = load_dataset(nome, n_samples=200, test_size=0.3)

    assert ds.n_features == 2
    assert len(ds.X_train) == 140
    assert len(ds.X_test) == 60
    assert len(ds.y_train) == len(ds.X_train)


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


def test_dataset_desconhecido_falha():
    with pytest.raises(ValueError, match="desconhecido"):
        load_dataset("iris")