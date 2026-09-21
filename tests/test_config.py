"""Testes do protocolo experimental.

O protocolo é a fonte fiel para escrever o Capítulo 3. Estes testes garantem
que o que está escrito lá é o que de fato roda.
"""

import numpy as np

from tccqml.config import PADRAO, Protocolo
from tccqml.data import load_dataset
from tccqml.embeddings import ENCODINGS


def test_n_samples_bate_com_a_secao_2_5_7():
    """A divergência que o config.py fechou: o código dizia 200, o texto diz 300."""
    assert PADRAO.n_samples == 300


def test_load_dataset_usa_o_protocolo_por_padrao():
    ds = load_dataset("moons")
    n_treino = round(PADRAO.n_samples * (1 - PADRAO.test_size))

    assert len(ds.X_train) == n_treino
    assert len(ds.X_train) + len(ds.X_test) == PADRAO.n_samples


def test_normalizacao_padrao_e_0_pi():
    ds = load_dataset("moons")

    assert np.isclose(ds.X_train.min(), 0.0)
    assert np.isclose(ds.X_train.max(), np.pi)


def test_ansatz_congelado_no_protocolo():
    """Seção 2.2.3.4: mesma camada, mesmo número de camadas, em tudo."""
    assert PADRAO.ansatz == "strongly_entangling"
    assert PADRAO.n_layers == 2


def test_grade_so_referencia_codificacoes_registradas():
    """Se a grade citasse um nome não registrado, ela quebraria no meio."""
    assert set(PADRAO.encodings_grade) <= set(ENCODINGS)


def test_grade_cobre_as_quatro_da_etapa_9():
    assert set(PADRAO.encodings_grade) == {"angle", "amplitude", "reuploading", "zz"}


def test_protocolo_e_imutavel():
    """Congelado é congelado: mudar no meio de uma grade invalidaria a comparação."""
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        PADRAO.epocas = 5


def test_batch_efetivo_deriva_do_protocolo():
    p = Protocolo(n_samples=300, test_size=0.3, batch_size=20)

    assert p.batch_efetivo == 21
