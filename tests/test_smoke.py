"""Testes de fumaça: garantem que o ambiente está pronto para o TCC."""

import sys


def test_versao_python():
    assert sys.version_info >= (3, 11), "PennyLane 0.45 exige Python 3.11+"


def test_imports_principais():
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pennylane  # noqa: F401
    import sklearn  # noqa: F401


def test_pacote_local_importavel():
    import tccqml

    assert tccqml.__version__
