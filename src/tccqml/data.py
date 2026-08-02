"""Datasets clássicos usados nos experimentos de codificação (Etapa 7)."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Dataset:
    """Dataset já dividido e normalizado, pronto para entrar no circuito."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def y_train_pm1(self) -> np.ndarray:
        """Rótulos em {-1, +1}, que é a imagem natural de <Z>."""
        return 2 * self.y_train - 1

    @property
    def y_test_pm1(self) -> np.ndarray:
        return 2 * self.y_test - 1


def make_xor(n_samples: int = 200, noise: float = 0.15, seed: int = 42):
    """XOR: duas classes em quadrantes opostos. Não separável linearmente."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    X = X + rng.normal(0.0, noise, size=X.shape)
    return X, y


_GENERATORS = {
    "xor": lambda n, noise, seed: make_xor(n, noise, seed),
    "moons": lambda n, noise, seed: make_moons(n_samples=n, noise=noise, random_state=seed),
    "circles": lambda n, noise, seed: make_circles(
        n_samples=n, noise=noise, factor=0.4, random_state=seed
    ),
}


def load_dataset(
    name: str,
    n_samples: int = 200,
    noise: float = 0.15,
    test_size: float = 0.3,
    seed: int = 42,
    feature_range: tuple[float, float] = (0.0, np.pi),
) -> Dataset:
    """Gera, divide e normaliza um dataset sintético.

    O scaler é ajustado SÓ no treino e aplicado ao teste. Ajustar no
    conjunto completo vaza informação do teste e infla a acurácia.

    O intervalo padrão [0, pi] é pensado para angle encoding: RY(theta)
    percorre <Z> de +1 a -1 nesse trecho sem dar a volta, então dois
    valores distintos de entrada nunca caem no mesmo estado.
    """
    if name not in _GENERATORS:
        raise ValueError(f"dataset desconhecido: {name!r}. Opções: {sorted(_GENERATORS)}")

    X, y = _GENERATORS[name](n_samples, noise, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = MinMaxScaler(feature_range=feature_range).fit(X_train)

    return Dataset(
        name=name,
        X_train=scaler.transform(X_train),
        X_test=scaler.transform(X_test),
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
    )