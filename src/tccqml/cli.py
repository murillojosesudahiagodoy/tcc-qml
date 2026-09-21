"""Linha de comando do projeto (T13).

Trocar de codificação deixa de exigir caçar strings no notebook:

    python -m tccqml listar
    python -m tccqml treinar --encoding amplitude --dataset moons
    python -m tccqml treinar --encoding reuploading --dataset moons --L-reup 3
    python -m tccqml treinar --encoding angle --dataset xor --ansatz local
    python -m tccqml ablacao
    python -m tccqml diagnostico
    python -m tccqml espectro
    python -m tccqml comparar
    python -m tccqml varredura
    python -m tccqml tabelas
    python -m tccqml figuras

Nada é hardcoded aqui: as opções de `--encoding` saem de `ENCODINGS` e as de
`--dataset` de `_GENERATORS`. Registrar uma codificação nova basta para ela
aparecer na ajuda e poder entrar na grade — não existe uma segunda lista para
manter em sincronia, e é por isso que `Encoding.descricao` existe.

Toda execução de `treinar` imprime o bloco de custo do circuito junto da
acurácia: desempenho e custo lado a lado é a tese do trabalho.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tccqml import model
from tccqml.ansatz import ANSATZE
from tccqml.circuit_stats import formatar, stats
from tccqml.config import PADRAO
from tccqml.data import _GENERATORS, load_dataset
from tccqml.embeddings import ENCODINGS
from tccqml.train import treinar


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tccqml",
        description="Experimentos de codificação de dados do TCC em QML.",
    )
    sub = p.add_subparsers(dest="comando", required=True)

    sub.add_parser("listar", help="lista codificações, datasets e ansätze registrados")

    t = sub.add_parser("treinar", help="treina uma combinação e imprime acurácia + custo")
    t.add_argument("--encoding", default="angle", choices=sorted(ENCODINGS))
    t.add_argument("--dataset", default="moons", choices=sorted(_GENERATORS))
    t.add_argument("--ansatz", default=PADRAO.ansatz, choices=sorted(ANSATZE))
    t.add_argument("--L-reup", type=int, default=PADRAO.L_reup, dest="L_reup")
    t.add_argument("--n-layers", type=int, default=PADRAO.n_layers, dest="n_layers")
    t.add_argument("--epocas", type=int, default=PADRAO.epocas)
    t.add_argument("--lr", type=float, default=PADRAO.lr)
    t.add_argument("--seed", type=int, default=PADRAO.seed)
    t.add_argument("--n-samples", type=int, default=PADRAO.n_samples, dest="n_samples")
    t.add_argument("--salvar", action="store_true", help="grava o histórico em CSV")

    c = sub.add_parser("comparar", help="roda a grade completa: 4 codificações x 3 datasets x 5 sementes")
    c.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))
    c.add_argument("--encodings", nargs="+", default=list(PADRAO.encodings_grade))
    c.add_argument("--datasets", nargs="+", default=list(PADRAO.datasets_grade))

    v = sub.add_parser("varredura", help="varre o número de blocos L do re-uploading")
    v.add_argument(
        "--parametro", default="L-reup", choices=["L-reup"], help="parâmetro varrido"
    )
    v.add_argument("--valores", type=int, nargs="+", default=list(PADRAO.L_reup_varredura))
    v.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    a = sub.add_parser("ablacao", help="ablação: ansatz com e sem CNOTs no XOR")
    a.add_argument("--dataset", default="xor", choices=sorted(_GENERATORS))
    a.add_argument("--encoding", default="angle", choices=sorted(ENCODINGS))
    a.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    d = sub.add_parser(
        "diagnostico", help="amplitude encoding em [0,pi] contra [-1,1]"
    )
    d.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    e = sub.add_parser("espectro", help="Omega medido por FFT, contra a Tabela 5 do texto")
    e.add_argument("--valores", type=int, nargs="+", default=[1, 2, 3])

    sub.add_parser("tabelas", help="gera as tabelas .csv e .tex a partir dos CSVs")
    sub.add_parser("figuras", help="gera as figuras em PDF a partir dos CSVs")

    p.add_argument("--out", default=PADRAO.out, help="diretório de saída (default: results)")
    return p


def _listar() -> None:
    print("Codificações (--encoding):")
    for nome, enc in sorted(ENCODINGS.items()):
        print(f"  {nome:14s} {enc.descricao}")
    print("\nDatasets (--dataset):")
    for nome in sorted(_GENERATORS):
        print(f"  {nome}")
    print("\nAnsätze (--ansatz):")
    for nome, ans in sorted(ANSATZE.items()):
        print(f"  {nome:22s} {ans.descricao}")
    print("\nProtocolo congelado (config.Protocolo):")
    for campo in (
        "n_samples",
        "noise",
        "test_size",
        "n_layers",
        "epocas",
        "batch_size",
        "lr",
        "sementes",
        "shots",
    ):
        print(f"  {campo:14s} {getattr(PADRAO, campo)}")


def _treinar(args) -> None:
    ds = load_dataset(
        args.dataset,
        n_samples=args.n_samples,
        noise=PADRAO.noise,
        test_size=PADRAO.test_size,
        seed=args.seed,
    )
    enc_kwargs = {"L_reup": args.L_reup} if args.encoding == "reuploading" else None
    clf = model.build(
        args.encoding,
        n_features=ds.n_features,
        n_layers=args.n_layers,
        ansatz=args.ansatz,
        enc_kwargs=enc_kwargs,
    )

    print(f"codificação: {args.encoding}   dataset: {args.dataset}   ansatz: {args.ansatz}")
    print("custo do circuito:")
    print(
        formatar(
            stats(clf, ds.X_train[0], batch_size=PADRAO.batch_efetivo, shots=PADRAO.shots)
        )
    )
    print("\ntreino:")
    r = treinar(
        clf,
        ds,
        epocas=args.epocas,
        batch_size=PADRAO.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )
    print(f"\nacurácia  treino={r.acc_treino:.3f}  teste={r.acc_teste:.3f}")

    if args.salvar:
        destino = Path(args.out) / "metrics"
        destino.mkdir(parents=True, exist_ok=True)
        nome = f"treino_{args.encoding}_{args.dataset}_{args.seed}.csv"
        r.historico.to_csv(destino / nome, index=False)
        print(f"histórico salvo em {destino / nome}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.comando == "listar":
        _listar()
    elif args.comando == "treinar":
        _treinar(args)
    elif args.comando == "comparar":
        from tccqml.experiments import rodar_grade

        rodar_grade(
            out=args.out,
            encodings=tuple(args.encodings),
            datasets=tuple(args.datasets),
            sementes=tuple(args.sementes),
        )
    elif args.comando == "varredura":
        from tccqml.experiments import rodar_varredura_L

        rodar_varredura_L(
            out=args.out, valores=tuple(args.valores), sementes=tuple(args.sementes)
        )
    elif args.comando == "ablacao":
        from tccqml.experiments import rodar_ablacao

        rodar_ablacao(
            out=args.out,
            dataset=args.dataset,
            encoding=args.encoding,
            sementes=tuple(args.sementes),
        )
    elif args.comando == "diagnostico":
        from tccqml.experiments import diagnostico_amplitude

        diagnostico_amplitude(out=args.out, sementes=tuple(args.sementes))
    elif args.comando == "espectro":
        from tccqml.espectro import tabela_espectro

        tabela = tabela_espectro(L_reups=tuple(args.valores))
        destino = Path(args.out) / "metrics"
        destino.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(destino / "espectro.csv", index=False)
        print(tabela.to_string(index=False))
        print(f"\nescrito em {destino / 'espectro.csv'}")
    elif args.comando == "tabelas":
        from tccqml.tabelas import gerar_todas

        gerar_todas(out=args.out)
    elif args.comando == "figuras":
        from tccqml.figuras import gerar_todas

        gerar_todas(out=args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
