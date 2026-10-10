"""Linha de comando do projeto.

Cada experimento do trabalho é um subcomando:

    python -m tccqml listar
    python -m tccqml treinar --encoding amplitude --dataset moons
    python -m tccqml treinar --encoding reuploading --dataset moons --R 3
    python -m tccqml treinar --encoding angle --dataset xor --ansatz local
    python -m tccqml ablacao
    python -m tccqml varredura --parametro L_var
    python -m tccqml diagnostico
    python -m tccqml verificacoes
    python -m tccqml espectro
    python -m tccqml comparar
    python -m tccqml sensibilidade
    python -m tccqml varredura
    python -m tccqml tabelas
    python -m tccqml figuras

Nada é hardcoded aqui: as opções de `--encoding` saem de `ENCODINGS` e as de
`--dataset` de `_GENERATORS`. Registrar uma codificação nova basta para ela
aparecer na ajuda e poder entrar na grade — não existe uma segunda lista para
manter em sincronia, e é por isso que `Encoding.descricao` existe.

Toda execução de `treinar` imprime o bloco de custo do circuito junto da
acurácia, porque a comparação do trabalho é entre os dois.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
    t.add_argument(
        "--R",
        type=int,
        default=PADRAO.R,
        dest="R",
        help="repetições do data re-uploading",
    )
    t.add_argument(
        "--L-var", type=int, default=PADRAO.L_var, dest="L_var", help="camadas do ansatz"
    )
    t.add_argument("--epocas", type=int, default=PADRAO.epocas)
    t.add_argument("--eta", type=float, default=PADRAO.eta, help="taxa de aprendizado do Adam")
    t.add_argument("--seed", type=int, default=PADRAO.seed)
    t.add_argument("--n-samples", type=int, default=PADRAO.n_samples, dest="n_samples")
    t.add_argument(
        "--salvar",
        action="store_true",
        help="grava o histórico e o resumo em CSV e os pesos da época escolhida em .npz",
    )

    c = sub.add_parser("comparar", help="roda a grade completa: 4 codificações x 3 datasets x 5 sementes")
    c.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))
    c.add_argument("--encodings", nargs="+", default=list(PADRAO.encodings_grade))
    c.add_argument("--datasets", nargs="+", default=list(PADRAO.datasets_grade))

    sens = sub.add_parser(
        "sensibilidade",
        help="verificação do eta: varre eta_candidatos e escolhe pela validação",
    )
    sens.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    v = sub.add_parser(
        "varredura",
        help="varre as repetições R do re-uploading ou, como controle, as camadas do angle",
    )
    v.add_argument(
        "--parametro",
        default="R",
        choices=["R", "L_var"],
        help="R: repetições R do re-uploading; L_var: camadas L_var do ansatz no angle",
    )
    v.add_argument(
        "--valores",
        type=int,
        nargs="+",
        default=None,
        help="valores varridos (padrão: 1 2 3 4 5)",
    )
    v.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    a = sub.add_parser("ablacao", help="ablação: ansatz com e sem CNOTs")
    a.add_argument(
        "--datasets",
        nargs="+",
        default=list(PADRAO.datasets_grade),
        choices=sorted(_GENERATORS),
    )
    a.add_argument("--encoding", default="angle", choices=sorted(ENCODINGS))
    a.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    d = sub.add_parser(
        "diagnostico", help="amplitude encoding em [0,pi] contra [-1,1]"
    )
    d.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    vf = sub.add_parser(
        "verificacoes",
        help="controle linear nas 9 funções de Fourier e limiar de g no circles",
    )
    vf.add_argument("--sementes", type=int, nargs="+", default=list(PADRAO.sementes))

    e = sub.add_parser("espectro", help="Omega medido por FFT, contra o suporte previsto no Cap. 2")
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
        "val_size",
        "test_size",
        "L_var",
        "epocas",
        "batch_size",
        "eta",
        "eta_candidatos",
        "sementes",
        "shots",
    ):
        print(f"  {campo:14s} {getattr(PADRAO, campo)}")


def _treinar(args) -> None:
    ds = load_dataset(
        args.dataset,
        n_samples=args.n_samples,
        noise=PADRAO.noise,
        val_size=PADRAO.val_size,
        test_size=PADRAO.test_size,
        seed=args.seed,
    )
    enc_kwargs = {"R": args.R} if args.encoding == "reuploading" else None
    clf = model.build(
        args.encoding,
        n_features=ds.n_features,
        L_var=args.L_var,
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
        eta=args.eta,
        seed=args.seed,
        verbose=True,
    )
    print(
        f"\nacurácia (época {r.epoca_escolhida})  treino={r.acc_treino:.3f}  "
        f"val={r.acc_val:.3f}  teste={r.acc_teste:.3f}"
    )

    if args.salvar:
        from tccqml.experiments import _salvar_pesos

        nome = nome_treino(args)
        metrics = Path(args.out) / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        r.historico.to_csv(metrics / f"{nome}.csv", index=False)
        # Uma linha com o modelo da época escolhida: é o que a fronteira de
        # decisão precisa para ser desenhada sem retreinar.
        resumo = {
            **r.meta,
            "R": args.R if args.encoding == "reuploading" else None,
            "n_samples": args.n_samples,
            "acc_treino": r.acc_treino,
            "acc_val": r.acc_val,
            "acc_teste": r.acc_teste,
            "epoca_escolhida": r.epoca_escolhida,
        }
        pd.DataFrame([resumo]).to_csv(metrics / f"{nome}_resumo.csv", index=False)
        _salvar_pesos(Path(args.out) / "weights", nome, r)
        print(f"histórico salvo em {metrics / f'{nome}.csv'}")
        print(f"resumo salvo em {metrics / f'{nome}_resumo.csv'}")
        print(f"pesos salvos em {Path(args.out) / 'weights' / f'{nome}.npz'}")


def nome_treino(args) -> str:
    """Nome dos arquivos de `treinar --salvar`.

    Com o protocolo padrão é `treino_<encoding>_<dataset>_<seed>`; cada opção
    que foge de `PADRAO` acrescenta um sufixo, para que dois treinos diferentes
    não se sobrescrevam.
    """
    nome = f"treino_{args.encoding}_{args.dataset}_{args.seed}"
    if args.ansatz != PADRAO.ansatz:
        nome += f"_{args.ansatz}"
    if args.L_var != PADRAO.L_var:
        nome += f"_Lvar{args.L_var}"
    if args.encoding == "reuploading" and args.R != PADRAO.R:
        nome += f"_R{args.R}"
    if args.eta != PADRAO.eta:
        nome += f"_eta{args.eta:g}"
    if args.epocas != PADRAO.epocas:
        nome += f"_ep{args.epocas}"
    if args.n_samples != PADRAO.n_samples:
        nome += f"_n{args.n_samples}"
    return nome


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
    elif args.comando == "sensibilidade":
        from tccqml.experiments import rodar_sensibilidade_eta

        rodar_sensibilidade_eta(out=args.out, sementes=tuple(args.sementes))
    elif args.comando == "varredura":
        from tccqml.experiments import rodar_varredura_L_var, rodar_varredura_R

        valores = tuple(args.valores) if args.valores else None
        runner = rodar_varredura_R if args.parametro == "R" else rodar_varredura_L_var
        runner(out=args.out, valores=valores, sementes=tuple(args.sementes))
    elif args.comando == "ablacao":
        from tccqml.experiments import rodar_ablacao

        rodar_ablacao(
            out=args.out,
            datasets=tuple(args.datasets),
            encoding=args.encoding,
            sementes=tuple(args.sementes),
        )
    elif args.comando == "diagnostico":
        from tccqml.experiments import diagnostico_amplitude

        diagnostico_amplitude(out=args.out, sementes=tuple(args.sementes))
    elif args.comando == "verificacoes":
        from tccqml.verificacoes import rodar_verificacoes

        rodar_verificacoes(out=args.out, sementes=tuple(args.sementes))
    elif args.comando == "espectro":
        from tccqml.espectro import tabela_espectro

        tabela = tabela_espectro(valores_R=tuple(args.valores))
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
