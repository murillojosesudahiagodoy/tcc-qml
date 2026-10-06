"""Gera `tcc_qml_executado.ipynb`, a cópia de leitura de `tcc_qml.ipynb`.

A cópia é o notebook interativo executado de cima a baixo com RODAR = False,
RODAR_TESTES = False e USAR_WIDGETS = False (widgets não aparecem no GitHub; as
células alternativas, sim), com todas as saídas gravadas. Lê os resultados
versionados em results/ e não roda nenhum experimento; só o treino de exemplo
da seção 3, que leva segundos e grava em results/notebook/.

Uso, da raiz do repositório e com o ambiente ativado:

    python notebooks/gerar_executado.py

Rode de novo sempre que tcc_qml.ipynb ou os resultados mudarem.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PASTA = Path(__file__).resolve().parent
ORIGEM = PASTA / "tcc_qml.ipynb"
DESTINO = PASTA / "tcc_qml_executado.ipynb"

PARAMETROS = {"RODAR": "False", "RODAR_TESTES": "False", "USAR_WIDGETS": "False"}

AVISO = """\
> **Versão executada, só para leitura.** Gerada a partir de
> [`tcc_qml.ipynb`](tcc_qml.ipynb) por `python notebooks/gerar_executado.py`, com
> `RODAR = False`, `RODAR_TESTES = False` e `USAR_WIDGETS = False`: nenhum experimento roda
> aqui, as tabelas e figuras são as de `results/`, e os painéis dão lugar às células
> alternativas. Para rodar ou mexer, abra o `tcc_qml.ipynb`. Não edite este arquivo: ele é
> sobrescrito a cada geração."""


def fixar_parametros(nb) -> None:
    celula = next(c for c in nb.cells if c.get("id") == "c02")
    fonte = celula.source
    for nome, valor in PARAMETROS.items():
        fonte, n = re.subn(
            rf"^{nome}(\s*)=\s*\w+", rf"{nome}\g<1>= {valor}", fonte, flags=re.MULTILINE
        )
        if n != 1:
            raise SystemExit(f"parâmetro {nome} não encontrado na célula de parâmetros (c02)")
    celula.source = fonte


def caminhos_pessoais(nb) -> list[str]:
    # Pedaços de caminho absoluto que não podem aparecer nas saídas.
    proibidos = {
        str(PASTA.parent),
        PASTA.parent.as_posix(),
        str(Path.home()),
        Path.home().as_posix(),
    }
    proibidos |= {p.replace("\\", "\\\\") for p in proibidos}
    achados = []
    for i, celula in enumerate(nb.cells):
        texto = json.dumps(celula.get("outputs", []), ensure_ascii=False)
        for p in proibidos:
            if p in texto:
                achados.append(f"célula {i} ({celula.get('id')}): contém {p!r}")
    return achados


def sem_repr_redundante(nb) -> None:
    # O text/plain ao lado de uma tabela ou figura é só o repr do objeto
    # (`<... at 0x...>`), que muda a cada execução e sujaria o diff.
    for celula in nb.cells:
        for saida in celula.get("outputs", []):
            dados = saida.get("data", {})
            if "text/plain" in dados and ("text/html" in dados or "image/png" in dados):
                del dados["text/plain"]


def juntar_streams(nb) -> None:
    # O kernel às vezes manda o fim de um print numa mensagem à parte, e a mesma saída
    # viraria um ou dois blocos conforme a execução; junta os blocos seguidos do mesmo canal.
    for celula in nb.cells:
        saidas = []
        for saida in celula.get("outputs", []):
            anterior = saidas[-1] if saidas else None
            if (
                saida.get("output_type") == "stream"
                and anterior is not None
                and anterior.get("output_type") == "stream"
                and anterior["name"] == saida["name"]
            ):
                anterior["text"] += saida["text"]
            else:
                saidas.append(saida)
        if "outputs" in celula:
            celula["outputs"] = saidas


def sem_tempo_de_execucao(nb) -> None:
    # O "[ok em N s]" de cada subcomando varia de uma execução para outra; na cópia vira
    # "[ok]". Os tempos de treino da coluna `segundos` dos CSVs não passam por aqui.
    for celula in nb.cells:
        for saida in celula.get("outputs", []):
            if saida.get("output_type") == "stream":
                saida["text"] = re.sub(r"\[ok em \d+ s\]", "[ok]", saida["text"])


def main() -> None:
    nb = nbformat.read(ORIGEM, as_version=4)
    fixar_parametros(nb)
    NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(PASTA)}},
    ).execute()
    sem_repr_redundante(nb)
    juntar_streams(nb)
    sem_tempo_de_execucao(nb)

    nb.cells.insert(0, nbformat.v4.new_markdown_cell(AVISO, id="executado"))
    # O nbstripout respeita esta chave e não apaga as saídas deste arquivo.
    nb.metadata["keep_output"] = True

    achados = caminhos_pessoais(nb)
    if achados:
        raise SystemExit(
            "Saídas com caminho absoluto; troque por caminho relativo no tcc_qml.ipynb:\n  "
            + "\n  ".join(achados)
        )
    nbformat.validate(nb)
    nbformat.write(nb, DESTINO)
    print(f"gravado {DESTINO.relative_to(PASTA.parent).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
