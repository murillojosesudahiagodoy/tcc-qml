# TCC — Quantum Machine Learning

Repositório de código, experimentos e resultados do Trabalho de Conclusão de Curso.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
pytest
```

## Estrutura

```
src/tccqml/     código reutilizável (circuitos, embeddings, treino, métricas)
tests/          testes automatizados (pytest)
notebooks/      exploração e geração de figuras
data/raw/       datasets originais (não versionados)
data/processed/ datasets tratados (não versionados)
results/        figuras e métricas geradas (não versionadas)
docs/           anotações, referências, rascunhos do texto
```

## Convenção de commits

`tipo: descrição curta no imperativo`

- `feat:` nova funcionalidade / experimento
- `fix:` correção
- `exp:` rodada de experimento
- `docs:` texto do TCC, anotações
- `test:` testes
- `chore:` infraestrutura
