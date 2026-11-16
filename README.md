# VersionRAG

Version-aware retrieval-augmented generation for evolving documents.

Reference implementation for *VersionRAG: Version-Aware Retrieval-Augmented Generation for
Evolving Documents* (Findings of EACL 2027).

## Install

Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
uv sync
cp .env.example .env     # OPENAI_API_KEY, NEO4J_USER, NEO4J_PASSWORD
docker compose up -d     # Neo4j + Qdrant
```

## Use

```bash
uv run versionrag index --corpus path/to/corpus
uv run versionrag query "What changed in Docker Compose 2.38.1?"
```

## How it works

Indexing builds a typed graph over five node types: Category, Document, Version, Content and Change. `NEXT` edges carry the version order; Change nodes hold the diff between each consecutive
pair. Chunks live in Qdrant with `(document, version)` as filterable payload, the graph in Neo4j.

At query time a parser resolves the question to a `(document, version)` pair and an intent, and
the router picks a path:

| Intent | Path |
|---|---|
| Content retrieval | vector search filtered on `(document, version)` |
| Version listing | enumeration over Version nodes |
| Change retrieval | lookup in Change nodes for the version pair |

```
src/versionrag/
  indexing/     pipeline, graph builder, attribute and change extraction, diff filtering
  retrieval/    query parser, router, graph and vector retrievers, fusion
  generation/   answer generation and prompts
  clients/      Neo4j, Qdrant, LLM, embeddings, usage tracking
  models/       schemas and enums
```

## Datasets

- [VersionQA-Software](https://huggingface.co/datasets/VersionRAG/VersionQA-Software): 869 questions, 882 versions of open-source documentation
- [VersionQA-Legal](https://huggingface.co/datasets/VersionRAG/VersionQA-Legal): 547 questions, 355 versions of amended UK legislation

## Citation

```bibtex
@inproceedings{stuhlmann2027versionrag,
  title     = {{VersionRAG}: Version-Aware Retrieval-Augmented Generation for Evolving Documents},
  author    = {Stuhlmann, Linus and Mathitharan, Kajaluxan and Huwiler, Daniel and
               Stockinger, Kurt and F{\"u}rst, Jonathan},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

## Licence

MIT. The bachelor-thesis prototype this grew out of is tagged [`v1-thesis`](../../tree/v1-thesis).
