# VersionRAG

Version-aware retrieval-augmented generation for evolving documents.

A versioned corpus holds the same fact in several mutually contradicting states, and the version
identifier is too weak a signal for embedding similarity to separate them. Asked about one
version, a standard RAG system retrieves all of them — they are near-duplicates and score alike —
and answers from whichever the retriever ranked first. Worse, some facts live in no single
document: *"in which version was `strings.Lines` added?"* is a property of the boundary between
two versions, and no context length recovers it.

VersionRAG indexes a corpus as an explicit **version lineage** with **materialised change facts**,
and routes each query by intent.

Reference implementation for *VersionRAG: Version-Aware Retrieval-Augmented Generation for
Evolving Documents* (Findings of EACL 2027).

## Install

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker for the two stores.

```bash
uv sync
cp .env.example .env     # set OPENAI_API_KEY, NEO4J_USER, NEO4J_PASSWORD
docker compose up -d     # Neo4j + Qdrant
```

## Use

```bash
uv run versionrag index  --corpus path/to/corpus
uv run versionrag query  "What changed in Docker Compose 2.38.1?"
```

See `src/versionrag/main.py` for the full CLI.

## How it works

Indexing builds a typed graph over five node types — Category, Document, Version, Content and
Change — where `NEXT` edges carry the version order and Change nodes hold the diff between each
consecutive pair. Content chunks live in Qdrant with `(document, version)` as filterable payload;
the graph lives in Neo4j.

At query time a parser resolves the question to a `(document, version)` pair and an intent, and
the router sends it down one of three paths:

| Intent | Path |
|---|---|
| Content retrieval | vector search **filtered** on `(document, version)` |
| Version listing | enumeration over Version nodes |
| Change retrieval | lookup in Change nodes for the relevant version pair |

Implicit changes — facts stated in no document — exist only because they are materialised at index
time. That is what the change layer buys.

```
src/versionrag/
  indexing/     pipeline, graph builder, attribute and change extraction, diff filtering
  retrieval/    query parser, router, graph and vector retrievers, fusion
  generation/   answer generation and prompts
  clients/      Neo4j, Qdrant, LLM, embeddings, usage tracking
  models/       schemas and enums
```

## Datasets

The two benchmarks are on the Hub, with corpora, evaluation sets and pre-chunked indexes:

- [VersionQA-Software](https://huggingface.co/datasets/VersionRAG/VersionQA-Software) — 869 questions over 882 versions of open-source documentation
- [VersionQA-Legal](https://huggingface.co/datasets/VersionRAG/VersionQA-Legal) — 547 questions over 355 point-in-time versions of amended UK legislation

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

The bachelor-thesis prototype this grew out of is tagged [`v1-thesis`](../../tree/v1-thesis).

## Licence

MIT. See [LICENSE](LICENSE).
