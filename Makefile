.PHONY: up down dev serve logs health start stop reset index evaluate score eval-all index-large evaluate-large score-large eval-large legal-serve legal-start legal-index legal-evaluate legal-score legal-eval meta-serve meta-start meta-index meta-evaluate meta-score meta-eval meta-legal-serve meta-legal-start meta-legal-index meta-legal-evaluate meta-legal-score meta-legal-eval naive-setup naive-legal-index naive-legal-run naive-legal naive-legal-verbatim-index parser-serve parser-start parser-index parser-snapshot parser-evaluate parser-score parser-eval parser-changes parser-snapshot-phase-a

# The VersionRAG run harness — index / evaluate / score / rebuild-changes. Shared
# by every mode (full, parser, metadata_filtered) and both corpora, so it lives
# with the system rather than under one dataset.
HARNESS := VersionRAG/harness

start: up
	@echo "Waiting for services..."
	@until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do printf '.'; sleep 2; done
	@until curl -sf http://localhost:7474 > /dev/null 2>&1; do printf '.'; sleep 2; done
	@echo " ready."
	$(MAKE) serve

stop:
	docker compose down

up:
	docker compose up -d

down:
	docker compose down

reset:
	docker compose down -v

dev:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src uv run uvicorn versionrag.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000

serve:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src uv run uvicorn versionrag.main:app --host 0.0.0.0 --port 8000

logs:
	docker compose logs -f

health:
	curl -s localhost:8000/health | python3 -m json.tool

index:
	PYTHONPATH=src uv run python scripts/index.py

evaluate:
	PYTHONPATH=src uv run python scripts/evaluate.py

score:
	PYTHONPATH=src uv run python scripts/score.py

eval-all: index evaluate score

# ---------------------------------------------------------------------------
# Large dataset evaluation — indexes large_dataset/raw/** recursively
# and writes results to large_dataset/results/.
# ---------------------------------------------------------------------------
index-large:
	PYTHONPATH=src uv run python $(HARNESS)/index_large.py

evaluate-large:
	PYTHONPATH=src uv run python $(HARNESS)/evaluate_large.py

score-large:
	PYTHONPATH=src uv run python $(HARNESS)/score_large.py

eval-large: index-large evaluate-large score-large

# ---------------------------------------------------------------------------
# UK Legal Iteration — BA thesis only. Targets the new corpus at
# large_dataset/legal/raw/ with a legal-specific attribute-extractor prompt
# (VERSIONRAG_EXTRACT_PROMPT) and writes results to large_dataset/legal/results/.
#
# Workflow:
#   Terminal 1:  make legal-start   (or  make legal-serve  if docker is up)
#   Terminal 2:  make legal-eval    (= legal-index + legal-evaluate + legal-score)
#
# The legal-serve target sets VERSIONRAG_EXTRACT_PROMPT so the API loads
# the legal prompt at startup — the variable is read once at module import,
# so it MUST be set before uvicorn starts. Restart the server when toggling
# between the software and legal corpora.
# ---------------------------------------------------------------------------
LEGAL_DATA_DIR    := large_dataset/legal/raw
LEGAL_TEST_DIR    := large_dataset/legal/test
LEGAL_RESULTS_DIR := large_dataset/legal/results
LEGAL_PROMPT      := large_dataset/legal/extract_prompt.txt

legal-serve:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src VERSIONRAG_EXTRACT_PROMPT=$(LEGAL_PROMPT) \
	  uv run uvicorn versionrag.main:app --host 0.0.0.0 --port 8000

legal-start: up
	@echo "Waiting for services..."
	@until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do printf '.'; sleep 2; done
	@until curl -sf http://localhost:7474 > /dev/null 2>&1; do printf '.'; sleep 2; done
	@echo " ready."
	$(MAKE) legal-serve

legal-index:
	@mkdir -p $(LEGAL_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/index_large.py \
	  --data-dir $(LEGAL_DATA_DIR)

legal-evaluate:
	@mkdir -p $(LEGAL_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/evaluate_large.py \
	  --input $(LEGAL_TEST_DIR)/evaluation_set.csv \
	  --output $(LEGAL_RESULTS_DIR)/evaluation_answers.csv

legal-score:
	PYTHONPATH=src uv run python $(HARNESS)/score_large.py \
	  --input $(LEGAL_RESULTS_DIR)/evaluation_answers.csv \
	  --output $(LEGAL_RESULTS_DIR)/evaluation_scored.csv \
	  --summary $(LEGAL_RESULTS_DIR)/evaluation_summary.txt

legal-eval: legal-index legal-evaluate legal-score
# --- end UK Legal block ---

# ---------------------------------------------------------------------------
# Metadata-Filtered RAG (Deterministic Version Control) — ablation baseline.
# Same VersionRAG architecture with the expensive LLM summary layer removed:
# metadata is derived deterministically from the path/filename (no LLM
# attribute call) and change extraction is skipped (no Change nodes).
# Selected via INDEX_MODE=metadata_filtered, set at server startup (read once
# at import, like VERSIONRAG_EXTRACT_PROMPT). Results go to *_metadata_filtered
# dirs so full-VersionRAG results are never clobbered.
#
# Workflow (software):  Terminal 1: make meta-start   Terminal 2: make meta-eval
# Workflow (legal):     Terminal 1: make meta-legal-start   Terminal 2: make meta-legal-eval
# Switching modes rewrites the Neo4j/Qdrant volumes — restart the server.
# ---------------------------------------------------------------------------
META_DATA_DIR    := large_dataset/raw
META_TEST_DIR    := large_dataset/test
META_RESULTS_DIR := large_dataset/results_metadata_filtered

meta-serve:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src INDEX_MODE=metadata_filtered \
	  uv run uvicorn versionrag.main:app --host 0.0.0.0 --port 8000

meta-start: up
	@echo "Waiting for services..."
	@until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do printf '.'; sleep 2; done
	@until curl -sf http://localhost:7474 > /dev/null 2>&1; do printf '.'; sleep 2; done
	@echo " ready."
	$(MAKE) meta-serve

meta-index:
	@mkdir -p $(META_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/index_large.py \
	  --data-dir $(META_DATA_DIR)

meta-evaluate:
	@mkdir -p $(META_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/evaluate_large.py \
	  --input $(META_TEST_DIR)/evaluation_set_final.csv \
	  --output $(META_RESULTS_DIR)/evaluation_answers.csv

meta-score:
	PYTHONPATH=src uv run python $(HARNESS)/score_large.py \
	  --input $(META_RESULTS_DIR)/evaluation_answers.csv \
	  --output $(META_RESULTS_DIR)/evaluation_scored.csv \
	  --summary $(META_RESULTS_DIR)/evaluation_summary.txt

meta-eval: meta-index meta-evaluate meta-score

# Legal variant of the metadata-filtered baseline. Keeps MULTI_PART_DOCUMENTS
# for legal chunking, but sets NO VERSIONRAG_EXTRACT_PROMPT (LLM extraction is
# gone entirely in this mode).
META_LEGAL_RESULTS_DIR := large_dataset/legal/results_metadata_filtered

meta-legal-serve:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src INDEX_MODE=metadata_filtered MULTI_PART_DOCUMENTS=true \
	  uv run uvicorn versionrag.main:app --host 0.0.0.0 --port 8000

meta-legal-start: up
	@echo "Waiting for services..."
	@until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do printf '.'; sleep 2; done
	@until curl -sf http://localhost:7474 > /dev/null 2>&1; do printf '.'; sleep 2; done
	@echo " ready."
	$(MAKE) meta-legal-serve

meta-legal-index:
	@mkdir -p $(META_LEGAL_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/index_large.py \
	  --data-dir $(LEGAL_DATA_DIR)

meta-legal-evaluate:
	@mkdir -p $(META_LEGAL_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/evaluate_large.py \
	  --input $(LEGAL_TEST_DIR)/evaluation_set.csv \
	  --output $(META_LEGAL_RESULTS_DIR)/evaluation_answers.csv

meta-legal-score:
	PYTHONPATH=src uv run python $(HARNESS)/score_large.py \
	  --input $(META_LEGAL_RESULTS_DIR)/evaluation_answers.csv \
	  --output $(META_LEGAL_RESULTS_DIR)/evaluation_scored.csv \
	  --summary $(META_LEGAL_RESULTS_DIR)/evaluation_summary.txt

meta-legal-eval: meta-legal-index meta-legal-evaluate meta-legal-score
# --- end Metadata-Filtered block ---

# ---------------------------------------------------------------------------
# Parser baseline — deterministic metadata extraction + the LLM change layer.
# The cell between full VersionRAG and Metadata-Filtered: it keeps the change
# summaries (the contribution) and drops only the LLM attribute call, which the
# path parser reproduces for free. Software corpus only.
#
# Workflow:  Terminal 1: make parser-start   Terminal 2: make parser-eval
#
# Run artifacts (indexing/eval resource reports, graph snapshot) go to
# VersionRAG/parser/; the scored CSVs go into the results/ comparison tree
# under a new system key so analysis/ picks them up alongside the others.
# ---------------------------------------------------------------------------
PARSER_MODEL       := mistralai/mistral-small-2603
PARSER_TAG         := mistral-small-2603
PARSER_EMBED       := Text-Embedding-3-small
# Mirrors VersionRAG/full/indexing/<embedding>/…_<generator>: resource reports are
# a property of the embedding x generator pair, so both belong in the path.
PARSER_INDEX_DIR   := VersionRAG/parser/indexing/$(PARSER_EMBED)
PARSER_EVAL_DIR    := VersionRAG/parser/evaluation/$(PARSER_EMBED)
PARSER_RESULTS_DIR := results/versionrag_parser/software/mistral_2603
PARSER_SNAPSHOT    := VersionRAG/snapshots/software_parser_$(PARSER_EMBED)_$(PARSER_TAG).json.gz

parser-serve:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src INDEX_MODE=parser \
	  LLM_MODEL_FAST=$(PARSER_MODEL) LLM_MODEL_STANDARD=$(PARSER_MODEL) \
	  LLM_MODEL_EXTRACT=$(PARSER_MODEL) LLM_MODEL_CHAT=$(PARSER_MODEL) \
	  LLM_MODEL_CHANGE=$(PARSER_MODEL) \
	  uv run uvicorn versionrag.main:app --host 0.0.0.0 --port 8000

parser-start: up
	@echo "Waiting for services..."
	@until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do printf '.'; sleep 2; done
	@until curl -sf http://localhost:7474 > /dev/null 2>&1; do printf '.'; sleep 2; done
	@echo " ready."
	$(MAKE) parser-serve

parser-index:
	@mkdir -p $(PARSER_INDEX_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/index_large.py \
	  --data-dir $(META_DATA_DIR) \
	  --resources-dir $(PARSER_INDEX_DIR) \
	  --resources-tag $(PARSER_TAG)

# Persist the built graph as a file — the Change nodes are the expensive part
# and the next indexing run resets the Neo4j volume.
parser-snapshot:
	PYTHONPATH=src uv run python scripts/graph_snapshot.py dump $(PARSER_SNAPSHOT)

# Phase A only makes sense to run once: in parser mode it uses no LLM, so with
# the embedding fixed it is identical for every generator. Snapshot it before
# the change layer is built, then re-run only Phase B per model.
PARSER_PHASE_A := VersionRAG/snapshots/software_parser_phaseA_$(PARSER_EMBED).json.gz

parser-snapshot-phase-a:
	PYTHONPATH=src uv run python scripts/graph_snapshot.py dump $(PARSER_PHASE_A)

# Swap PARSER_MODEL/PARSER_TAG on the command line to do another generator:
#   make parser-changes PARSER_MODEL=gpt-5.4-nano PARSER_TAG=gpt-5.4-nano
# Requires a server started with that model and a graph already holding Phase A
# (restore $(PARSER_PHASE_A) if the volume was reset; Qdrant must still be filled
# from the same ingest, since snapshots cover Neo4j only).
parser-changes:
	@mkdir -p $(PARSER_INDEX_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/rebuild_changes.py \
	  --resources-dir $(PARSER_INDEX_DIR) \
	  --resources-tag $(PARSER_TAG)

parser-evaluate:
	@mkdir -p $(PARSER_RESULTS_DIR)
	PYTHONPATH=src uv run python $(HARNESS)/evaluate_large.py \
	  --input $(META_TEST_DIR)/evaluation_set_final.csv \
	  --output $(PARSER_RESULTS_DIR)/evaluation_answers_$(PARSER_TAG).csv

parser-score:
	PYTHONPATH=src uv run python $(HARNESS)/score_large.py \
	  --input $(PARSER_RESULTS_DIR)/evaluation_answers_$(PARSER_TAG).csv \
	  --output $(PARSER_RESULTS_DIR)/evaluation_scored_$(PARSER_TAG).csv \
	  --summary $(PARSER_RESULTS_DIR)/evaluation_summary_$(PARSER_TAG).txt \
	  --resources-dir $(PARSER_EVAL_DIR) \
	  --resources-tag $(PARSER_TAG)

parser-eval: parser-index parser-snapshot parser-evaluate parser-score
# --- end Parser baseline block ---

# ---------------------------------------------------------------------------
# Naive RAG baseline — port of the paper's reference implementation
# (danielhuwiler/versionrag). Fills the empty results/naive_rag/legal/ cell.
#
# Needs NO docker: Milvus Lite is a local file, and everything else is a direct
# API call. Runs in its own venv (naive_rag_baseline/.venv) so the main
# pyproject/uv.lock stay untouched — run `make naive-setup` once first.
#
# See naive_rag_baseline/README.md for the one deliberate deviation from
# upstream (a chunk/vector mispairing in the reference indexer) and why the
# software Naive row is affected by it too.
# ---------------------------------------------------------------------------
NAIVE_PY       := naive_rag_baseline/.venv/bin/python
NAIVE_DB       := naive_rag_baseline/db/milvus_legal.db
NAIVE_CACHE    := naive_rag_baseline/db/retrieval_legal.json
NAIVE_EVAL_SET := large_dataset/legal/test/evaluation_set.csv
NAIVE_OUT      := results/naive_rag/legal

# <output-dir-label>:<model-slug>. Labels match results/naive_rag/software/ so
# analysis/loaders.py normalises them to the same generator keys.
NAIVE_PAIRS := \
  gpt_5_4_nano:gpt-5.4-nano \
  gpt_5_4_mini:gpt-5.4-mini \
  mistral_2603:mistralai/mistral-small-2603 \
  qwen_3_32B:qwen/qwen3-32b

naive-setup:
	uv venv naive_rag_baseline/.venv --python 3.12
	uv pip install --python $(NAIVE_PY) -r naive_rag_baseline/requirements.txt

naive-legal-index:
	$(NAIVE_PY) naive_rag_baseline/index.py \
	  --data-dir $(LEGAL_DATA_DIR) --db $(NAIVE_DB) --fresh

# Upstream's indexer verbatim, defects included — for sizing the fixed-vs-
# verbatim delta before deciding whether to re-run the software row.
naive-legal-verbatim-index:
	NAIVE_VERBATIM_INDEX=1 $(NAIVE_PY) naive_rag_baseline/index.py \
	  --data-dir $(LEGAL_DATA_DIR) \
	  --db naive_rag_baseline/db/milvus_legal_verbatim.db --fresh

# Answer + score all four generators. Resumable: re-run to pick up where it
# stopped. Retrieval is computed once and shared, so all four see identical
# context.
naive-legal-run:
	@for pair in $(NAIVE_PAIRS); do \
	  g=$${pair%%:*}; model=$${pair#*:}; \
	  echo ""; echo "=== $$g ($$model) ==="; \
	  $(NAIVE_PY) naive_rag_baseline/run_queries.py \
	    --eval-set $(NAIVE_EVAL_SET) --db $(NAIVE_DB) \
	    --retrieval-cache $(NAIVE_CACHE) --model "$$model" \
	    --out $(NAIVE_OUT)/$$g/evaluation_out_Baseline.csv || exit 1; \
	  $(NAIVE_PY) naive_rag_baseline/score.py \
	    --input   $(NAIVE_OUT)/$$g/evaluation_out_Baseline.csv \
	    --output  $(NAIVE_OUT)/$$g/evaluation_out_Baseline_scored.csv \
	    --summary $(NAIVE_OUT)/$$g/evaluation_summary_Baseline.txt || exit 1; \
	done

naive-legal: naive-legal-index naive-legal-run
# --- end Naive RAG block ---
