# pgvector-basics — Postgres as a vector store (ports 5435 + 8312)

Two Jupyter notebooks that take a class from "a table with normal columns
and one `vector` column" to "HNSW index, recall dial, and the
filtered-search gotcha", on a real Postgres 16 + pgvector 0.8.6 stack in
Docker. Built + executed end-to-end 2026-10-15 (both notebooks ship with
their outputs baked in).

Companions: U2T03 turns these mechanics into an assignment (reference
impl at [`experiments/doc-search-pgvector`](../../../../experiments/doc-search-pgvector/));
deck-07's outline wires its scale/fp16/MRL callbacks to this demo;
week-9's `hybrid-retrieval-lab` later re-meets ANN from the RAG side.

## The question everyone asks: does pgvector index vectors automatically?

**No.** Nothing indexes a `vector` column until you run
`CREATE INDEX ... USING hnsw (col vector_cosine_ops)` yourself. Three
corollaries the notebooks make visible:

1. without an index, every distance query is an **exact sequential scan**
   (fine and optimal at small N — notebook 01 never builds one);
2. a plain `CREATE INDEX` (B-tree) cannot index vectors — the operator
   class (`vector_cosine_ops` / `vector_l2_ops` / `vector_ip_ops`) is
   part of the DDL, and each one is a separate index;
3. even with an HNSW index present, the planner **ignores it on small
   tables** because a seq scan is genuinely cheaper — you need scale
   before the index wins (measured here: 50k rows).

## Run

```bash
docker compose up -d --build   # first build ~3 min; model downloads once (~470 MB)
open http://localhost:8312     # JupyterLab, no token (class demo)
```

- Postgres on host port **5435**: `docker compose exec db psql -U demo -d vectordb`
- The e5-small model caches into the host `~/.cache/huggingface` (mounted),
  so classroom runs after the first download are offline-safe.
- Teardown: `docker compose down` (keeps data) / `down -v` (wipes).
- Both notebooks are idempotent — safe to re-run top-to-bottom in class.

## Stack and model choices

| piece | choice | why |
|---|---|---|
| database | `pgvector/pgvector:pg16` | same image as the U2T03 reference impl |
| driver | `psycopg` 3 + `pgvector.psycopg.register_vector` | NumPy in/out of the `vector` type |
| embedder | `intfloat/multilingual-e5-small` (384-d) | small + CPU-fast, multilingual (Spanish queries work, the course pivot beat), keeps the `query:`/`passage:` prefix discipline students need for U2T03 |

Table: `documents(title, category, lang, year, content, embedding
vector(384))`, seeded with 36 bilingual course-themed docs (4 categories).
Dimension is fixed at DDL time — the reference impl's one-table-per-model
design is the direct consequence.

## Notebook 01 — insert and search (exact)

Connect + register the type; CREATE TABLE with the vector column; embed
the corpus with prefixes (normalized, so cosine = dot product); insert
(NumPy binding + the literal `'[...]'::vector` wire format); exact top-k
with `ORDER BY embedding <=> q`. Beats, all measured in the file:

- "how do I speed up a slow query" → EXPLAIN/index/pooling rows on top;
- Spanish query, same space → right row first (0.87), zero shared
  keywords; the tail's language-drift is called out honestly;
- `<=>` / `<->` / `<#>` rank identically on unit vectors — the operator
  choice is an *index* concern, not a correctness one;
- metadata scoping: the same query filtered to `category='embeddings'`
  flips the answer set to ANN/MRL/prefix rows;
- score calibration: self-similarity 1.0000, `query:` vs `passage:` of
  the same text 0.9477 (asymmetry made visible), random doc pairs ~0.81 —
  rank is the signal, not the absolute number. Prefix-drop footnote:
  ≤0.006 on this corpus, matching the U2T03 reference's e5-base finding.

## Notebook 02 — HNSW and filtered search (approximate)

50k rows of **clustered** synthetic vectors (100 Gaussian clusters — real
embedding spaces cluster; uniform noise is the adversarial worst case and
sinks recall to ~0.4 at default settings, worth saying out loud).
Measured on this machine (M1-class CPU, numbers baked in the file):

| beat | measured |
|---|---|
| exact seq scan @ 50k | 17.1 ms |
| HNSW query | 0.20 ms (**87×**), build 12 s one-time |
| index vs table size | 102 MB index > 82 MB table |
| recall@10 ladder (`ef_search` 1/8/40/200) | 0.09 / 0.73 / 0.94 / 0.98 |
| `<->` query vs cosine index | planner back to seq scan — one index per operator |
| rare-category filter (2%), defaults | **asked for 5, got 1** (post-filter starves the LIMIT) |
| same + `iterative_scan=strict_order` | 5 of 5, ~0.2 ms extra |
| partial index `WHERE category='databases'` | planner picks it unprompted; 2 MB vs 102 MB |
| `halfvec` expression index | 58 MB vs 102 MB — deck-06's fp16 slide as an index |

The shrink gotcha is the centerpiece: with `hnsw.iterative_scan = off`
(the 0.8.6 default), the index walk yields ~`ef_search` candidates,
Postgres post-filters by category, and `LIMIT k` can return fewer than
k rows. `strict_order` fixes it; for very selective filters the planner
may rightly seq-scan instead. Common categories don't trigger it at all.

## Classroom notes

- Notebook 01 runtime ~30 s (model load dominates); notebook 02 ~90 s
  (COPY 10 s + HNSW build 12 s + halfvec build 6 s).
- The summary table in notebook 02's last cell assembles from the run's
  own measurements, so re-running refreshes the story.
- `hnsw.*` settings only exist after the pgvector module loads in a
  session (first vector op); `SHOW hnsw.ef_search` on a fresh connection
  errors until then. Not a bug, just surprising.
- GUCs are per-session: `SET hnsw.iterative_scan = ...` from one notebook
  connection does not leak into `psql`.
