-- pgvector-basics: only the extension is created here. The demo table is
-- created inside notebook 01 so the class sees the full DDL flow
-- (CREATE EXTENSION would also work from the notebook as user `demo`).
CREATE EXTENSION IF NOT EXISTS vector;
