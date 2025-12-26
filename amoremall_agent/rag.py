from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import psycopg2
from pgvector.psycopg2 import register_vector


DEFAULT_VECTOR_DIM = 64


@dataclass
class SearchResult:
    item_id: str
    content: str
    metadata: dict[str, Any]
    score: float


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def simple_hash_embedding(text: str, dim: int = DEFAULT_VECTOR_DIM) -> list[float]:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return [0.0] * dim
    values = [0.0] * dim
    for token in normalized.split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(dim):
            values[idx] += digest[idx % len(digest)] / 255.0
    total = sum(values) or 1.0
    return [v / total for v in values]


class PgVectorStore:
    def __init__(self, dsn: str, dim: int = DEFAULT_VECTOR_DIM):
        self.dsn = dsn
        self.dim = dim

    def _connect(self):
        conn = psycopg2.connect(self.dsn)
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS amoremall_products (
                    product_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    brand TEXT,
                    category TEXT,
                    description TEXT,
                    metadata JSONB,
                    embedding VECTOR(%s)
                )
                """,
                (self.dim,),
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS amoremall_brand_tones (
                    brand TEXT PRIMARY KEY,
                    tone_text TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR(%s)
                )
                """,
                (self.dim,),
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS amoremall_reviews (
                    review_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    review_text TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR(%s)
                )
                """,
                (self.dim,),
            )

    def upsert_products(self, rows: Iterable[dict[str, Any]]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for row in rows:
                description = _normalize_text(row.get("description", ""))
                embedding = simple_hash_embedding(
                    " ".join(
                        filter(
                            None,
                            [
                                row.get("name", ""),
                                row.get("brand", ""),
                                row.get("category", ""),
                                description,
                            ],
                        )
                    ),
                    self.dim,
                )
                cur.execute(
                    """
                    INSERT INTO amoremall_products
                        (product_id, name, brand, category, description, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (product_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        brand = EXCLUDED.brand,
                        category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        row.get("product_id"),
                        row.get("name"),
                        row.get("brand"),
                        row.get("category"),
                        description,
                        json.dumps(row.get("metadata", {})),
                        embedding,
                    ),
                )

    def upsert_brand_tones(self, rows: Iterable[dict[str, Any]]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for row in rows:
                tone_text = _normalize_text(row.get("tone_text", ""))
                embedding = simple_hash_embedding(tone_text, self.dim)
                cur.execute(
                    """
                    INSERT INTO amoremall_brand_tones
                        (brand, tone_text, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (brand) DO UPDATE SET
                        tone_text = EXCLUDED.tone_text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        row.get("brand"),
                        tone_text,
                        json.dumps(row.get("metadata", {})),
                        embedding,
                    ),
                )

    def upsert_reviews(self, rows: Iterable[dict[str, Any]]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for row in rows:
                review_text = _normalize_text(row.get("review_text", ""))
                embedding = simple_hash_embedding(review_text, self.dim)
                cur.execute(
                    """
                    INSERT INTO amoremall_reviews
                        (review_id, product_id, review_text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (review_id) DO UPDATE SET
                        product_id = EXCLUDED.product_id,
                        review_text = EXCLUDED.review_text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        row.get("review_id"),
                        row.get("product_id"),
                        review_text,
                        json.dumps(row.get("metadata", {})),
                        embedding,
                    ),
                )

    def search(
        self,
        query: str,
        table: str,
        content_field: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query:
            return []
        embedding = simple_hash_embedding(query, self.dim)
        filter_sql = ""
        params: list[Any] = [embedding]
        if filters:
            clauses = []
            for key, value in filters.items():
                clauses.append(f"{key} = %s")
                params.append(value)
            filter_sql = "WHERE " + " AND ".join(clauses)
        id_field = "review_id"
        if table == "amoremall_products":
            id_field = "product_id"
        elif table == "amoremall_brand_tones":
            id_field = "brand"
        query_sql = f"""
            SELECT
                CAST({id_field} AS TEXT) AS item_id,
                {content_field} AS content,
                metadata,
                1 - (embedding <=> %s) AS score
            FROM {table}
            {filter_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        params.append(embedding)
        params.append(top_k)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                SearchResult(
                    item_id=row[0],
                    content=row[1],
                    metadata=row[2] or {},
                    score=float(row[3]),
                )
            )
        return results
