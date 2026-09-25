import os
from typing import Any

import psycopg2


def run_query(
    query: str,
    host: str = "localhost",
    database: str = "postgres",
    user: str = "postgres",
    password: str | None = None,
    port: int = 5432,
    readonly_only: bool = True,
) -> list[dict[str, Any]] | None:
    """
    Executes a PostgreSQL query.
    In readonly mode only SELECT/WITH are allowed.
    Returns rows as list of dictionaries for read queries.
    """
    qstart = query.lstrip().split(None, 1)
    if not qstart:
        print("Empty query.")
        return None

    first_token = qstart[0].lower()
    if readonly_only and first_token not in {"select", "with"}:
        print(f"Safety block: '{first_token}' is not allowed in readonly mode.")
        return None

    final_password = password if password is not None else os.getenv("PG_PASSWORD", "2660")

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=final_password,
            port=port,
        )
        cursor = conn.cursor()
        cursor.execute(query)

        if first_token in {"select", "with"}:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return [dict(zip(columns, row)) for row in rows]

        conn.commit()
        return None
    except Exception as e:
        print("Error executing query:", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
