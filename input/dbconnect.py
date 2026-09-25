import psycopg2
def run_query(query: str,
              host="localhost",
              database="postgres",
              user="postgres",
              password="2660",
              port=5432,
              readonly_only: bool = True):
    """
    Executes a query on PostgreSQL.
    By default only allows read-only queries (SELECT / WITH) to avoid accidental destructive operations.
    Returns fetched rows for SELECT queries or None otherwise.
    """

    # Basic safety check: allow only SELECT or WITH (common read-only patterns)
    qstart = query.lstrip().split(None, 1)
    if not qstart:
        print("Empty query.")
        return None

    first_token = qstart[0].lower()
    allowed_readonly = {"select", "with"}
    if readonly_only and first_token not in allowed_readonly:
        print(f"Safety block: query starts with '{first_token}' which is not allowed in readonly mode.")
        print("If you want to allow non-read queries, call run_query(..., readonly_only=False).")
        return None

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(host=host, database=database, user=user, password=password, port=port)
        cursor = conn.cursor()
        cursor.execute(query)

        if first_token == "select" or first_token == "with":
            rows = cursor.fetchall()
            print(f"Fetched {len(rows)} rows.")
            return rows
        else:
            # commit for other statements
            conn.commit()
            print("Query executed and committed.")
            return None

    except Exception as e:
        print("Error executing query:", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()    


