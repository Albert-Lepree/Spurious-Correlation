import psycopg2, pandas as pd, os


def get_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "spuriousCorrelationdb"),
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host=os.getenv("DB_HOST", "localhost")
    )


def query(sql):
    conn = get_db()
    df = pd.read_sql(sql, conn)
    conn.close()
    return df
