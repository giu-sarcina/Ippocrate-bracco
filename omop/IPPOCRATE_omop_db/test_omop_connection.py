import os
import psycopg2

# Default connection parameters (from documentation)
DB_NAME = os.getenv('POSTGRES_DB', 'omop4ippocrate')
DB_USER = os.getenv('POSTGRES_USER', 'giulia')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'cambiami')
#DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_HOST = 'ippocratedb'
DB_PORT = os.getenv('POSTGRES_PORT', '5432')

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print(f"Connected to database {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
    cur = conn.cursor()
    cur.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY table_schema, table_name;")
    tables = cur.fetchall()
    print("Tables present in the database:")
    for schema, table in tables:
        print(f"{schema}.{table}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"Error connecting to the database: {e}")
