import libsql_client
import os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("TURSO_DATABASE_URL", "sqlite:///turso_local.db")
token = os.getenv("TURSO_AUTH_TOKEN", "")

try:
    print(f"Connecting to {url}")
    client = libsql_client.create_client_sync(url=url, auth_token=token)
    rs = client.execute("SELECT 1 as val")
    print(rs.rows[0][0])
except Exception as e:
    print(e)
