import os
import libsql_client
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    turso_url = os.getenv("TURSO_DATABASE_URL")
    turso_auth_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if turso_url and turso_auth_token:
        print("[DB] Menghubungkan ke Turso Cloud Database (LibSQL Client)...")
        # Ubah libsql:// menjadi https:// untuk mencegah isu WebSockets (wss://) di Windows
        http_url = turso_url.replace("libsql://", "https://")
        return libsql_client.create_client_sync(url=http_url, auth_token=turso_auth_token)
    else:
        print("[DB] Menghubungkan ke SQLite Lokal (turso_local.db)...")
        return libsql_client.create_client_sync(url="file:turso_local.db")
