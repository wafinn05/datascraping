from src.database.connection import get_db_connection
from src.database.schema import init_tables

def test_connection():
    try:
        print("Mencoba koneksi ke database...")
        client = get_db_connection()
        init_tables(client)
        
        # Query sederhana
        result = client.execute("SELECT 1 as val")
        print(f"Koneksi Berhasil! Turso/LibSQL merespons dengan nilai: {result.rows[0][0]}")
        client.close()
    except Exception as e:
        print(f"Koneksi Gagal: {e}")

if __name__ == "__main__":
    test_connection()
