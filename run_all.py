import subprocess
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    "src/collectors/prices.py",
    "src/collectors/fundamental.py",
    "src/collectors/macro.py",
    "src/collectors/sentiment.py",
    "src/features/process.py"
]

print("=== MEMULAI PROSES SCRAPING MASSAL KE TURSO ===")
print(f"DEBUG ENV: CHUNK_INDEX = {os.environ.get('CHUNK_INDEX')}")
print(f"DEBUG ENV: TURSO_DATABASE_URL exists = {bool(os.environ.get('TURSO_DATABASE_URL'))}")
print(f"DEBUG ENV: TURSO_AUTH_TOKEN exists = {bool(os.environ.get('TURSO_AUTH_TOKEN'))}")

for script in scripts:
    script_path = os.path.join(CURRENT_DIR, script)
    print(f"\n>>> Menjalankan {script}...")
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"XXX Gagal menjalankan {script}. Error: {e}")
        break

print("\n=== PROSES SCRAPING SELESAI ===")
