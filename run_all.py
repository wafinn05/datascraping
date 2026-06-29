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

for script in scripts:
    script_path = os.path.join(CURRENT_DIR, script)
    print(f"\n>>> Menjalankan {script}...")
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"XXX Gagal menjalankan {script}. Error: {e}")
        break

print("\n=== PROSES SCRAPING SELESAI ===")
