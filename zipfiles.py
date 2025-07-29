# zipfiles.py
import os
import zipfile
import time

ZIP_NAME = "functionapp.zip"
# Fecha segura (año, mes, día, hora, minuto, segundo)
SAFE_DATE = time.localtime(time.time())[:6]

def should_skip_dir(d):
    return d.startswith(".") or d in ("__pycache__",)

def should_skip_file(f):
    return f.startswith(".") or f.endswith((".pyc", ".pyo"))

with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("."):
        # Filtramos subdirectorios ocultos o __pycache__
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        for fname in files:
            if should_skip_file(fname) or fname == "zipfiles.py":
                continue
            full = os.path.join(root, fname)
            arc  = os.path.relpath(full, ".")
            data = open(full, "rb").read()
            info = zipfile.ZipInfo(arc)
            info.date_time     = SAFE_DATE
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)

print(f"✅ Creado {ZIP_NAME}")
