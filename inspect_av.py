"""بيفحص الداتا الجديدة (zip أو فولدر) ويكتب تقرير نصي صغير: inspect_report.txt

الاستخدام:
    python inspect_av.py data_av/اسم_الملف.zip
    python inspect_av.py data_av/فولدر_مفتوح

ابعت ملف inspect_report.txt بس (حجمه صغير). السكريبت مبيعدّلش على الملفات الأصلية.
"""
import os
import sys
import zipfile
from collections import Counter

OUT = "inspect_report.txt"
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def safe_extract(zpath, dest):
    root = os.path.abspath(dest)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = os.path.abspath(os.path.join(dest, m.filename))
            if not target.startswith(root + os.sep) and target != root:
                sys.exit(f"Unsafe path in zip: {m.filename}")
        z.extractall(dest)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python inspect_av.py <zip-or-folder>")
    src = sys.argv[1]
    if zipfile.is_zipfile(src):
        dest = os.path.splitext(src)[0] + "_extracted"
        if not os.path.isdir(dest):
            log("Extracting to", dest, "...")
            os.makedirs(dest)
            safe_extract(src, dest)
        src = dest
    if not os.path.isdir(src):
        sys.exit("Path not found: " + src)

    files = []
    for r, _, fs in os.walk(src):
        for f in fs:
            p = os.path.join(r, f)
            files.append((p, os.path.getsize(p)))
    log(f"\nTotal files: {len(files)} | total size: {sum(s for _, s in files)/1e6:.1f} MB")

    ext = Counter(os.path.splitext(p)[1].lower() for p, _ in files)
    log("Files by extension:", dict(ext.most_common()))

    log("\n--- Folder tree (depth<=3, first entries) ---")
    for r, ds, fs in os.walk(src):
        depth = os.path.relpath(r, src).count(os.sep) if r != src else 0
        if depth > 2:
            ds[:] = []
            continue
        ds.sort()
        log("  " * depth + os.path.basename(r) + f"/  ({len(fs)} files, {len(ds)} subfolders)")
        for f in sorted(fs)[:5]:
            log("  " * (depth + 1) + f)

    import pandas as pd
    tab = [p for p, _ in files if p.lower().endswith((".csv", ".parquet"))]
    log(f"\n--- Tables found: {len(tab)} ---")
    for p in tab[:12]:
        try:
            df = pd.read_parquet(p) if p.lower().endswith(".parquet") else pd.read_csv(p)
            log(f"\n{os.path.relpath(p, src)}  shape={df.shape}")
            log("columns (first 15):", list(df.columns[:15]))
            log(df.head(3).iloc[:, :8].to_string())
        except Exception as e:
            log(f"\n{os.path.relpath(p, src)}  could not read: {e}")

    imgs = [p for p, _ in files if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if imgs:
        log(f"\n--- Images: {len(imgs)} ---")
        try:
            from PIL import Image
            for p in imgs[:3]:
                with Image.open(p) as im:
                    log(os.path.relpath(p, src), im.size, im.mode)
        except Exception as e:
            log("could not open images:", e)

    aud = [p for p, _ in files if p.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))]
    if aud:
        log(f"\n--- Audio files: {len(aud)} ---", [os.path.relpath(p, src) for p in aud[:3]])

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {OUT}  <-- ابعتلي الملف ده")


if __name__ == "__main__":
    main()
