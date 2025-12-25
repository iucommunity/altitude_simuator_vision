import os
import shutil

# =========================
# CONFIG
# =========================
SOURCE_FOLDER = "images"          # folder containing original images
DEST_FOLDER = "images_sorted"    # output folder
START_INDEX = 1                  # starting index (usually 1)
INDEX_WIDTH = 4                  # 0001, 0002, ...

VALID_EXTS = (".png", ".jpg", ".jpeg")

# =========================
# PREPARE OUTPUT FOLDER
# =========================
os.makedirs(DEST_FOLDER, exist_ok=True)

# =========================
# COLLECT FILES
# =========================
files = []

for filename in os.listdir(SOURCE_FOLDER):
    if filename.lower().endswith(VALID_EXTS):
        full_path = os.path.join(SOURCE_FOLDER, filename)
        mtime = os.path.getmtime(full_path)
        files.append((filename, mtime))

# =========================
# SORT BY MODIFIED TIME
# =========================
files.sort(key=lambda x: x[1])  # oldest → newest
# files.sort(key=lambda x: x[1], reverse=True)  # newest → oldest

# =========================
# COPY WITH INDEX PREFIX
# =========================
for idx, (filename, _) in enumerate(files, start=START_INDEX):
    index_str = str(idx).zfill(INDEX_WIDTH)
    new_name = f"{index_str}.{filename}"

    src = os.path.join(SOURCE_FOLDER, filename)
    dst = os.path.join(DEST_FOLDER, new_name)

    shutil.copy2(src, dst)  # preserves metadata

    print(f"✔ {filename} → {new_name}")

print(f"\n✅ Done! {len(files)} files saved to '{DEST_FOLDER}'")
