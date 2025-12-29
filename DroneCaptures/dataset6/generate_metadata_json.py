import os
import re
import json

# =========================
# CONFIG
# =========================
IMAGE_FOLDER = "images_sorted"        # change this to your folder path
OUTPUT_JSON = "metadata.json"

# Regex to parse filename
pattern = re.compile(
    r"Lat_(?P<lat>[\d.]+)_"
    r"Lon_(?P<lon>[\d.]+)_"
    r"Alt_(?P<height>[\d.]+)m_"
    r"P(?P<pitch>[-\d.]+)_"
    r"Y(?P<yaw>[-\d.]+)_"
    r"R(?P<roll>[-\d.]+)"
)

data = []

# =========================
# PARSE FILES
# =========================
for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    match = pattern.search(filename)
    if not match:
        print(f"Skipping (no match): {filename}")
        continue

    file_path = os.path.join(IMAGE_FOLDER, filename)
    modified_time = os.path.getmtime(file_path)  # 🔥 get modified time

    entry = {
        "filename": filename,
        "lat": float(match.group("lat")),
        "lon": float(match.group("lon")),
        "height": float(match.group("height")),
        "pitch": 0 - (float(match.group("pitch")) - 60),
        "yaw": float(match.group("yaw")) - 360 if float(match.group("yaw")) > 180 else float(match.group("yaw")),
        "roll": float(match.group("roll")),
        "_mtime": modified_time,  # temp field for sorting
    }

    data.append(entry)

# =========================
# SORT BY MODIFIED TIME
# =========================
data.sort(key=lambda x: x["_mtime"])  # oldest → newest
# data.sort(key=lambda x: x["_mtime"], reverse=True)  # newest → oldest

# Remove temporary field
for item in data:
    item.pop("_mtime")

# =========================
# SAVE JSON
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ Saved {len(data)} entries to {OUTPUT_JSON}")
