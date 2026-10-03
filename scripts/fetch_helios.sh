#!/usr/bin/env bash
# Downloads the Helios GPU-cluster traces (S-Lab-System-Group/HeliosData,
# CC-BY-4.0) into the ignored benchmarks/outputs/helios/ and extracts the two
# Venus files the replay scenario (S13) reads. Run from the repository root:
#     bash scripts/fetch_helios.sh
# Nothing from the traces is committed; the replay results cite the source.
set -euo pipefail
dest=benchmarks/outputs/helios
url=https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/data.zip
want=3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac
py=${PYTHON:-python}
mkdir -p "$dest"
if [ ! -f "$dest/data.zip" ]; then
  curl -fL --retry 3 --max-time 900 -o "$dest/data.zip.part" "$url"
  mv "$dest/data.zip.part" "$dest/data.zip"
fi
"$py" - "$dest" "$want" <<'PY'
import hashlib, sys, zipfile
dest, want = sys.argv[1], sys.argv[2]
h = hashlib.sha256(open(f"{dest}/data.zip", "rb").read()).hexdigest()
if h != want:
    sys.exit(f"data.zip has SHA-256 {h}, expected {want}")
zipfile.ZipFile(f"{dest}/data.zip").extractall(dest, members=["data/Venus/cluster_log.csv", "data/Venus/cluster_gpu_number.csv"])
print(f"extracted the Venus files into {dest}/data/Venus (CC-BY-4.0, S-Lab-System-Group/HeliosData)")
PY
