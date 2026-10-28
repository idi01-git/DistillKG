"""
Downloads and extracts the 4 primary PRONTO Process Data CSV files from Zenodo (DOI: 10.5281/zenodo.1341583)
using targeted HTTP byte-range requests without downloading the 1.7 GB of high-frequency audio/video data.
"""

import os
import struct
import zlib
import subprocess
import pandas as pd

PRONTO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pronto")
os.makedirs(PRONTO_DIR, exist_ok=True)

ZENODO_URL = "https://zenodo.org/api/records/1341583/files/PRONTO%20benchmark%20case%20study.zip/content"

# Target files to extract
TARGETS = [
    {
        "filename": "0912Testday4.csv",
        "description": "C0: Normal & Slugging conditions (Tests 9 & 10)",
        "local_hdr_offset": 357811160,
        "comp_size": 1487131,
        "uncomp_size": 4920870
    },
    {
        "filename": "0626Testday5.csv",
        "description": "C0: Additional Slugging conditions (Test 11)",
        "local_hdr_offset": 325973309,
        "comp_size": 308034,
        "uncomp_size": 1197062
    },
    {
        "filename": "0907Testday2.csv",
        "description": "C1 Air Blockage (Tests 2, 3) & C2 Air Leakage (Test 4)",
        "local_hdr_offset": 656982298,
        "comp_size": 1917180,
        "uncomp_size": 6440042
    },
    {
        "filename": "0911Testday3.csv",
        "description": "C2 Air Leakage (Tests 5, 6) & C3 Diverted Flow (Tests 7, 8)",
        "local_hdr_offset": 822499213,
        "comp_size": 2519653,
        "uncomp_size": 8724815
    }
]


def extract_target_file(target: dict):
    out_path = os.path.join(PRONTO_DIR, target["filename"])
    if os.path.exists(out_path) and os.path.getsize(out_path) == target["uncomp_size"]:
        print(f"  {target['filename']} already extracted ({target['uncomp_size']} bytes).")
        return out_path

    print(f"Downloading {target['filename']} ({target['description']})...")
    # Fetch local header + compressed payload
    # Add 1024 bytes buffer for variable local header (filename + extra field)
    fetch_len = target["comp_size"] + 2048
    start_byte = target["local_hdr_offset"]
    end_byte = start_byte + fetch_len - 1

    temp_chunk_path = os.path.join(PRONTO_DIR, f"temp_{target['filename']}.bin")
    cmd = [
        "curl.exe", "-s", "-L",
        "-r", f"{start_byte}-{end_byte}",
        ZENODO_URL,
        "-o", temp_chunk_path
    ]
    subprocess.run(cmd, check=True)

    with open(temp_chunk_path, "rb") as f:
        raw_data = f.read()

    # Parse local file header:
    # 0..4: sig (0x04034b50)
    # 26..28: name_len
    # 28..30: extra_len
    sig = raw_data[:4]
    assert sig == b"\x50\x4b\x03\x04", f"Invalid local header signature for {target['filename']}: {sig}"
    name_len = struct.unpack("<H", raw_data[26:28])[0]
    extra_len = struct.unpack("<H", raw_data[28:30])[0]

    header_len = 30 + name_len + extra_len
    compressed_payload = raw_data[header_len:header_len + target["comp_size"]]

    # Decompress raw deflate
    decomp = zlib.decompressobj(-zlib.MAX_WBITS)
    uncompressed_data = decomp.decompress(compressed_payload)

    with open(out_path, "wb") as f:
        f.write(uncompressed_data)

    os.remove(temp_chunk_path)
    print(f"  Successfully extracted {target['filename']}: {len(uncompressed_data)} bytes.")
    return out_path


if __name__ == "__main__":
    print("=" * 65)
    print("DOWNLOADING & EXTRACTING PRONTO MULTIPHASE PROCESS DATASET")
    print("=" * 65)
    for t in TARGETS:
        extract_target_file(t)
    print("All PRONTO process data successfully extracted into data/raw/pronto/")

