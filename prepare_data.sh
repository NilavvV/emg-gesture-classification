#!/bin/bash
set -e

RAW_DIR="EMG_data_for_gestures-master"
OUT_DIR="data"
mkdir -p "$OUT_DIR"
i=1
for file in $(find "$RAW_DIR" -name "*.txt" | sort -V); do
    cp -- "$file" "$OUT_DIR/gesture${i}.txt"
    ((i++))
done

echo "Copied $((i-1)) files into $OUT_DIR/" 

