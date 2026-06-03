#!/usr/bin/env bash

SRC="${1:-out/src}"
OUT="${2:-out/}"

while IFS= read -r file; do
    relative="${file#"$SRC"/}"
    out="${relative//\//.}.svg"
    echo "Freezing $file -> $out"
    freeze --theme dracula "$file" -o "$OUT/$out" < /dev/null
done < <(find "$SRC" -name "*.py" ! -name "__init__.py")

inkscape "${OUT}/*.svg" --export-type=pdf --export-area-page
