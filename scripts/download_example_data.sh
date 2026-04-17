#!/usr/bin/env bash
#
# download_example_data.sh — Download example text and prepare training data.
#
# Downloads a small corpus from Project Gutenberg (public domain books),
# then runs prepare.py to tokenize it into binary shards ready for mixlab.
#
# Usage:
#   bash scripts/download_example_data.sh
#   bash scripts/download_example_data.sh --output /path/to/data
#
# Requirements:
#   - curl or wget
#   - Python 3 with numpy and tokenizers (pip install numpy tokenizers)
#
# Total download: ~5 MB of public domain text from Project Gutenberg.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT="${SCRIPT_DIR}/../data/example"
OUTPUT_DIR="${1:---output}"

# Parse --output flag
if [ "$OUTPUT_DIR" = "--output" ]; then
    OUTPUT_DIR=""
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--output DIR]"
            echo ""
            echo "Downloads example text from Project Gutenberg and tokenizes"
            echo "it into binary shards for mixlab training."
            echo ""
            echo "Options:"
            echo "  --output DIR   Output directory (default: data/example)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$DEFAULT_OUTPUT"
fi

# Gutenberg books — plain UTF-8 text, public domain, good variety.
# Each is roughly 0.5-1.5 MB of text.
BOOKS=(
    # Title                                  URL
    "pride_and_prejudice|https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    "adventures_of_sherlock_holmes|https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
    "alice_in_wonderland|https://www.gutenberg.org/cache/epub/11/pg11.txt"
    "frankenstein|https://www.gutenberg.org/cache/epub/84/pg84.txt"
    "moby_dick|https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
    "great_expectations|https://www.gutenberg.org/cache/epub/1400/pg1400.txt"
)

RAW_DIR="${OUTPUT_DIR}/raw"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "==> $*"; }
warn()  { echo "WARNING: $*" >&2; }
die()   { echo "ERROR: $*" >&2; exit 1; }

# Detect download tool
detect_downloader() {
    if command -v curl &>/dev/null; then
        echo "curl"
    elif command -v wget &>/dev/null; then
        echo "wget"
    else
        die "Neither curl nor wget found. Please install one of them."
    fi
}

# Download a URL to a file
download() {
    local url="$1"
    local dest="$2"
    local tool
    tool="$(detect_downloader)"

    if [ "$tool" = "curl" ]; then
        curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
    else
        wget -q --tries=3 -O "$dest" "$url"
    fi
}

# Check Python dependencies
check_python() {
    if ! command -v python3 &>/dev/null; then
        die "python3 not found. Please install Python 3."
    fi

    python3 -c "import numpy" 2>/dev/null || {
        die "numpy not installed. Run: pip install numpy"
    }

    python3 -c "import tokenizers" 2>/dev/null || {
        die "tokenizers not installed. Run: pip install tokenizers"
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

info "Mixlab example data downloader"
info "Output directory: ${OUTPUT_DIR}"
echo ""

# Pre-flight checks
check_python

# Create directories
mkdir -p "$RAW_DIR"
mkdir -p "$OUTPUT_DIR"

# Download books
info "Downloading example texts from Project Gutenberg..."
echo ""

downloaded=0
failed=0

for entry in "${BOOKS[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    dest="${RAW_DIR}/${name}.txt"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  [cached] ${name}.txt"
        downloaded=$((downloaded + 1))
        continue
    fi

    printf "  [downloading] %-35s ... " "${name}.txt"
    if download "$url" "$dest" 2>/dev/null; then
        size=$(wc -c < "$dest" | tr -d ' ')
        echo "ok ($(( size / 1024 )) KB)"
        downloaded=$((downloaded + 1))
    else
        echo "FAILED"
        rm -f "$dest"
        failed=$((failed + 1))
    fi
done

echo ""

if [ "$downloaded" -eq 0 ]; then
    die "No books downloaded. Check your internet connection."
fi

if [ "$failed" -gt 0 ]; then
    warn "${failed} download(s) failed, continuing with ${downloaded} book(s)."
fi

# Report raw data size
raw_size=$(du -sh "$RAW_DIR" | cut -f1)
info "Downloaded ${downloaded} books (${raw_size} total)"

# Strip Gutenberg boilerplate (header/footer) from each file.
# Gutenberg texts have "*** START OF THE PROJECT GUTENBERG EBOOK" and
# "*** END OF THE PROJECT GUTENBERG EBOOK" markers.
info "Stripping Project Gutenberg boilerplate..."

CLEAN_DIR="${OUTPUT_DIR}/clean"
mkdir -p "$CLEAN_DIR"

for f in "$RAW_DIR"/*.txt; do
    base="$(basename "$f")"
    out="${CLEAN_DIR}/${base}"

    # Extract text between START and END markers, or keep entire file if
    # markers are not found.
    python3 -c "
import sys, re
text = open(sys.argv[1], 'r', encoding='utf-8', errors='replace').read()
start = re.search(r'\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text)
end = re.search(r'\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', text)
if start and end:
    text = text[start.end():end.start()]
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(text.strip())
" "$f" "$out"
done

echo "  Cleaned ${downloaded} file(s)"

# Tokenize with prepare.py
info "Tokenizing into binary shards..."
echo ""

python3 "${SCRIPT_DIR}/prepare.py" \
    --input "$CLEAN_DIR" \
    --output "$OUTPUT_DIR" \
    --vocab-size 1024 \
    --val-split 0.1

echo ""

# Summary
info "Done! Example data is ready."
echo ""
echo "  Train shards: ${OUTPUT_DIR}/train_*.bin"
echo "  Val shards:   ${OUTPUT_DIR}/val_*.bin"
echo "  Tokenizer:    ${OUTPUT_DIR}/tokenizer.json"
echo ""
echo "To train a model:"
echo "  mixlab -mode arch -config examples/plain_3L.json \\"
echo "    -train '${OUTPUT_DIR}/train_*.bin'"
