#!/usr/bin/env bash
# zipselect.sh — interactive zip builder using gum, respects .gitignore
# Dependencies: gum, git (optional), zip

set -euo pipefail

# ── helpers ────────────────────────────────────────────────────────────────────

check_deps() {
  local missing=()
  for cmd in gum zip; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: missing required tools: ${missing[*]}"
    echo "Install gum: https://github.com/charmbracelet/gum#installation"
    exit 1
  fi
}

# Build the list of candidate files, honouring .gitignore when inside a git repo
list_files() {
  local root="${1:-.}"

  if git -C "$root" rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    # git ls-files: tracked + untracked non-ignored files
    git -C "$root" ls-files \
        --cached \
        --others \
        --exclude-standard \
      | sort
  else
    # Plain find, manually skip common ignored dirs
    find "$root" \
        -type f \
        \( \
          -path '*/.git/*' \
          -o -path '*/node_modules/*' \
          -o -path '*/__pycache__/*' \
          -o -path '*/.DS_Store' \
        \) -prune \
        -o -type f -print \
      | sed "s|^\./||" \
      | sort
  fi
}

# ── main ───────────────────────────────────────────────────────────────────────

check_deps

gum style \
  --border double \
  --border-foreground 212 \
  --padding "1 3" \
  --bold \
  "📦  zipselect — interactive zip builder"

# ── choose working directory ───────────────────────────────────────────────────
ROOT_DIR=$(gum input \
  --placeholder "Directory to search (default: current)" \
  --prompt "📁 Root dir › " \
  --value ".")

ROOT_DIR="${ROOT_DIR:-.}"

if [[ ! -d "$ROOT_DIR" ]]; then
  gum style --foreground 196 "Directory '$ROOT_DIR' not found."
  exit 1
fi

ROOT_DIR=$(realpath "$ROOT_DIR")

# ── collect file list ──────────────────────────────────────────────────────────
gum spin --spinner dot --title "Scanning files…" -- \
  bash -c "list_files() {
    local root=\"\${1:-.}\"
    if git -C \"\$root\" rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
      git -C \"\$root\" ls-files --cached --others --exclude-standard | sort
    else
      find \"\$root\" -type f \
        \\( -path '*/.git/*' -o -path '*/node_modules/*' \
           -o -path '*/__pycache__/*' -o -path '*/.DS_Store' \\) -prune \
        -o -type f -print | sed 's|^\./||' | sort
    fi
  }; list_files '$ROOT_DIR' > /tmp/_zipselect_files"

mapfile -t ALL_FILES < /tmp/_zipselect_files
rm -f /tmp/_zipselect_files

if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
  gum style --foreground 196 "No eligible files found in '$ROOT_DIR'."
  exit 1
fi

gum style --foreground 245 "Found ${#ALL_FILES[@]} eligible file(s)."

# ── optional filter ────────────────────────────────────────────────────────────
FILTER=$(gum input \
  --placeholder "Filter files (glob pattern, e.g. '*.go') — leave blank for all" \
  --prompt "🔍 Filter  › ")

if [[ -n "$FILTER" ]]; then
  FILTERED=()
  for f in "${ALL_FILES[@]}"; do
    # shellcheck disable=SC2254
    case "$f" in
      $FILTER) FILTERED+=("$f") ;;
    esac
  done
  ALL_FILES=("${FILTERED[@]}")
  if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    gum style --foreground 196 "No files match filter '$FILTER'."
    exit 1
  fi
  gum style --foreground 245 "${#ALL_FILES[@]} file(s) after filter."
fi

# ── interactive multi-select ───────────────────────────────────────────────────
echo ""
gum style --foreground 212 --bold "Select files to include  (space = toggle, enter = confirm):"
echo ""

mapfile -t SELECTED < <(
  printf '%s\n' "${ALL_FILES[@]}" \
  | gum choose \
      --no-limit \
      --height 20 \
      --cursor "▶ " \
      --selected-prefix "◉ " \
      --unselected-prefix "○ "
)

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  gum style --foreground 226 "Nothing selected — exiting."
  exit 0
fi

gum style --foreground 245 "${#SELECTED[@]} file(s) selected."

# ── choose / confirm output path ───────────────────────────────────────────────
DEFAULT_ZIP="archive_$(date +%Y%m%d_%H%M%S).zip"
OUTPUT_ZIP=$(gum input \
  --placeholder "$DEFAULT_ZIP" \
  --prompt "💾 Output zip › " \
  --value "$DEFAULT_ZIP")
OUTPUT_ZIP="${OUTPUT_ZIP:-$DEFAULT_ZIP}"

# Make absolute if relative
if [[ "$OUTPUT_ZIP" != /* ]]; then
  OUTPUT_ZIP="$(pwd)/$OUTPUT_ZIP"
fi

# Warn on overwrite
if [[ -f "$OUTPUT_ZIP" ]]; then
  gum confirm "⚠  '$OUTPUT_ZIP' already exists. Overwrite?" || exit 0
  rm -f "$OUTPUT_ZIP"
fi

# ── create zip ─────────────────────────────────────────────────────────────────
echo ""
gum style --foreground 212 --bold "Files to zip:"
printf '  %s\n' "${SELECTED[@]}" | gum style --foreground 245

echo ""
gum confirm "Create '$OUTPUT_ZIP'?" || { gum style --foreground 226 "Aborted."; exit 0; }

cd "$ROOT_DIR"

gum spin --spinner globe --title "Creating zip…" -- \
  zip -r "$OUTPUT_ZIP" "${SELECTED[@]}"

echo ""
gum style \
  --border rounded \
  --border-foreground 82 \
  --padding "1 3" \
  --foreground 82 \
  --bold \
  "✅  Done!  →  $OUTPUT_ZIP"

# Show quick summary
ZIP_SIZE=$(du -sh "$OUTPUT_ZIP" 2>/dev/null | cut -f1)
gum style --foreground 245 "Size: ${ZIP_SIZE}   Files: ${#SELECTED[@]}"