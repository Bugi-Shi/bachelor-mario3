#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$ROOT_DIR/external/gym-retro"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake fehlt. Installiere Systemdeps z.B.:" >&2
  echo "  sudo apt update" >&2
  echo "  sudo apt-get install -y cmake git build-essential" >&2
  echo "  sudo apt-get install -y pkg-config" >&2
  echo "  sudo apt-get install -y libbz2-dev" >&2
  echo "  sudo apt-get install -y capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev zlib1g-dev" >&2
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "pkg-config fehlt (CMake findet sonst Cap'n Proto nicht -> Build bricht ab)." >&2
  echo "Installiere: sudo apt-get install -y pkg-config" >&2
  exit 1
fi

if [[ ! -f /usr/include/bzlib.h ]]; then
  echo "libbz2-dev fehlt (BZip2 headers)." >&2
  echo "Installiere: sudo apt-get install -y libbz2-dev" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git fehlt. Installiere: sudo apt-get install -y git" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/external"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "Cloning gym-retro source into: $SRC_DIR"
  git clone https://github.com/openai/retro "$SRC_DIR"
fi

# Some retro versions used git submodules; newer trees vendor deps without
# shipping a top-level .gitmodules. Only try submodule update if present.
if [[ -f "$SRC_DIR/.gitmodules" ]]; then
  echo "Updating git submodules ..."
  (cd "$SRC_DIR" && git submodule update --init --recursive)
fi

if [[ ! -f "$SRC_DIR/cores/nes/Makefile" && ! -f "$SRC_DIR/cores/nes/Makefile.libretro" && ! -f "$SRC_DIR/cores/nes/libretro/Makefile" ]]; then
  echo "ERROR: Missing cores/nes Makefile(s)." >&2
  echo "The source tree looks incomplete. Easiest fix:" >&2
  echo "  rm -rf external/gym-retro" >&2
  echo "  bash tools/retro_integration/build.sh" >&2
  exit 2
fi

echo "Configuring (BUILD_UI=ON) ..."
echo "Hinweis: gym-retro nutzt relative Pfade (cores/*/Makefile), daher in-source build."
pushd "$SRC_DIR" >/dev/null

# Default: build only the NES core (enough for SMB3) to avoid failures in
# other bundled cores on newer compilers.
if ! grep -q "BACHELOR_MIN_CORES" "$SRC_DIR/CMakeLists.txt"; then
  echo "Patching CMakeLists.txt to disable non-NES cores (local only) ..."

  # Comment out core builds we don't need. Keep NES (fceumm) enabled.
  perl -pi -e 's/^\s*add_core\(\s*snes\s+snes9x\s*\)\s*$/# add_core(snes snes9x)  # BACHELOR_MIN_CORES/g' CMakeLists.txt
  perl -pi -e 's/^\s*add_core\(\s*genesis\s+genesis_plus_gx\s*\)\s*$/# add_core(genesis genesis_plus_gx)  # BACHELOR_MIN_CORES/g' CMakeLists.txt
  perl -pi -e 's/^\s*add_core\(\s*atari2600\s+stella\s*\)\s*$/# add_core(atari2600 stella)  # BACHELOR_MIN_CORES/g' CMakeLists.txt
  perl -pi -e 's/^\s*add_core\(\s*gb\s+gambatte\s*\)\s*$/# add_core(gb gambatte)  # BACHELOR_MIN_CORES/g' CMakeLists.txt
  perl -pi -e 's/^\s*add_core\(\s*gba\s+mgba\s*\)\s*$/# add_core(gba mgba)  # BACHELOR_MIN_CORES/g' CMakeLists.txt
  perl -pi -e 's/^\s*add_core\(\s*pce\s+mednafen_pce_fast\s*\)\s*$/# add_core(pce mednafen_pce_fast)  # BACHELOR_MIN_CORES/g' CMakeLists.txt

  # Force a clean re-configure so the removed targets disappear.
  rm -f CMakeCache.txt
  rm -rf CMakeFiles
  rm -f Makefile
fi

# If a previous configure happened without pkg-config / system deps, CMake may
# have cached the fallback third-party settings. Clean so it can re-detect.
if [[ -f CMakeCache.txt ]]; then
  if grep -q "PKG_CONFIG_EXECUTABLE:FILEPATH=PKG_CONFIG_EXECUTABLE-NOTFOUND" CMakeCache.txt 2>/dev/null; then
    echo "Cleaning CMake cache (pkg-config was missing in previous configure) ..."
    rm -f CMakeCache.txt
    rm -rf CMakeFiles
    rm -f Makefile
  fi
fi

cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY

echo "Building ..."
cmake --build . --target gym-retro-integration -j"$(nproc)"
popd >/dev/null

echo "OK: built integration tool: $SRC_DIR/gym-retro-integration"
