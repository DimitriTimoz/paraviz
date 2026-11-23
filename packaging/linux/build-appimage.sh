#!/usr/bin/env bash
set -euo pipefail

# Build a Linux AppImage for ParaViz.
# Usage: ./packaging/linux/build-appimage.sh <path-to-paraviz-binary> <arch: x86_64|aarch64> <output-AppImage-path>

BIN_PATH="${1:?Usage: build-appimage.sh <binary> <arch> <output.AppImage>}"
ARCH="${2:-$(uname -m)}"
OUT_APPIMAGE="${3:-ParaViz-${ARCH}.AppImage}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPDIR="${ROOT_DIR}/target/AppDir-${ARCH}"

rm -rf "${APPDIR}"
mkdir -p \
  "${APPDIR}/usr/bin" \
  "${APPDIR}/usr/share/applications" \
  "${APPDIR}/usr/share/icons/hicolor/scalable/apps" \
  "${APPDIR}/usr/share/icons/hicolor/256x256/apps" \
  "${APPDIR}/usr/share/metainfo"

cp "${BIN_PATH}" "${APPDIR}/usr/bin/paraviz"
chmod +x "${APPDIR}/usr/bin/paraviz"

cp "${ROOT_DIR}/packaging/linux/io.github.dimitritimoz.paraviz.desktop" \
   "${APPDIR}/io.github.dimitritimoz.paraviz.desktop"
cp "${ROOT_DIR}/packaging/linux/io.github.dimitritimoz.paraviz.desktop" \
   "${APPDIR}/usr/share/applications/io.github.dimitritimoz.paraviz.desktop"

cp "${ROOT_DIR}/packaging/linux/io.github.dimitritimoz.paraviz.appdata.xml" \
   "${APPDIR}/usr/share/metainfo/io.github.dimitritimoz.paraviz.appdata.xml"

cp "${ROOT_DIR}/assets/icon.svg" "${APPDIR}/paraviz.svg"
cp "${ROOT_DIR}/assets/icon.svg" "${APPDIR}/usr/share/icons/hicolor/scalable/apps/paraviz.svg"

if command -v rsvg-convert >/dev/null 2>&1; then
  rsvg-convert -w 256 -h 256 "${ROOT_DIR}/assets/icon.svg" -o "${APPDIR}/paraviz.png"
  cp "${APPDIR}/paraviz.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/paraviz.png"
  ln -sf paraviz.png "${APPDIR}/.DirIcon"
elif command -v convert >/dev/null 2>&1; then
  convert -background none -resize 256x256 "${ROOT_DIR}/assets/icon.svg" "${APPDIR}/paraviz.png"
  cp "${APPDIR}/paraviz.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/paraviz.png"
  ln -sf paraviz.png "${APPDIR}/.DirIcon"
else
  ln -sf paraviz.svg "${APPDIR}/.DirIcon"
fi

cat > "${APPDIR}/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/paraviz" "$@"
EOF
chmod +x "${APPDIR}/AppRun"

TOOL_ARCH="${ARCH}"
APPIMAGETOOL="${APPIMAGETOOL:-$(command -v appimagetool || true)}"
if [ -z "${APPIMAGETOOL}" ]; then
  APPIMAGETOOL="${ROOT_DIR}/target/appimagetool-${TOOL_ARCH}.AppImage"
  if [ ! -x "${APPIMAGETOOL}" ]; then
    curl -fSL \
      "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${TOOL_ARCH}.AppImage" \
      -o "${APPIMAGETOOL}"
    chmod +x "${APPIMAGETOOL}"
  fi
fi

# Run appimagetool with --appimage-extract-and-run so FUSE is not required in Docker/CI runners
ARCH="${TOOL_ARCH}" "${APPIMAGETOOL}" --appimage-extract-and-run "${APPDIR}" "${OUT_APPIMAGE}"
chmod +x "${OUT_APPIMAGE}"
echo "Built AppImage: ${OUT_APPIMAGE}"
