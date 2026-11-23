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
APPIMAGETOOL_VERSION="1.9.0"
case "${TOOL_ARCH}" in
  x86_64)
    APPIMAGETOOL_SHA256="e8c1169c4af66c3184d7ee110713d4be04befe671fa358e2c1591ee840553b03"
    ;;
  aarch64)
    APPIMAGETOOL_SHA256="d6a061996de690434996c2a523eafedd0a23780aaf02d69ce364537e71607e94"
    ;;
  *)
    echo "Unsupported architecture for appimagetool verification: ${TOOL_ARCH}" >&2
    exit 1
    ;;
esac

APPIMAGETOOL="${APPIMAGETOOL:-$(command -v appimagetool || true)}"
if [ -z "${APPIMAGETOOL}" ]; then
  APPIMAGETOOL="${ROOT_DIR}/target/appimagetool-${TOOL_ARCH}.AppImage"
  if [ ! -x "${APPIMAGETOOL}" ]; then
    curl -fSL \
      "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-${TOOL_ARCH}.AppImage" \
      -o "${APPIMAGETOOL}"
    echo "${APPIMAGETOOL_SHA256}  ${APPIMAGETOOL}" | sha256sum -c -
    chmod +x "${APPIMAGETOOL}"
  fi
fi

# Run appimagetool with --appimage-extract-and-run so FUSE is not required in Docker/CI runners
ARCH="${TOOL_ARCH}" "${APPIMAGETOOL}" --appimage-extract-and-run "${APPDIR}" "${OUT_APPIMAGE}"
chmod +x "${OUT_APPIMAGE}"
echo "Built AppImage: ${OUT_APPIMAGE}"
