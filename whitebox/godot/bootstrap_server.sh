#!/usr/bin/env bash
# Standalone runtime only: no apt install, pip, conda, sudo, or H3 changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime
cd runtime
archive=Godot_v4.5-stable_linux.x86_64.zip
base=https://github.com/godotengine/godot/releases/download/4.5-stable
if [ ! -x Godot_v4.5-stable_linux.x86_64 ]; then
    curl -fL --retry 2 -o "$archive" "$base/$archive"
    unzip -o "$archive"
fi
chmod +x Godot_v4.5-stable_linux.x86_64
if ! command -v Xvfb >/dev/null; then
    apt-get download xvfb libxfont2 xserver-common
    for package in ./*.deb; do
        dpkg-deb -x "$package" x11
    done
fi
./Godot_v4.5-stable_linux.x86_64 --version
