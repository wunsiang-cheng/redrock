#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
architecture=$(dpkg --print-architecture)
output="target/release/redrock_${version}_${architecture}.deb"
package_root=$(mktemp -d)
trap 'rm -rf "$package_root"' EXIT

cargo build --release --locked
install -Dm755 target/release/redrock "$package_root/usr/bin/redrock"
install -Dm644 packaging/redrock.desktop "$package_root/usr/share/applications/redrock.desktop"
install -Dm644 packaging/redrock.svg "$package_root/usr/share/icons/hicolor/scalable/apps/redrock.svg"
mkdir -p "$package_root/DEBIAN"
cat > "$package_root/DEBIAN/control" <<EOF
Package: redrock
Version: $version
Architecture: $architecture
Maintainer: wunsiang-cheng
Depends: libc6 (>= 2.39), libgcc-s1
Section: utils
Priority: optional
Homepage: https://github.com/wunsiang-cheng/redrock
Description: Local autonomous AI agent connected to Telegram
 RedRock runs on the user's machine and provides graphical and CLI setup.
EOF

dpkg-deb --root-owner-group --build "$package_root" "$output"
printf '%s\n' "$output"
