# MalgoRouter PoCs

Proof-of-concept scripts for the findings in `../ANALYSIS.md`. **LAB ONLY** — run these
only against the MalgoRouter training firmware on an isolated network you own. The
firmware is a deliberately-vulnerable training target; several of these give **root**.

Set the target once:

```sh
export TARGET=192.168.75.1        # LAN IP of the MalgoRouter
export ATTACKER=192.168.75.100    # your host, for reverse shells / callbacks
```

| script | finding | auth | port |
|---|---|---|---|
| `01-ping-rce.sh`       | M1 ping.cgi OS command injection | none | 80 |
| `02-view-traversal.sh` | M7 view.cgi path traversal | none | 80 |
| `03-admin-bypass.sh`   | M6 admin.cgi cookie auth bypass | none | 80 |
| `04-getcfg-idor.sh`    | M8 getcfg.cgi secret export | none | 80 |
| `05-setcfg-overflow.sh`| M4 setcfg.cgi stack overflow | none | 80 |
| `06-api-massassign.sh` | M9 api.cgi BOLA / mass assignment | none | 80 |
| `07-cwmpd-rce.sh`      | M3 cwmpd TR-069 auth bypass + wget injection | none (low) | 7547 |
| `08-upnpd-rce.sh`      | M2 upnpd IGD iptables injection | none | 5000 |
| `09-remoteassist-flag.sh`| M17 magicd flag | authorized=1 | 80 |
| `10-usb-label-rce.sh`  | M10 USB filesystem-label traversal | physical | — |
| `11-openvpn-import-rce.sh`| M19 openvpn.cgi profile import RCE | authorized=1 | 80 |

Notes:
- Set the difficulty with the `difficulty=` cookie, or log in through the web UI and it is
  written server-side to `/tmp/malgorouter_difficulty`. The **low** tier is the
  unauthenticated attack; `01`, `02`, `05` work best at low.
- `06`, `07`, `09` chain the **M6 `authorized=1` bypass**.
