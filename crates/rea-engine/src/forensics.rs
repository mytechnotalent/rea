//! Firmware Security Forensics and Vulnerability Catalog
//!
//! **File:** `forensics.rs`
//! **Author:** Kevin Thomas
//! **Date:** 2026
//!
//! MIT License
//!
//! Copyright (c) 2026 Kevin Thomas
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

// Import dependencies from std
use std::fmt::Write;

/// A structured vulnerability finding with evidence and remediation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Vulnerability {
    /// Finding identifier (e.g. `M1`).
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// CWE classification numbers.
    pub cwe: Vec<String>,
    /// Associated CVE mirrors.
    pub cve_mirror: Vec<String>,
    /// Severity level.
    pub severity: String,
    /// Network or local reachability vector.
    pub reachability: String,
    /// Whether authentication is required.
    pub auth_required: bool,
    /// Applicable difficulty tiers.
    pub difficulty: Vec<String>,
    /// Network or API endpoint.
    pub endpoint: String,
    /// Code-verified decompiled evidence.
    pub evidence: String,
    /// Threat impact.
    pub impact: String,
    /// Recommended fix.
    pub remediation: String,
}

/// An end-to-end exploit chain connecting primitives into execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploitChain {
    /// Chain identifier (e.g. `Chain A`).
    pub id: String,
    /// Descriptive name.
    pub title: String,
    /// Attack vector explanation.
    pub description: String,
}

/// Compact definition of a code-verified vulnerability finding.
struct VulnDef {
    /// Identifier such as `M1`.
    id: &'static str,
    /// Human-readable title.
    title: &'static str,
    /// Severity rating.
    sev: &'static str,
    /// Network or execution reachability.
    reach: &'static str,
    /// Whether authentication is required.
    auth: bool,
    /// Associated CWE IDs.
    cwe: &'static [&'static str],
    /// Mirrored CVE references.
    cve: &'static [&'static str],
    /// Target endpoint or path.
    endpoint: &'static str,
    /// Code evidence string.
    evidence: &'static str,
    /// Threat impact statement.
    impact: &'static str,
    /// Remediation recommendation.
    remediation: &'static str,
}

/// All 19 static vulnerability definitions.
const VULNS: [VulnDef; 19] = [
    VulnDef {
        id: "M1",
        title: "ping.cgi unauthenticated command injection",
        sev: "Critical",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-78"],
        cve: &["CVE-2019-17621", "CVE-2016-6277"],
        endpoint: "/cgi-bin/ping.cgi?host=<INJECT>",
        evidence: "recovered_LAB_004007c4: snprintf(cmd, 0x200, \"ping -c %s %s > %s 2>&1\"); system(cmd)",
        impact: "root command execution",
        remediation: "replace system() with execvp and argv validation",
    },
    VulnDef {
        id: "M2",
        title: "upnpd IGD AddPortMapping command injection",
        sev: "Critical",
        reach: "network:udp1900,tcp5000",
        auth: false,
        cwe: &["CWE-78"],
        cve: &[],
        endpoint: "POST :5000/ctl/IPConn AddPortMapping NewProtocol",
        evidence: "FUN_00400f14 (upnpd): snprintf(cmd, 0x200, \"iptables ... -p %s ...\"); system(cmd)",
        impact: "root command execution",
        remediation: "validate protocol against allow-list; use execvp",
    },
    VulnDef {
        id: "M3",
        title: "cwmpd TR-069 auth bypass + wget injection",
        sev: "Critical",
        reach: "network:tcp7547",
        auth: false,
        cwe: &["CWE-287", "CWE-78"],
        cve: &[],
        endpoint: "POST :7547 cwmp:Download with URL",
        evidence: "FUN_00400bc8 (cwmpd): low tier forces auth; snprintf(cmd, \"wget -q %s ...\"); system(cmd)",
        impact: "root command execution / SSRF",
        remediation: "enforce TR-069 authentication and validate URL scheme",
    },
    VulnDef {
        id: "M4",
        title: "setcfg.cgi stack buffer overflow",
        sev: "High",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-120"],
        cve: &["CVE-2024-2850", "CVE-2022-40869"],
        endpoint: "/cgi-bin/setcfg.cgi?name=<VALUE>",
        evidence: "recovered_LAB_00400930: memcpy into 64-byte stack buffer with attacker length",
        impact: "DoS / potential root code execution",
        remediation: "enforce strict bounds check on name parameter",
    },
    VulnDef {
        id: "M19",
        title: "openvpn.cgi profile import command execution",
        sev: "Critical",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-78"],
        cve: &[],
        endpoint: "POST /cgi-bin/openvpn.cgi action=import&ovpn_config=up <cmd>",
        evidence: "execute loop (LAB_0040119e): system() runs up/down directives from uploaded config",
        impact: "unauthenticated root command execution",
        remediation: "disallow script execution directives in imported profiles",
    },
    VulnDef {
        id: "M5",
        title: "Hardcoded backdoor account in authcheck",
        sev: "High",
        reach: "web login / local",
        auth: false,
        cwe: &["CWE-798"],
        cve: &["CVE-2018-10824"],
        endpoint: "/cgi-bin/login.cgi (user=support)",
        evidence: "authcheck 0x4004d4: strcmp(user,\"support\") && strcmp(pass,\"sup3rs3cr3t_backd00r\")",
        impact: "authentication bypass to admin UI",
        remediation: "remove hardcoded credentials and hash with argon2",
    },
    VulnDef {
        id: "M6",
        title: "admin.cgi client-controlled authentication bypass",
        sev: "High",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-306", "CWE-287"],
        cve: &[],
        endpoint: "/cgi-bin/admin.cgi",
        evidence: "recovered_LAB_004007d2: accepts session=letmein_admin_2024 or authorized=1",
        impact: "admin panel access and secret disclosure",
        remediation: "implement server-side session management with HttpOnly tokens",
    },
    VulnDef {
        id: "M7",
        title: "view.cgi path traversal",
        sev: "High",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-22"],
        cve: &["CVE-2018-10822"],
        endpoint: "/cgi-bin/view.cgi?file=../../etc/device.conf",
        evidence: "recovered_LAB_00400750: fopen(\"%s/%s\", \"/www\", file); medium only strips first ../",
        impact: "arbitrary file read",
        remediation: "canonicalize path and restrict to root directory",
    },
    VulnDef {
        id: "M8",
        title: "getcfg.cgi cleartext secret export and IDOR",
        sev: "High",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-200", "CWE-312", "CWE-639"],
        cve: &[],
        endpoint: "/cgi-bin/getcfg.cgi?user=admin",
        evidence: "recovered_LAB_00400786: guest setting user=admin receives /etc/device.conf",
        impact: "admin credential and Wi-Fi PSK disclosure",
        remediation: "enforce server-side role check and encrypt secrets",
    },
    VulnDef {
        id: "M9",
        title: "api.cgi BOLA, mass assignment, and CSRF",
        sev: "High",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-639", "CWE-915", "CWE-352"],
        cve: &[],
        endpoint: "/cgi-bin/api.cgi?action=config",
        evidence: "recovered_LAB_00400a3c: _enable_telnet_ honored from body; authorized=1 grant",
        impact: "privilege escalation and state change",
        remediation: "whitelist permitted parameters and require CSRF token",
    },
    VulnDef {
        id: "M10",
        title: "USB automount filesystem-label traversal to root RCE",
        sev: "High",
        reach: "physical",
        auth: false,
        cwe: &["CWE-22", "CWE-73"],
        cve: &[],
        endpoint: "ext4 label ../../www + cgi-bin/sh",
        evidence: "/etc/hotplug.d/block/10-mount: mkdir -p \"$MOUNT_ROOT/$label\"; mount over /www",
        impact: "unauthenticated root command execution over HTTP",
        remediation: "sanitize volume label against path separators",
    },
    VulnDef {
        id: "M11",
        title: "Weak MD5-crypt password hashing ($1$ scheme)",
        sev: "Medium",
        reach: "local",
        auth: false,
        cwe: &["CWE-916"],
        cve: &[],
        endpoint: "/etc/shadow.vuln, /etc/uci-defaults/96",
        evidence: "admin:$1$vX7kLm2p$gPNcZy3f3hYG.e9nNpcDm0 cracks to \"admin\" via md5crypt",
        impact: "offline credential cracking",
        remediation: "migrate password storage to sha512crypt or argon2id",
    },
    VulnDef {
        id: "M12",
        title: "Shipped factory TLS private keys",
        sev: "Medium",
        reach: "network:443",
        auth: false,
        cwe: &["CWE-321", "CWE-295"],
        cve: &[],
        endpoint: "/etc/ssl/certs/malgorouter_factory.key, /etc/uhttpd.key",
        evidence: "RSA modulus matches shipped certificate; all units share identical key material",
        impact: "HTTPS decryption and device impersonation",
        remediation: "generate unique per-device keys at first boot",
    },
    VulnDef {
        id: "M13",
        title: "Predictable session tokens",
        sev: "Medium",
        reach: "network:80,443",
        auth: false,
        cwe: &["CWE-330", "CWE-338"],
        cve: &[],
        endpoint: "/cgi-bin/login.cgi, /cgi-bin/api.cgi",
        evidence: "low tier: srand(pid<<16 ^ time); rand(); medium tier: sess_%08x_%04x",
        impact: "session prediction and hijacking",
        remediation: "use /dev/urandom CSPRNG for all token generation",
    },
    VulnDef {
        id: "M14",
        title: "Unsigned firmware update with tar traversal",
        sev: "Medium",
        reach: "network:80,443",
        auth: true,
        cwe: &["CWE-494", "CWE-22"],
        cve: &[],
        endpoint: "/cgi-bin/firmware.cgi (magic VMNFW1)",
        evidence: "system(\"tail -c +8 ... | tar -xf - -C /tmp/fw_staging\"); no signature check",
        impact: "arbitrary root file overwrite via ../ tar entries",
        remediation: "enforce cryptographic signature check before extraction",
    },
    VulnDef {
        id: "M15",
        title: "Fabricated component SBOM trap",
        sev: "Medium",
        reach: "analysis",
        auth: false,
        cwe: &["CWE-1104"],
        cve: &["CVE-2014-0160", "CVE-2017-16544"],
        endpoint: "/www/sbom/malgorouter.cdx.json",
        evidence: "Planted SBOM claims BusyBox 1.27.0/OpenSSL 1.0.1f; actual opkg is 1.36.1/3.0.16",
        impact: "analyst mis-triage and false positives",
        remediation: "verify software bill of materials against package database",
    },
    VulnDef {
        id: "M16",
        title: "Default credentials and unencrypted HTTP transport",
        sev: "Medium",
        reach: "network:80",
        auth: false,
        cwe: &["CWE-1391", "CWE-319"],
        cve: &[],
        endpoint: "HTTP :80, SSH :22",
        evidence: "admin/admin, guest/guest; uhttpd redirect_https 0; dropbear password auth enabled",
        impact: "credential guessing and cleartext traffic interception",
        remediation: "enforce initial password change and mandatory HTTPS redirect",
    },
    VulnDef {
        id: "M17",
        title: "Remote assistance executes helper with user code",
        sev: "Low",
        reach: "network:80,443",
        auth: false,
        cwe: &[],
        cve: &[],
        endpoint: "/cgi-bin/remoteassist.cgi",
        evidence: "execl(\"/usr/sbin/magicd\", code); XOR-0x5a comparison leaks CTF flag",
        impact: "diagnostic flag disclosure",
        remediation: "remove diagnostic backdoor helpers from production firmware",
    },
    VulnDef {
        id: "M18",
        title: "Weak factory Wi-Fi PSK and residual shadow file",
        sev: "Low",
        reach: "physical",
        auth: false,
        cwe: &["CWE-521"],
        cve: &[],
        endpoint: "/etc/uci-defaults/98-malgorouter-wifi, /etc/shadow.vuln",
        evidence: "Pre-shared keys goodpass123 and guestpass123 configured in uci-defaults",
        impact: "unauthorized wireless network association",
        remediation: "generate unique WPA2/3 pre-shared keys per device",
    },
];

/// Builds an owned vulnerability record.
///
/// Builds initial vulnerability record.
///
/// # Arguments
///
/// * `id` - Finding ID.
/// * `title` - Title.
/// * `sev` - Severity.
///
/// # Returns
///
/// Base vulnerability.
fn base_with_id(id: &str, title: &str, sev: &str) -> Vulnerability {
    Vulnerability {
        id: id.to_string(),
        title: title.to_string(),
        severity: sev.to_string(),
        difficulty: vec!["low".to_string()],
        ..Default::default()
    }
}

/// Builds an owned vulnerability record.
///
/// # Arguments
///
/// * `id` - Finding ID.
/// * `title` - Title.
/// * `sev` - Severity.
/// * `reach` - Reachability.
/// * `auth` - Auth requirement.
///
/// # Returns
///
/// Initialized vulnerability with empty vectors.
fn make_base(id: &str, title: &str, sev: &str, reach: &str, auth: bool) -> Vulnerability {
    let mut v = base_with_id(id, title, sev);
    v.reachability = reach.to_string();
    v.auth_required = auth;
    v
}

/// Populates CWE, CVE, and endpoint fields of a vulnerability.
///
/// # Arguments
///
/// * `v` - Vulnerability to update.
/// * `cwe` - CWE strings.
/// * `cve` - CVE strings.
/// * `ep` - Endpoint string.
fn fill_meta(v: &mut Vulnerability, cwe: &[&str], cve: &[&str], ep: &str) {
    v.cwe = cwe.iter().map(ToString::to_string).collect();
    v.cve_mirror = cve.iter().map(ToString::to_string).collect();
    v.endpoint = ep.to_string();
}

/// Populates evidence, impact, and remediation fields.
///
/// # Arguments
///
/// * `v` - Vulnerability to update.
/// * `ev` - Evidence text.
/// * `imp` - Impact text.
/// * `rem` - Remediation text.
fn fill_details(v: &mut Vulnerability, ev: &str, imp: &str, rem: &str) {
    v.evidence = ev.to_string();
    v.impact = imp.to_string();
    v.remediation = rem.to_string();
}

/// Converts a static definition into an owned vulnerability.
///
/// # Arguments
///
/// * `d` - Static vulnerability definition.
///
/// # Returns
///
/// Fully populated vulnerability.
fn from_def(d: &VulnDef) -> Vulnerability {
    let mut v = make_base(d.id, d.title, d.sev, d.reach, d.auth);
    fill_meta(&mut v, d.cwe, d.cve, d.endpoint);
    fill_details(&mut v, d.evidence, d.impact, d.remediation);
    v
}

/// Constructs finding M1 for testing.
///
/// # Returns
///
/// Finding M1.
#[cfg(test)]
fn vuln_m1() -> Vulnerability {
    from_def(&VULNS[0])
}

/// Returns the complete code-verified vulnerability catalog.
///
/// # Returns
///
/// All 19 discovered vulnerabilities.
#[must_use]
pub fn all_vulnerabilities() -> Vec<Vulnerability> {
    VULNS.iter().map(from_def).collect()
}

/// Verified multi-step exploit chains definitions.
const CHAINS: [(&str, &str, &str); 7] = [
    ("Chain A", "LAN Web to Root RCE", "GET /cgi-bin/ping.cgi?host=127.0.0.1;<payload> executes system() as root without authentication."),
    ("Chain A2", "LAN Web to Root RCE via Profile Import", "POST /cgi-bin/openvpn.cgi with authorized=1 and ovpn_config=up <cmd> triggers root execution."),
    ("Chain B", "LAN UPnP to Root RCE", "SOAP AddPortMapping with NewProtocol=tcp;<payload>; to upnpd:5000 injects into iptables system call."),
    ("Chain C", "LAN TR-069 to Root RCE", "cwmp:Download to cwmpd:7547 with shell metacharacters in URL passes unvalidated to wget system call."),
    ("Chain D", "Web Auth Bypass to Admin", "Cookie authorized=1 bypasses admin.cgi, api.cgi, and unlocks cleartext credentials in getcfg.cgi."),
    ("Chain E", "Physical USB to Root RCE", "Ext4 drive labeled ../../www replaces web root on automount, granting HTTP shell access."),
    ("Chain F", "TLS Private Key MITM", "Shared factory private keys allow passive decryption and active impersonation of HTTPS traffic."),
];

/// Constructs an exploit chain record.
///
/// # Arguments
///
/// * `id` - Identifier.
/// * `title` - Title.
/// * `desc` - Description.
///
/// # Returns
///
/// Exploit chain.
fn chain(id: &str, title: &str, desc: &str) -> ExploitChain {
    ExploitChain { id: id.to_string(), title: title.to_string(), description: desc.to_string() }
}

/// Returns all verified multi-step exploit chains.
///
/// # Returns
///
/// Exploit chains A through F.
#[must_use]
pub fn all_exploit_chains() -> Vec<ExploitChain> {
    CHAINS.iter().map(|&(i, t, d)| chain(i, t, d)).collect()
}

/// Formats the opening header of findings JSON.
///
/// # Arguments
///
/// * `target` - Target name.
/// * `sha256` - Target hash.
///
/// # Returns
///
/// Header JSON string.
fn json_header(target: &str, sha256: &str) -> String {
    format!("{{\n  \"schema\": \"malgorouter.findings/v1\",\n  \"target\": {{\n    \"firmware\": \"{target}\",\n    \"sha256\": \"{sha256}\"\n  }},\n  \"findings\": [\n")
}

/// Appends classification fields to a JSON buffer.
///
/// # Arguments
///
/// * `s` - Destination buffer.
/// * `v` - Source vulnerability.
fn append_sec_fields(s: &mut String, v: &Vulnerability) {
    let sev = v.severity.to_lowercase();
    let _ = writeln!(
        s,
        "      \"severity\": \"{sev}\",\n      \"reachability\": \"{}\",",
        v.reachability
    );
}

/// Appends authentication and endpoint fields to a JSON buffer.
///
/// # Arguments
///
/// * `s` - Destination buffer.
/// * `v` - Source vulnerability.
fn append_auth_fields(s: &mut String, v: &Vulnerability) {
    let _ = writeln!(
        s,
        "      \"auth_required\": {},\n      \"endpoint\": \"{}\",",
        v.auth_required, v.endpoint
    );
}

/// Appends impact and remediation fields to a JSON buffer.
///
/// # Arguments
///
/// * `s` - Destination buffer.
/// * `v` - Source vulnerability.
fn append_impact_fields(s: &mut String, v: &Vulnerability) {
    let _ = write!(
        s,
        "      \"impact\": \"{}\",\n      \"remediation\": \"{}\"\n    }}",
        v.impact, v.remediation
    );
}

/// Appends vulnerability fields to a JSON string buffer.
///
/// # Arguments
///
/// * `s` - Destination buffer.
/// * `v` - Source vulnerability.
fn append_json_fields(s: &mut String, v: &Vulnerability) {
    append_sec_fields(s, v);
    append_auth_fields(s, v);
    append_impact_fields(s, v);
}

/// Converts one vulnerability to its JSON object string.
///
/// # Arguments
///
/// * `v` - Vulnerability to serialize.
///
/// # Returns
///
/// JSON fragment.
fn vuln_to_json(v: &Vulnerability) -> String {
    let mut s = format!(
        "    {{\n      \"id\": \"{}\",\n      \"title\": \"{}\",\n",
        v.id, v.title
    );
    append_json_fields(&mut s, v);
    s
}

/// Serializes vulnerabilities into the standardized `findings.json` schema.
///
/// # Arguments
///
/// * `vulns` - Discovered vulnerabilities.
/// * `target` - Target binary or firmware name.
/// * `sha256` - Target SHA-256 hash.
///
/// # Returns
///
/// Formatted JSON text.
#[must_use]
pub fn format_findings_json(vulns: &[Vulnerability], target: &str, sha256: &str) -> String {
    let mut out = json_header(target, sha256);
    let items: Vec<String> = vulns.iter().map(vuln_to_json).collect();
    out.push_str(&items.join(",\n"));
    out.push_str("\n  ]\n}\n");
    out
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies that all 19 findings are generated.
    #[test]
    fn test_all_vulnerabilities_count() {
        let vulns = all_vulnerabilities();
        assert_eq!(vulns.len(), 19);
        assert_eq!(vulns[0].id, "M1");
        assert_eq!(vulns[4].id, "M19");
    }

    /// Verifies exploit chains generation.
    #[test]
    fn test_exploit_chains_count() {
        let chains = all_exploit_chains();
        assert_eq!(chains.len(), 7);
        assert_eq!(chains[0].id, "Chain A");
    }

    /// Verifies findings JSON formatting.
    #[test]
    fn test_format_findings_json() {
        let vulns = vec![vuln_m1()];
        let json = format_findings_json(&vulns, "fw.bin", "aabbcc");
        assert!(json.contains("malgorouter.findings/v1"));
        assert!(json.contains("M1"));
    }
}
