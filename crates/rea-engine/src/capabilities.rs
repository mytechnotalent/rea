//! Capability Classification
//!
//! **File:** `capabilities.rs`
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

// Import crate::model
use crate::model::Import;

/// One capability rule: a behavior and the APIs that implement it.
pub struct Rule {
    /// Capability category.
    pub category: &'static str,

    /// What the capability lets the program do.
    pub description: &'static str,

    /// API name prefixes that indicate the capability.
    pub apis: &'static [&'static str],
}

/// A capability supported by the target's imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capability {
    /// Capability category.
    pub category: &'static str,

    /// What the capability lets the program do.
    pub description: &'static str,

    /// Imported APIs that matched, sorted and unique.
    pub apis: Vec<String>,
}

/// Capability rules evaluated against the import table.
pub const RULES: [Rule; 16] = [
    Rule {
        category: "Process injection",
        description: "Allocates, writes, or starts code inside other processes",
        apis: &[
            "VirtualAllocEx",
            "WriteProcessMemory",
            "CreateRemoteThread",
            "NtCreateThreadEx",
            "RtlCreateUserThread",
            "QueueUserAPC",
            "SetThreadContext",
            "NtUnmapViewOfSection",
            "NtWriteVirtualMemory",
            "NtMapViewOfSection",
        ],
    },
    Rule {
        category: "Process execution",
        description: "Launches other programs or shell commands",
        apis: &[
            "CreateProcess",
            "ShellExecute",
            "WinExec",
            "system",
            "execve",
            "execv",
            "execl",
            "popen",
            "posix_spawn",
            "fork",
        ],
    },
    Rule {
        category: "Network communication",
        description: "Opens sockets or performs HTTP and other network transfers",
        apis: &[
            "WSAStartup",
            "socket",
            "connect",
            "send",
            "recv",
            "bind",
            "listen",
            "accept",
            "gethostbyname",
            "getaddrinfo",
            "InternetOpen",
            "InternetConnect",
            "InternetReadFile",
            "HttpOpenRequest",
            "HttpSendRequest",
            "WinHttpOpen",
            "WinHttpConnect",
            "WinHttpSendRequest",
            "URLDownloadToFile",
            "curl_easy",
        ],
    },
    Rule {
        category: "Cryptography",
        description: "Uses encryption, hashing, or key management primitives",
        apis: &[
            "CryptAcquireContext",
            "CryptEncrypt",
            "CryptDecrypt",
            "CryptGenKey",
            "CryptImportKey",
            "CryptHashData",
            "CryptProtectData",
            "CryptUnprotectData",
            "BCrypt",
            "NCrypt",
            "EVP_",
            "AES_",
            "RSA_",
            "CCCrypt",
        ],
    },
    Rule {
        category: "Registry access",
        description: "Reads or modifies the Windows registry",
        apis: &[
            "RegOpenKey",
            "RegCreateKey",
            "RegSetValue",
            "RegQueryValue",
            "RegDeleteKey",
            "RegDeleteValue",
            "RegEnumKey",
            "RegEnumValue",
        ],
    },
    Rule {
        category: "File system",
        description: "Creates, modifies, deletes, or enumerates files",
        apis: &[
            "CreateFile",
            "WriteFile",
            "DeleteFile",
            "MoveFile",
            "CopyFile",
            "FindFirstFile",
            "FindNextFile",
            "fopen",
            "unlink",
            "rename",
        ],
    },
    Rule {
        category: "Anti-analysis",
        description: "Detects debuggers or hides from analysis",
        apis: &[
            "IsDebuggerPresent",
            "CheckRemoteDebuggerPresent",
            "NtQueryInformationProcess",
            "OutputDebugString",
            "NtSetInformationThread",
            "ptrace",
        ],
    },
    Rule {
        category: "Dynamic API resolution",
        description: "Loads libraries and resolves functions at run time",
        apis: &[
            "LoadLibrary",
            "GetProcAddress",
            "LdrLoadDll",
            "LdrGetProcedureAddress",
            "dlopen",
            "dlsym",
        ],
    },
    Rule {
        category: "Service control",
        description: "Creates, starts, or reconfigures system services",
        apis: &[
            "OpenSCManager",
            "CreateService",
            "StartService",
            "ControlService",
            "DeleteService",
            "ChangeServiceConfig",
        ],
    },
    Rule {
        category: "Privilege manipulation",
        description: "Adjusts tokens, privileges, or user identity",
        apis: &[
            "AdjustTokenPrivileges",
            "OpenProcessToken",
            "LookupPrivilegeValue",
            "ImpersonateLoggedOnUser",
            "DuplicateToken",
            "setuid",
            "setgid",
        ],
    },
    Rule {
        category: "Input capture",
        description: "Installs input hooks or polls keyboard state",
        apis: &[
            "SetWindowsHookEx",
            "GetAsyncKeyState",
            "GetKeyboardState",
            "RegisterRawInputDevices",
        ],
    },
    Rule {
        category: "Screen capture",
        description: "Copies screen or window contents",
        apis: &[
            "PrintWindow",
            "DwmGetDxSharedSurface",
            "IDXGIOutputDuplication",
        ],
    },
    Rule {
        category: "Clipboard access",
        description: "Reads or writes clipboard data",
        apis: &["OpenClipboard", "GetClipboardData", "SetClipboardData"],
    },
    Rule {
        category: "Memory protection changes",
        description: "Allocates executable memory or changes page protections",
        apis: &[
            "VirtualProtect",
            "VirtualAlloc",
            "NtProtectVirtualMemory",
            "NtAllocateVirtualMemory",
            "mprotect",
            "mmap",
        ],
    },
    Rule {
        category: "Process enumeration",
        description: "Lists running processes or loaded modules",
        apis: &[
            "CreateToolhelp32Snapshot",
            "Process32First",
            "Process32Next",
            "EnumProcesses",
            "Module32First",
            "EnumProcessModules",
        ],
    },
    Rule {
        category: "System reconnaissance",
        description: "Collects host, user, or operating system details",
        apis: &[
            "GetComputerName",
            "GetUserName",
            "GetVersionEx",
            "RtlGetVersion",
            "GetSystemInfo",
            "GetNativeSystemInfo",
            "uname",
            "gethostname",
        ],
    },
];

/// Classifies imports into capability categories.
///
/// # Arguments
///
/// * `imports` - Imported functions.
///
/// # Returns
///
/// One capability per rule with at least one matching import.
#[must_use]
pub fn classify(imports: &[Import]) -> Vec<Capability> {
    RULES.iter().filter_map(|rule| evaluate(rule, imports)).collect()
}

/// Evaluates one rule.
///
/// # Arguments
///
/// * `rule` - Capability rule.
/// * `imports` - Imported functions.
///
/// # Returns
///
/// The capability when any import matches.
fn evaluate(rule: &Rule, imports: &[Import]) -> Option<Capability> {
    let matching = imports.iter().filter(|i| matches_rule(rule, &i.name));
    let mut apis: Vec<String> = matching.map(|i| i.name.clone()).collect();
    apis.sort();
    apis.dedup();
    (!apis.is_empty()).then(|| Capability::new(rule, apis))
}

impl Capability {
    /// Creates a capability from a rule and its matching APIs.
    ///
    /// # Arguments
    ///
    /// * `rule` - Matching rule.
    /// * `apis` - Imported APIs that matched.
    ///
    /// # Returns
    ///
    /// The capability.
    #[must_use]
    pub fn new(rule: &Rule, apis: Vec<String>) -> Self {
        Self { category: rule.category, description: rule.description, apis }
    }
}

/// Reports whether an API name matches a rule.
///
/// # Arguments
///
/// * `rule` - Capability rule.
/// * `name` - Imported function name.
///
/// # Returns
///
/// `true` when the name starts with any of the rule's prefixes.
#[must_use]
pub fn matches_rule(rule: &Rule, name: &str) -> bool {
    let name = name.trim_start_matches('_');
    rule.apis.iter().any(|api| name.starts_with(api))
}

/// Finds the category of the first rule matching an API name.
///
/// # Arguments
///
/// * `name` - Imported function name.
///
/// # Returns
///
/// The category, when any rule matches.
#[must_use]
pub fn category_of(name: &str) -> Option<&'static str> {
    RULES.iter().find(|rule| matches_rule(rule, name)).map(|rule| rule.category)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Builds an import with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - Function name.
    ///
    /// # Returns
    ///
    /// An import from `KERNEL32.dll`.
    fn import(name: &str) -> Import {
        Import::new("KERNEL32.dll", name, 0)
    }

    /// Verifies that injection APIs are classified together and deduplicated.
    #[test]
    fn test_classify_groups_injection() {
        let (write, start) = ("WriteProcessMemory", "CreateRemoteThread");
        let found = classify(&[write, start, start].map(import));
        let want = vec!["CreateRemoteThread", "WriteProcessMemory"];
        assert_eq!(found[0].category, "Process injection");
        assert_eq!(found[0].apis, want);
    }

    /// Verifies that A/W suffixed names match their prefixes.
    #[test]
    fn test_classify_matches_suffixed_names() {
        let found = classify(&[import("CreateProcessW")]);
        assert_eq!(found[0].category, "Process execution");
    }

    /// Verifies that benign imports yield no capabilities.
    #[test]
    fn test_classify_ignores_benign() {
        assert!(classify(&[import("GetLastError")]).is_empty());
    }

    /// Verifies category lookup with leading underscores.
    #[test]
    fn test_category_of_strips_underscores() {
        assert_eq!(category_of("_ptrace"), Some("Anti-analysis"));
        assert_eq!(category_of("HeapFree"), None);
    }
}
