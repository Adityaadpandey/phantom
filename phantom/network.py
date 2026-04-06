from __future__ import annotations
import random
from dataclasses import dataclass


@dataclass
class Host:
    host_id: str
    ip: str
    hostname: str
    services: list[str]
    os: str
    subnet: str
    is_crown_jewel: bool = False
    is_compromised: bool = False
    is_isolated: bool = False
    is_patched: bool = False
    compromise_turn: int | None = None
    last_scanned_turn: int | None = None


class NetworkState:
    def __init__(self, hosts: dict[str, Host], edges: list[tuple[str, str]]):
        self.hosts = hosts
        self.edges = edges
        self._adj: dict[str, list[str]] = {h: [] for h in hosts}
        for a, b in edges:
            if a not in self._adj or b not in self._adj:
                raise ValueError(f"Edge ({a!r}, {b!r}) references unknown host(s)")
            self._adj[a].append(b)
            self._adj[b].append(a)

    # --- Mutations ---

    def compromise_host(self, host_id: str, turn: int) -> None:
        h = self.hosts[host_id]
        if not h.is_isolated:
            h.is_compromised = True
            h.compromise_turn = turn

    def isolate_host(self, host_id: str) -> None:
        self.hosts[host_id].is_isolated = True

    def patch_host(self, host_id: str) -> None:
        h = self.hosts[host_id]
        h.is_compromised = False
        h.is_patched = True

    def restore_host(self, host_id: str) -> None:
        h = self.hosts[host_id]
        h.is_isolated = False
        h.is_patched = False

    def scan_host(self, host_id: str, turn: int) -> None:
        self.hosts[host_id].last_scanned_turn = turn

    # --- Queries ---

    def get_neighbors(self, host_id: str) -> list[str]:
        return self._adj.get(host_id, [])

    def compromised_hosts(self) -> list[str]:
        return [
            hid for hid, h in self.hosts.items()
            if h.is_compromised and not h.is_isolated
        ]

    def any_compromised(self) -> bool:
        return any(h.is_compromised for h in self.hosts.values())

    def all_contained(self) -> bool:
        """All compromised hosts are isolated (or patched)."""
        return all(
            h.is_isolated or h.is_patched
            for h in self.hosts.values()
            if h.is_compromised
        )

    def exfiltration_complete(self) -> bool:
        """Any crown jewel is compromised and not isolated."""
        return any(
            h.is_crown_jewel and h.is_compromised and not h.is_isolated
            for h in self.hosts.values()
        )

    # --- Factory ---

    @classmethod
    def from_preset(cls, preset: str, rng: random.Random) -> "NetworkState":
        presets = {
            "small_corp": _small_corp,
            "mid_corp": _mid_corp,
            "enterprise": _enterprise,
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset!r}. Choose from {list(presets)}")
        return presets[preset](rng)


# ── Preset builders ──────────────────────────────────────────────────────────

def _make_host(host_id: str, ip: str, hostname: str, services: list[str],
               os: str, subnet: str, crown_jewel: bool = False) -> Host:
    return Host(
        host_id=host_id, ip=ip, hostname=hostname,
        services=services, os=os, subnet=subnet, is_crown_jewel=crown_jewel,
    )


def _small_corp(rng: random.Random) -> NetworkState:
    """8-host small corporate network."""
    hosts = {
        "fw-01": _make_host("fw-01", "10.0.0.1", "firewall-01",
                             ["firewall"], "Linux", "dmz"),
        "web-01": _make_host("web-01", "10.0.1.10", "web-server-01",
                              ["http", "https"], "Linux", "dmz"),
        "app-01": _make_host("app-01", "10.1.0.10", "app-server-01",
                              ["http", "https"], "Linux", "internal"),
        "db-01": _make_host("db-01", "10.1.0.20", "db-server-01",
                             ["postgresql"], "Linux", "internal", crown_jewel=True),
        "dc-01": _make_host("dc-01", "10.1.0.5", "domain-controller-01",
                             ["ldap", "kerberos", "smb"], "Windows Server", "internal"),
        "ws-01": _make_host("ws-01", "10.2.0.10", "workstation-01",
                             ["rdp", "smb"], "Windows 11", "corp"),
        "ws-02": _make_host("ws-02", "10.2.0.11", "workstation-02",
                             ["rdp", "smb"], "Windows 11", "corp"),
        "mgmt-01": _make_host("mgmt-01", "10.3.0.5", "mgmt-server-01",
                               ["ssh", "https"], "Linux", "management"),
    }
    edges = [
        ("fw-01", "web-01"), ("fw-01", "app-01"),
        ("web-01", "app-01"), ("app-01", "db-01"),
        ("app-01", "dc-01"), ("dc-01", "ws-01"),
        ("dc-01", "ws-02"), ("mgmt-01", "fw-01"),
        ("mgmt-01", "dc-01"),
    ]
    return NetworkState(hosts, edges)


def _mid_corp(rng: random.Random) -> NetworkState:
    """~25 host mid-sized corporate network."""
    hosts: dict[str, Host] = {}

    # DMZ
    hosts["fw-01"] = _make_host("fw-01", "10.0.0.1", "firewall-01",
                                 ["firewall"], "Linux", "dmz")
    hosts["lb-01"] = _make_host("lb-01", "10.0.0.2", "load-balancer-01",
                                 ["http", "https"], "Linux", "dmz")
    for i in range(1, 4):
        hid = f"web-0{i}"
        hosts[hid] = _make_host(hid, f"10.0.1.{10+i}", f"web-server-0{i}",
                                  ["http", "https"], "Linux", "dmz")

    # Internal
    hosts["dc-01"] = _make_host("dc-01", "10.1.0.5", "domain-controller-01",
                                 ["ldap", "kerberos", "smb"], "Windows Server", "internal")
    hosts["dc-02"] = _make_host("dc-02", "10.1.0.6", "domain-controller-02",
                                 ["ldap", "kerberos", "smb"], "Windows Server", "internal")
    for i in range(1, 5):
        hid = f"app-0{i}"
        hosts[hid] = _make_host(hid, f"10.1.1.{10+i}", f"app-server-0{i}",
                                  ["http", "https"], "Linux", "internal")
    hosts["db-01"] = _make_host("db-01", "10.1.2.10", "db-primary-01",
                                 ["postgresql"], "Linux", "internal", crown_jewel=True)
    hosts["db-02"] = _make_host("db-02", "10.1.2.11", "db-replica-01",
                                 ["postgresql"], "Linux", "internal", crown_jewel=True)
    hosts["file-01"] = _make_host("file-01", "10.1.2.20", "file-server-01",
                                   ["smb", "nfs"], "Windows Server", "internal", crown_jewel=True)

    # Corp workstations
    for i in range(1, 8):
        hid = f"ws-{i:02d}"
        hosts[hid] = _make_host(hid, f"10.2.0.{10+i}", f"workstation-{i:02d}",
                                  ["rdp", "smb"], "Windows 11", "corp")

    # Management
    hosts["mgmt-01"] = _make_host("mgmt-01", "10.3.0.5", "mgmt-server-01",
                                   ["ssh", "https"], "Linux", "management")
    hosts["siem-01"] = _make_host("siem-01", "10.3.0.10", "siem-server-01",
                                   ["syslog", "https"], "Linux", "management")

    # Edges (simplified connectivity)
    edges = [
        ("fw-01", "lb-01"),
        ("lb-01", "web-01"), ("lb-01", "web-02"), ("lb-01", "web-03"),
        ("web-01", "app-01"), ("web-02", "app-02"), ("web-03", "app-03"),
        ("app-01", "db-01"), ("app-02", "db-01"), ("app-03", "db-02"),
        ("app-01", "file-01"), ("app-04", "file-01"),
        ("dc-01", "dc-02"), ("dc-01", "ws-01"), ("dc-01", "ws-02"),
        ("dc-02", "ws-03"), ("dc-02", "ws-04"), ("dc-01", "ws-05"),
        ("dc-01", "ws-06"), ("dc-02", "ws-07"),
        ("mgmt-01", "fw-01"), ("mgmt-01", "dc-01"), ("siem-01", "mgmt-01"),
    ]
    return NetworkState(hosts, edges)


def _enterprise(rng: random.Random) -> NetworkState:
    """~60 host enterprise network."""
    hosts: dict[str, Host] = {}

    # DMZ
    hosts["fw-ext-01"] = _make_host("fw-ext-01", "203.0.113.1", "ext-firewall-01",
                                     ["firewall"], "Linux", "edge")
    hosts["fw-int-01"] = _make_host("fw-int-01", "10.0.0.1", "int-firewall-01",
                                     ["firewall"], "Linux", "dmz")
    hosts["lb-01"] = _make_host("lb-01", "10.0.0.10", "load-balancer-01",
                                 ["http", "https"], "Linux", "dmz")
    hosts["lb-02"] = _make_host("lb-02", "10.0.0.11", "load-balancer-02",
                                 ["http", "https"], "Linux", "dmz")
    for i in range(1, 6):
        hid = f"web-{i:02d}"
        hosts[hid] = _make_host(hid, f"10.0.1.{10+i}", f"web-server-{i:02d}",
                                  ["http", "https"], "Linux", "dmz")

    # Internal servers
    hosts["dc-01"] = _make_host("dc-01", "10.1.0.5", "dc-primary-01",
                                 ["ldap", "kerberos", "smb", "dns"], "Windows Server", "internal")
    hosts["dc-02"] = _make_host("dc-02", "10.1.0.6", "dc-secondary-01",
                                 ["ldap", "kerberos", "smb", "dns"], "Windows Server", "internal")
    hosts["exchange-01"] = _make_host("exchange-01", "10.1.0.20", "exchange-server-01",
                                       ["smtp", "https"], "Windows Server", "internal")
    for i in range(1, 11):
        hid = f"app-{i:02d}"
        hosts[hid] = _make_host(hid, f"10.1.1.{10+i}", f"app-server-{i:02d}",
                                  ["http", "https"], "Linux", "internal")

    # Databases
    hosts["db-01"] = _make_host("db-01", "10.1.2.10", "db-primary-01",
                                 ["postgresql", "ssl"], "Linux", "data", crown_jewel=True)
    hosts["db-02"] = _make_host("db-02", "10.1.2.11", "db-replica-01",
                                 ["postgresql", "ssl"], "Linux", "data", crown_jewel=True)
    hosts["db-03"] = _make_host("db-03", "10.1.2.12", "db-analytics-01",
                                 ["postgresql", "ssl"], "Linux", "data", crown_jewel=True)
    hosts["vault-01"] = _make_host("vault-01", "10.1.2.20", "secrets-vault-01",
                                    ["https", "api"], "Linux", "data", crown_jewel=True)
    hosts["backup-01"] = _make_host("backup-01", "10.1.2.30", "backup-server-01",
                                     ["nfs", "smb"], "Linux", "data", crown_jewel=True)

    # Corp floor (20 workstations)
    for i in range(1, 21):
        hid = f"ws-{i:02d}"
        hosts[hid] = _make_host(hid, f"10.2.{i//10}.{10+i%10}", f"workstation-{i:02d}",
                                  ["rdp", "smb"], "Windows 11", "corp")

    # Management
    hosts["mgmt-01"] = _make_host("mgmt-01", "10.3.0.5", "jump-server-01",
                                   ["ssh"], "Linux", "management")
    hosts["siem-01"] = _make_host("siem-01", "10.3.0.10", "siem-server-01",
                                   ["syslog", "https"], "Linux", "management")
    hosts["nac-01"] = _make_host("nac-01", "10.3.0.15", "nac-server-01",
                                  ["radius"], "Linux", "management")

    edges: list[tuple[str, str]] = [
        ("fw-ext-01", "fw-int-01"),
        ("fw-int-01", "lb-01"), ("fw-int-01", "lb-02"),
        ("lb-01", "web-01"), ("lb-01", "web-02"), ("lb-01", "web-03"),
        ("lb-02", "web-04"), ("lb-02", "web-05"),
        ("web-01", "app-01"), ("web-02", "app-02"), ("web-03", "app-03"),
        ("web-04", "app-04"), ("web-05", "app-05"),
        ("app-01", "db-01"), ("app-02", "db-01"), ("app-03", "db-02"),
        ("app-04", "db-03"), ("app-05", "db-03"), ("app-06", "vault-01"),
        ("app-07", "db-01"), ("app-08", "db-02"), ("app-09", "vault-01"), ("app-10", "backup-01"),
        ("db-01", "backup-01"), ("db-02", "backup-01"),
        ("dc-01", "dc-02"), ("dc-01", "exchange-01"),
        ("dc-01", "ws-01"), ("dc-01", "ws-02"), ("dc-01", "ws-03"),
        ("dc-01", "ws-04"), ("dc-01", "ws-05"), ("dc-01", "ws-06"),
        ("dc-01", "ws-07"), ("dc-01", "ws-08"), ("dc-01", "ws-09"),
        ("dc-01", "ws-10"),
        ("dc-02", "ws-11"), ("dc-02", "ws-12"), ("dc-02", "ws-13"),
        ("dc-02", "ws-14"), ("dc-02", "ws-15"), ("dc-02", "ws-16"),
        ("dc-02", "ws-17"), ("dc-02", "ws-18"), ("dc-02", "ws-19"),
        ("dc-02", "ws-20"),
        ("mgmt-01", "fw-int-01"), ("mgmt-01", "dc-01"), ("mgmt-01", "siem-01"),
        ("nac-01", "dc-01"), ("siem-01", "nac-01"),
    ]
    return NetworkState(hosts, edges)
