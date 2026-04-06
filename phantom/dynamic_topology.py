from __future__ import annotations
import json
import random
from phantom.gpt_client import GPTClient
from phantom.network import NetworkState, Host

_DOMAIN_SYSTEM_PROMPT = """You are a network architect. Generate realistic corporate network topologies as JSON.
The JSON must have exactly this structure:
{
  "hosts": [
    {
      "host_id": "fw-01",
      "ip": "10.0.0.1",
      "hostname": "firewall-01",
      "services": ["firewall"],
      "os": "Linux",
      "subnet": "dmz",
      "is_crown_jewel": false
    }
  ],
  "edges": [["host_id_a", "host_id_b"]]
}
Subnets must be one of: dmz, internal, corp, management, data, edge.
At least one host must have is_crown_jewel=true.
Return ONLY valid JSON. No markdown, no explanation."""

_SIZE_PARAMS = {
    "small": (8, 15),
    "medium": (20, 40),
    "large": (50, 100),
}

_DOMAIN_PROMPTS = {
    "financial_services": "financial services firm with trading floor, compliance servers, and customer database",
    "healthcare": "hospital network with EMR systems, medical devices, and research servers",
    "manufacturing": "manufacturing plant with OT/SCADA systems, corporate IT, and engineering workstations",
    "tech_startup": "tech startup with development environments, CI/CD servers, and SaaS platform infrastructure",
}


class DynamicTopologyGenerator:
    def __init__(self, client: GPTClient):
        self._client = client

    async def generate(self, template: str, size: str, seed: int) -> NetworkState:
        min_hosts, max_hosts = _SIZE_PARAMS.get(size, _SIZE_PARAMS["small"])
        domain_desc = _DOMAIN_PROMPTS.get(template, template)
        prompt = (
            f"Generate a realistic {domain_desc} network topology "
            f"with {min_hosts}-{max_hosts} hosts. Use seed {seed} for consistency."
        )

        raw = await self._client.generate(
            prompt=prompt,
            system=_DOMAIN_SYSTEM_PROMPT,
            fallback_fn=lambda: "",
        )

        net = self._parse_topology(raw)
        if net is None:
            # Fall back to static preset based on size
            preset = {"small": "small_corp", "medium": "mid_corp", "large": "enterprise"}.get(size, "small_corp")
            net = NetworkState.from_preset(preset, random.Random(seed))
        return net

    def _parse_topology(self, raw: str) -> NetworkState | None:
        if not raw.strip():
            return None
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        try:
            hosts: dict[str, Host] = {}
            for h in data["hosts"]:
                # Validate required fields
                for field in ("host_id", "ip", "hostname", "services", "os", "subnet"):
                    if field not in h:
                        return None
                hosts[h["host_id"]] = Host(
                    host_id=h["host_id"],
                    ip=h["ip"],
                    hostname=h["hostname"],
                    services=h["services"],
                    os=h["os"],
                    subnet=h["subnet"],
                    is_crown_jewel=h.get("is_crown_jewel", False),
                )
            edges: list[tuple[str, str]] = [
                (e[0], e[1]) for e in data["edges"]
                if e[0] in hosts and e[1] in hosts
            ]
            if not hosts:
                return None
            return NetworkState(hosts, edges)
        except (KeyError, TypeError, IndexError):
            return None
