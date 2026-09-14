from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence


# TMX_DOCTOR_ALLOW_FORBIDDEN_NETWORK_TOKENS
FORBIDDEN_NETWORK_TOKENS = (
    "--internal-ip",
    "--internal-ips",
    "--internal_ip",
    "--internal_ips",
    "enable_external_ips=false",
    "enableexternalips=false",
    "enableExternalIps: false",
    "cloud nat",
    "routers nats",
    "compute routers",
)

# The TPU Research Cloud (TRC) project whose networking rules are enforced
# below: nanny-created v5litepod-256 only, external IPs required, no NAT.
# Set TMX_TRC_PROJECT to the project id of your TRC allocation; when it is
# unset no project is treated as TRC and only the generic checks apply.
TRC_PROJECT = os.environ.get("TMX_TRC_PROJECT", "")


def is_trc_project(project: str) -> bool:
    return bool(TRC_PROJECT) and project == TRC_PROJECT


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def find_forbidden_network_tokens(values: Iterable[str]) -> tuple[str, ...]:
    hits: list[str] = []
    for value in values:
        lowered = str(value).lower()
        for token in FORBIDDEN_NETWORK_TOKENS:
            if token.lower() in lowered:
                hits.append(token)
    return tuple(sorted(set(hits)))


def text_allows_forbidden_network_tokens(text: str) -> bool:
    return "TMX_DOCTOR_ALLOW_FORBIDDEN_NETWORK_TOKENS" in text


def assert_trc_external_ip_policy(args: Sequence[str] = ()) -> None:
    hits = find_forbidden_network_tokens(args)
    if hits:
        raise ValueError(
            "TRC networking policy violation: external IPs are required and "
            f"NAT/internal-IP paths are forbidden. Offending tokens: {', '.join(hits)}"
        )


def validate_network_policy(project: str, accelerator_type: str, external_ips_required: bool) -> SafetyResult:
    errors: list[str] = []
    warnings: list[str] = []
    if is_trc_project(project):
        if accelerator_type != "v5litepod-256":
            errors.append(f"{project} (TRC) runs must use v5litepod-256 only")
        if not external_ips_required:
            errors.append(f"{project} (TRC) runs must require external IPs")
    if not external_ips_required:
        warnings.append("external_ips_required=false is not allowed for TRC runs")
    return SafetyResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
