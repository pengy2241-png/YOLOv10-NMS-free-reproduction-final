"""Offline final-state validator for the local Task-3 fixture.

This validator is deliberately independent from the agent policy. It checks
the final brief, citation provenance, and the presence of a complete tool trace.
It is part of the student-defined offline fixture and implements the final-state checks locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "fixture" / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def validate_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    manifest = json.loads(MANIFEST_PATH.read_text())
    valid_ids = {s["id"] for s in manifest["sources"]}
    required_ids = set(manifest["required_source_ids"])
    brief_path = run_dir / "final-brief.md"
    usage_path = run_dir / "source-usage.json"

    brief = brief_path.read_text() if brief_path.exists() else ""
    add("final_brief_exists", brief_path.exists(), str(brief_path))
    add("brief_length", 300 <= len(brief) <= 5000, f"{len(brief)} characters (expected 300-5000)")

    citations = sorted(set(re.findall(r"\[(S\d+)\]", brief)), key=lambda x: int(x[1:]))
    add("citation_count", len(citations) == 3, f"found {citations}")
    add("citation_ids_valid", bool(citations) and set(citations).issubset(valid_ids), f"valid_ids={sorted(valid_ids)}")
    add("required_topics_covered", required_ids.issubset(set(citations)), f"required={sorted(required_ids)}, found={citations}")
    add("no_placeholder_text", not re.search(r"\b(TODO|TBD|PLACEHOLDER)\b", brief, re.I), "checked TODO/TBD/PLACEHOLDER")

    usage_ok = False
    usage_detail = "missing or malformed source-usage.json"
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text())
            cited_usage = usage.get("citations", {})
            usage_ok = set(citations) == set(cited_usage)
            for sid in citations:
                item = cited_usage.get(sid, {})
                src = next((s for s in manifest["sources"] if s["id"] == sid), None)
                if src is None:
                    usage_ok = False
                    break
                actual_hash = sha256_file(ROOT / src["path"])
                if item.get("sha256") != actual_hash or item.get("path") != src["path"]:
                    usage_ok = False
                    break
            usage_detail = f"citations={sorted(cited_usage)}, path/hash checked"
        except Exception as exc:
            usage_detail = f"source-usage parse error: {exc}"
    add("source_usage_provenance", usage_ok, usage_detail)

    trajectory = load_jsonl(run_dir / "trajectory.jsonl")
    tool_results = load_jsonl(run_dir / "tool-results.jsonl")
    tools = [r.get("tool") for r in tool_results]
    add("trajectory_present", len(trajectory) >= 5, f"{len(trajectory)} steps")
    add("tool_results_present", len(tool_results) >= 5, f"{len(tool_results)} tool results")
    add("required_tool_sequence", tools[:1] == ["search_sources"] and tools.count("read_source") >= 3 and "write_final_brief" in tools, f"tools={tools}")

    passed = all(c["pass"] for c in checks)
    return {
        "validator": "local_final_state_validator_v1",
        "run_dir": str(run_dir),
        "passed": passed,
        "checks": checks,
        "citation_ids": citations,
        "brief_chars": len(brief),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = validate_run(Path(args.run_dir))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
