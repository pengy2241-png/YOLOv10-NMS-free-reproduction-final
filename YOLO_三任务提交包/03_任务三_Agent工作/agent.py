"""Deterministic offline agent for the local Task-3 fixture.

No API, model, GPU or network is used. The policy is intentionally simple and
auditable: search -> read three related sources -> write brief -> validate.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validator import validate_run

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "fixture" / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def search_sources(query: str, limit: int = 5) -> dict[str, Any]:
    manifest = parse_manifest()
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9-]+", query)]
    scored = []
    for src in manifest["sources"]:
        tags = {str(t).lower() for t in src["tags"]}
        title = src["title"].lower()
        summary = src["summary"].lower()
        score = 0
        for token in tokens:
            if token in tags:
                score += 4
            elif token in title:
                score += 2
            elif token in summary:
                score += 1
        scored.append((score, src["id"], src))
    scored.sort(key=lambda x: (-x[0], x[1]))
    hits = []
    for score, sid, src in scored[:limit]:
        hits.append(
            {
                "source_id": sid,
                "title": src["title"],
                "year": src["year"],
                "venue": src["venue"],
                "tags": src["tags"],
                "score": score,
                "summary": src["summary"],
            }
        )
    return {"query": query, "result_count": len(hits), "results": hits}


def choose_three_related(results: list[dict[str, Any]]) -> list[str]:
    """Pick one source for each required topic, using search output only."""
    target_ids: list[str] = []
    for topic_tag in ("nms-free", "open-vocabulary", "gradient"):
        match = next((r for r in results if topic_tag in r["tags"]), None)
        if match is None:
            continue
        sid = match["source_id"]
        if sid not in target_ids:
            target_ids.append(sid)
    for row in results:
        if row["source_id"] not in target_ids:
            target_ids.append(row["source_id"])
        if len(target_ids) == 3:
            break
    return target_ids[:3]


def read_source(source_id: str) -> dict[str, Any]:
    manifest = parse_manifest()
    src = next((s for s in manifest["sources"] if s["id"] == source_id), None)
    if src is None:
        raise ValueError(f"unknown source_id: {source_id}")
    path = ROOT / src["path"]
    return {
        "source_id": source_id,
        "title": src["title"],
        "year": src["year"],
        "venue": src["venue"],
        "path": src["path"],
        "sha256": sha256_file(path),
        "content": path.read_text(),
    }


def compose_brief(read_sources: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Compose a concise brief from the three read source records."""
    required = ["S3", "S5", "S2"]
    if not all(sid in read_sources for sid in required):
        raise ValueError(f"missing required sources: {required}")
    s2, s3, s5 = read_sources["S2"], read_sources["S3"], read_sources["S5"]
    brief = f"""# YOLO 的下一步：从闭集检测到 NMS-free 与开放词汇

**研究问题：** YOLO 的演进是否只是一连串版本升级，还是同时在改变“如何部署”和“能检测什么”这两个问题定义？

## 一页摘要

YOLO 早期的方法把检测统一为单网络回归，显著提升实时性，但类别集合在训练时固定，推理仍依赖后处理 [S3]。近年的三条代表性路线说明，YOLO 的边界正在发生变化。

**部署边界：YOLOv10。** YOLOv10 将 NMS 视为端到端部署的瓶颈，提出一致双重标签分配：one-to-many 分支提供密集监督，one-to-one 分支以一致的匹配代价训练并在推理时直接输出，不再执行 NMS [S3]。论文报告 YOLOv10-S 在相近 AP 下比 RT-DETR-R18 约快 1.8 倍，参数与 FLOPs 约少 2.8 倍 [S3]。这说明实时检测的竞争已从单纯追求 AP/FPS，转向后处理、延迟与整体部署成本。

**任务边界：YOLO-World。** YOLO-World 针对闭集检测的类别限制，引入 RepVL-PAN 与区域-文本对比学习，把视觉特征和语言嵌入连接起来，实现开放词汇检测 [S5]。论文在 LVIS 上报告 35.4 AP、V100 上 52 FPS 的零样本结果，并支持下游微调与开放词汇实例分割 [S5]。这意味着 YOLO 的能力不再只由训练时的固定类别表决定，而部分转移到文本编码器、词汇设计和图文对齐。

**信息流边界：YOLOv9。** YOLOv9 关注深层网络中的信息瓶颈，提出 Programmable Gradient Information（PGI）与 GELAN，以梯度路径规划改善信息保留和参数利用 [S2]。其主要实验集中在 COCO 目标检测，说明“架构创新”仍在继续，但跨任务通用性仍需独立验证 [S2]。

## 综合判断

三条路线可以合并成一个判断：YOLO 正在从“单一闭集检测器”演化为一个同时追求**更低的端到端延迟、更开放的类别边界、更可控的梯度与信息流**的检测框架。真正的转折点不是版本号，而是问题定义的变化：YOLOv10 改变后处理，YOLO-World 改变类别条件，YOLOv9 改变训练时的信息通路 [S2][S3][S5]。

## 局限与下一步

现有证据仍主要来自 COCO 或特定视觉-语言基准，不能直接外推到边缘设备和专业领域。YOLOv10 的 NMS-free 需要在密集场景与不同硬件上做压力测试；YOLO-World 需要评估词表、提示和领域数据的敏感性；YOLOv9 需要跨任务验证 [S2][S3][S5]。下一步应固定数据、输入分辨率、硬件与推论框架，比较 NMS-free、开放词汇和梯度改进三者的真实部署收益。

## 参考来源

- [S2] {s2['title']} ({s2['year']}, {s2['venue']})
- [S3] {s3['title']} ({s3['year']}, {s3['venue']})
- [S5] {s5['title']} ({s5['year']}, {s5['venue']})
"""
    citations = ["S3", "S5", "S2"]
    return brief.strip() + "\n", citations


@dataclass
class AgentRuntime:
    run_id: str
    max_steps: int
    run_dir: Path
    started_at: str = field(default_factory=now_iso)
    step: int = 0
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    search_output: dict[str, Any] | None = None
    target_ids: list[str] = field(default_factory=list)
    read_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    brief_written: bool = False
    validated: bool = False
    completed: bool = False
    failure_reason: str = ""

    def flush_logs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "trajectory.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in self.trajectory) + ("\n" if self.trajectory else "")
        )
        (self.run_dir / "tool-results.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in self.tool_results) + ("\n" if self.tool_results else "")
        )

    def _record(self, tool: str, arguments: dict[str, Any], result: Any, ok: bool, latency_ms: float) -> None:
        call_id = f"{self.run_id}-call-{self.step:02d}"
        record = {
            "call_id": call_id,
            "run_id": self.run_id,
            "step": self.step,
            "tool": tool,
            "arguments": arguments,
            "ok": bool(ok),
            "result": result,
            "latency_ms": round(latency_ms, 3),
            "timestamp": now_iso(),
        }
        self.tool_results.append(record)
        self.trajectory.append(
            {
                "trajectory_id": self.run_id,
                "step": self.step,
                "policy": "deterministic_fixture_agent_v1",
                "tool": tool,
                "arguments": arguments,
                "observation": result,
                "state_after": {
                    "read_source_ids": sorted(self.read_sources),
                    "brief_written": self.brief_written,
                    "validated": self.validated,
                    "completed": self.completed,
                },
                "ok": bool(ok),
                "timestamp": record["timestamp"],
            }
        )

    def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.step >= self.max_steps:
            raise RuntimeError("step budget exhausted")
        self.step += 1
        start = time.perf_counter()
        try:
            if tool == "search_sources":
                result = search_sources(**arguments)
                self.search_output = result
                self.target_ids = choose_three_related(result["results"])
            elif tool == "read_source":
                result = read_source(arguments["source_id"])
                self.read_sources[arguments["source_id"]] = result
            elif tool == "write_final_brief":
                content = arguments["content"]
                citations = list(arguments["citations"])
                brief_path = self.run_dir / "final-brief.md"
                brief_path.write_text(content)
                usage = {}
                for sid in citations:
                    src = read_source(sid)
                    usage[sid] = {"path": src["path"], "sha256": src["sha256"], "title": src["title"]}
                (self.run_dir / "source-usage.json").write_text(
                    json.dumps({"citations": usage, "created_at": now_iso()}, ensure_ascii=False, indent=2)
                )
                self.brief_written = True
                result = {"path": str(brief_path), "chars": len(content), "citations": citations}
            elif tool == "validate_final_state":
                # The validator must see the tool evidence accumulated so far.
                self.flush_logs()
                result = validate_run(self.run_dir)
                (self.run_dir / "validator-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                self.validated = bool(result.get("passed"))
                self.completed = self.validated
            else:
                raise ValueError(f"unknown tool: {tool}")
            latency = (time.perf_counter() - start) * 1000
            self._record(tool, arguments, result, True, latency)
            return result
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            result = {"error": type(exc).__name__, "message": str(exc)}
            self._record(tool, arguments, result, False, latency)
            raise

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.max_steps <= 0:
                raise RuntimeError("max_steps must be positive")
            self.call_tool("search_sources", {"query": "YOLO NMS-free open-vocabulary closed-set gradient", "limit": 5})
            while len(self.read_sources) < 3 and self.step < self.max_steps:
                next_id = next((sid for sid in self.target_ids if sid not in self.read_sources), None)
                if next_id is None:
                    raise RuntimeError("no target source left")
                self.call_tool("read_source", {"source_id": next_id})
            if len(self.read_sources) < 3:
                raise RuntimeError("step budget exhausted before reading three sources")
            if not self.brief_written:
                content, citations = compose_brief(self.read_sources)
                self.call_tool("write_final_brief", {"content": content, "citations": citations})
            if not self.validated:
                self.call_tool("validate_final_state", {})
        except Exception as exc:
            self.failure_reason = str(exc)
        finally:
            duration_ms = (datetime.now(timezone.utc) - datetime.fromisoformat(self.started_at)).total_seconds() * 1000
            meta = {
                "run_id": self.run_id,
                "max_steps": self.max_steps,
                "completed": self.completed,
                "validated": self.validated,
                "steps_used": self.step,
                "tool_calls": len(self.tool_results),
                "read_source_ids": sorted(self.read_sources),
                "target_source_ids": self.target_ids,
                "brief_written": self.brief_written,
                "failure_reason": self.failure_reason,
                "duration_ms": round(duration_ms, 3),
                "started_at": self.started_at,
                "ended_at": now_iso(),
            }
            (self.run_dir / "run-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            self.flush_logs()
        return meta


def run_agent(run_id: str, max_steps: int, run_dir: Path) -> dict[str, Any]:
    return AgentRuntime(run_id=run_id, max_steps=max_steps, run_dir=run_dir).run()
