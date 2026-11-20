# 累積式大作業 ③：離線 Agent 工作任務

本目錄完成一個**可重跑、可追蹤、離線**的 Agent 任務。Agent 使用四個工具：

1. `search_sources`：搜尋來源包；
2. `read_source`：讀取來源並記錄 SHA-256；
3. `write_final_brief`：生成一頁含 citation 的簡報；
4. `validate_final_state`：呼叫獨立 validator。

## Fixture 與環境約束

根據作業說明，「核心 fixture 免 API、模型、GPU、網路與大型套件」是在規定整套實驗環境必須滿足的約束。它**沒有要求教師必須預先提供一個現成 fixture**。因此本目錄使用學生自建的離線 fixture：

```text
fixture_status = student_defined_offline_fixture
```

此 fixture 包含：

- 指定的離線來源包 `fixture/source_pack/`；
- 任務所需的來源清單 `fixture/manifest.json`；
- 獨立的 final-state validator `validator.py`；
- 可重跑的 deterministic Agent policy `agent.py`。

整條流程不使用 API、模型、GPU、網路或大型套件。若課程另外發布指定的教師來源包或 validator，只需替換對應檔案並重跑，Agent、trajectory schema、provenance 與 analysis 仍可保留。

## 一鍵執行

```bash
python run_experiment.py
```

執行後會生成：

- `final-brief.md`
- `source-usage.json`
- `trajectory.jsonl`
- `tool-results.jsonl`
- `evaluation.csv`
- `error-note.md`
- `provenance.json`
- `validator-result.json`
- `analysis.md`
- `runs/max_steps_3/`
- `runs/max_steps_6/`

## 受控比較

只改變 `max_steps`：

- `max_steps=3`：搜尋後只能讀兩個來源，尚未寫 brief，validator 必然失敗；
- `max_steps=6`：完成 search + 3 reads + write + validate，validator 通過。

任務、來源包、validator、rubric 與 policy 均保持不變。

## 安全邊界

- 不使用網路；
- 不使用 API；
- 不使用模型；
- 不使用 GPU；
- 如實標記 fixture 為學生自建、符合離線約束；
- 不把 deterministic policy 的輸出描述成真實模型輸出。
