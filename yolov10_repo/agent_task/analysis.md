# 任務三分析

## 1. 完成狀態

本 Task-3 提交包使用學生自建的離線 fixture，在固定任務、來源包、validator、rubric 與 policy 下執行了兩次，只改變 `max_steps`。

- `max_steps=3`：在讀取第三個來源前耗盡步數，沒有產生 brief，validator 失敗。
- `max_steps=6`：完成搜尋、三個來源讀取、brief 寫入與 final-state validation，validator 通過。

根據作業文字，「核心 fixture 免 API、模型、GPU、網路與大型套件」是環境約束，並未要求教師預先提供現成 fixture。本版本因此自行構建 source pack 與獨立 validator，並確保整條流程符合上述離線約束。

## 2. Tool calls

成功 run（`max_steps=6`）：

1. 以研究問題呼叫 `search_sources`；
2. `read_source(S3)` 讀取 YOLOv10；
3. `read_source(S5)` 讀取 YOLO-World；
4. `read_source(S2)` 讀取 YOLOv9；
5. `write_final_brief` 寫入剛好三個 citations；
6. `validate_final_state` 通過。

完整 trajectory 與 observations 位於 `runs/max_steps_6/`。根目錄的 `trajectory.jsonl` 與 `tool-results.jsonl` 是成功 run 的副本。

## 3. 小型受控比較

這次比較隔離的是 tool budget：

| max_steps | Validator | 失敗位置 |
|---:|---|---|
| 3 | 失敗 | 第三個來源讀取前、brief 寫入前 |
| 6 | 通過 | 無 |

結論很直接：在這套 policy 下，最低成功預算由任務 contract 決定，而不是由模型能力決定。Search + 3 reads + write + validate，共六個 calls。

按本地 rubric 計算，`max_steps=3` 得到 45/100（完成與 validator 為 0，工具證據不完整），`max_steps=6` 得到 100/100。這是本提交包內部 rubric 的分數，最終成績仍以課程評分為準。

## 4. 最容易出錯的步驟與失敗分析

最容易發生的操作性錯誤，是讀了一些來源卻沒有走到可驗證的 final state。這種 run 看起來有活動，但仍缺少：

- `final-brief.md`；
- 剛好三個有效 citations；
- 與來源 path、SHA-256 一致的 `source-usage.json`；
- 完整 tool trace；
- 通過的 validator result。

`max_steps=3` 刻意展示這個失敗模式。即使 agent 更強，如果它過度搜尋、遲遲不進入 synthesis 與 validation，也可能在同一個位置失敗。

## 5. Provenance、可重跑性與安全邊界

- `provenance.json` 記錄來源、agent、validator 的雜湊與兩個 run 的結果。
- `task-contract.json` 固定任務與比較條件。
- `tool-schema.json` 定義四個工具的輸入契約。
- `rubric.json` 記錄四個評分維度。
- 全程不使用網路、API、模型或 GPU。
- fixture 明確標記為 `student_defined_offline_fixture`，符合免 API、模型、GPU、網路與大型套件的要求。
- brief 是 deterministic policy 的輸出，不是真實模型生成。

## 6. 限制

- 來源包由學生自建，但路徑、required source IDs 與 citation 規則均由 task contract 固定。
- validator 只檢查 final state 與 evidence integrity，不判斷語義真值。
- policy 很窄且 deterministic，不測試模型推理或自適應搜尋。
- citation 集合由本地 fixture 的 required source IDs 固定。

## 7. 下一輪測試

若課程另行提供指定來源包，替換後應重跑相同的 `max_steps=3` 與 `max_steps=6` 比較；若沒有另行指定，本版本已包含完整 fixture。之後可再加入 prompt-injection 來源，檢查 agent 是否會把來源內容當成資料，而不是執行其中的指令。
