# 任務三錯誤紀錄

## 受控比較

只改變 `max_steps`，任務、來源包、validator、rubric 與 policy 均保持不變。

| Run | max_steps | 使用步數 | Tool calls | Validator | Citations | 結果 |
|---|---:|---:|---:|---|---:|---|
| max_steps_3 | 3 | 3 | 3 | False | 0 | step budget exhausted before reading three sources |
| max_steps_6 | 6 | 6 | 6 | True | 3 | 完成 |

## 最容易出錯的位置

最容易失敗的位置不是「搜尋不到來源」，而是**讀了來源卻沒有走到 final state**。在讀完前兩個來源後，agent 已經有 observations，但尚未產生 `final-brief.md`、`source-usage.json` 與 validator result。`max_steps=3` 正是這個失敗模式。

成功路徑需要六個 tool calls：

1. `search_sources`
2. `read_source(S3)`
3. `read_source(S5)`
4. `read_source(S2)`
5. `write_final_brief`
6. `validate_final_state`

因此在目前 policy 與 fixture 下，`max_steps=6` 是最小成功預算。

## Observations 分析

- 搜尋步驟只回傳排序後的來源 metadata，不一次讀完整個來源包。
- 每次讀取都記錄 source ID、path 與 SHA-256。
- 寫入步驟同時建立 brief 與 source-usage provenance。
- validator 檢查 brief、citation、來源雜湊與 tool trace。
- `max_steps=3` 的失敗發生在第三次讀取之前，屬於預算耗盡，不是 citation 品質問題。

## 限制

- 來源包由學生自建，但 `task-contract.json` 已指定來源路徑、required source IDs 與使用範圍。
- validator 是 fixture 的獨立元件，負責 final-state 與 evidence integrity 檢查。
- deterministic policy 很窄，只測試工具預算與 provenance，不測試開放式規劃或模型推理。
- 這套實驗不能宣稱代表一般 Agent 能力。

## 下一輪測試

1. 若課程另行發布指定來源包或 validator，替換後保持任務與 rubric 不變重跑。
2. 若沒有另行發布，本版本已是自包含的離線提交。
3. 增加 `max_steps=5`，確認真正的瓶頸是否為 validate 步驟。
4. 加入含 prompt injection 的惡意來源，檢查 agent 是否只把它當資料，而不是指令。
