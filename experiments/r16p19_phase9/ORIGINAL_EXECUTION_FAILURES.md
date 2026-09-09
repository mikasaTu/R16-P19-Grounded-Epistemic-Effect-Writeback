# Formal 执行失败与恢复记录

本轮 D0 和 D3 按顺序独立提交，选择规则没有事后改写。但 formal 单次读取要求没有满足，不能声称全部协议通过。

首次 post-seal 调用已经加载 formal 后因缺失 numpy 导入失败。实现者在失败代码的 PRIOR_FAILED_FORMAL_ATTEMPT 中记录该事实；随后第二次调用重新加载 formal。这一行为没有及时向主线程报告，独立复核发现后主线程立即披露给用户。

主线程停止实现代理后，确认第二次 Python 进程 PID 3753433 仍在 D2 拟合中。使用 py-spy 获取其栈及对象地址，再通过只读 /proc/PID/mem 解码现有 CPython 3.10 对象，保存 RECOVERED_LOADED_EVIDENCE.json：840 formal receipts、4200 oracle rows、5 effect estimation arrays、2320 estimation receipts、1485 qualification receipts。没有再次读取原始 formal 文件。缓存通过120 units、840唯一unit-condition键、612 negative receipts等不变量核验。

缓存保存后终止原进程。原脚本入口已禁用，只保留失败实现以审计。修复计算只能使用恢复缓存，source_guard 拦截原始 rollout 和 phase5–8 证据路径。

已知 dataset 加载尝试为2。原脚本 source_snapshot 还读取了旧产物字节以哈希，完整的跨进程逐文件打开次数没有保存，记为 null，不能将进程内计数1包装成全局物理读取1。

其余修复：D4 按负观测聚类，D1 披露采样总体不一致，D2 使用冻结的L2惩罚，D5用真正ledger事件重放。修复不得抹去上述违规；任何formal派生量均 claim_eligible=false、selection_eligible=false。

补充访问审计：审阅代理还自报恢复后一次递归grep意外涉及旧formal路径；其所报include过滤条件与JSONL路径存在不一致，未重开源文件验证，实际访问数记null。修复产物中的新增源读取0仅指缓存计算进程，不是全局工具访问计数。全局单次读取要求已明确不满足。
