# step8 / S1b checkpoint（非 Phase-6/7 最终报告）

当前状态：B1 上界诊断通过继续条件；B2 无法认证完整阈值向量。B3 做完可定义的部分检验，B4 两类曲线已完成，B5 因 linear 权重缺失无法完成双候选重选，B6 的 20 个目标均无法形成认证工作点；B7 依赖缺失，未重放、不生成有效 S2 候选。

## B0 与边界

B0 在任何 B1 数据分析前封存，commit 8871bf9792cf029c765becfb61aad16a6b0253c6，SHA256 7fd8f1cc3392aea5a375abe293843305178c789d5e38f243e34f9738f04f17ec。B2 在 qualification 计算前提交封存，commit a4bb9fb。选择规则始终未改；B2/B5/B6 选择过程有文件访问阻断，formal 读取数 0。F 从未进入任何选择 seal。

全部 CPU，GPU/PAI/rollout/S2 均为 0。Phase-5/6 原产物只读。

## B1 天花板（仅诊断，claim_eligible=false）

| Receipt FPR 约束 | 可达 min effect TPR | 假升级 | 实际 receipt FPR |
| --- | ---: | ---: | ---: |
| 0.02 | 1.000000 | 10/612 | 0.016340 |
| 0.05 | 1.000000 | 10/612 | 0.016340 |

840 个去重 receipt、120 个 unit；228 positive receipt、612 negative receipt。下面阈值使用了 formal 标签，仅供上界诊断，永远不能被选作工作点。

| Effect | 上界诊断阈值 | TP/positive | TPR |
| --- | ---: | ---: | ---: |
| task0:effect0:in alphabet_soup_1 basket_1_contain_region | 0.93185788393 | 80/80 | 1.000000 |
| task0:effect1:in tomato_sauce_1 basket_1_contain_region | 0.0203938111663 | 76/76 | 1.000000 |
| task5:effect0:in black_book_1 desk_caddy_1_back_contain_region | 0.0487076938152 | 80/80 | 1.000000 |
| task9:effect0:in white_yellow_mug_1 microwave_1_heating_region | 0.990629255772 | 74/74 | 1.000000 |
| task9:effect1:close microwave_1 | 0.0585017316043 | 72/72 | 1.000000 |

不能据此下结论“检测器容量不足”。F 只说明这个冻结 formal 样本上的分数存在可分工作区间；它既不能证明泛化，也不能替代独立阈值认证。

## B1 拟合来源与独立性审计

natural 共 210 集，但 base-model fit 只读取 seed 0–4 的 150 集：13138 frame-effect rows。seed 5/6 的 60 集未用于拟合，但 natural split 整体是训练来源，未事后重定义成新的 B2 split。calibration 曾拟合 Platt-like scale/bias 和原阈值；本次依用户明确要求纳入 calibration+pilot。qualification 已用于原始模型选型，且其 clean 与 pilot noop 共享 15 个环境 seed，所以 B3 仅对本轮阈值估计 held-out，不是严格独立的全流程验证集。

计数单位为 frame-effect，episode 内存在时序相关性；以下 Wilson 是计划指定的计数区间，不能把相关帧声称为独立样本。

## B2 合并估计与 B3 held-out

calibration+pilot 共45集、4030 frame-effect rows。

| Effect | 正样本数 | 可达最高 Wilson LCB | 阈值 | 状态 |
| --- | ---: | ---: | ---: | --- |
| task0:effect0:in alphabet_soup_1 basket_1_contain_region | 371 | 0.989752 | 0.7940068244934082 | CERTIFIABLE_ON_ESTIMATION |
| task0:effect1:in tomato_sauce_1 basket_1_contain_region | 14 | 0.784689 | None | UNCERTIFIABLE |
| task5:effect0:in black_book_1 desk_caddy_1_back_contain_region | 14 | 0.784689 | None | UNCERTIFIABLE |
| task9:effect0:in white_yellow_mug_1 microwave_1_heating_region | 381 | 0.990018 | 0.8930768370628357 | CERTIFIABLE_ON_ESTIMATION |
| task9:effect1:close microwave_1 | 15 | 0.796117 | None | UNCERTIFIABLE |

三个稀有 effect 的 14/14/15 个正样本即使全对，LCB 也不足 0.80，更不可能达到 0.90；因此不能产生完整向量。这是认证数据不足，不是已证实的 formal 迁移失败。

| 可估计 effect | qualification TP/positive | TPR | Wilson 95% |
| --- | ---: | ---: | --- |
| task0:effect0:in alphabet_soup_1 basket_1_contain_region | 232/254 | 0.913386 | [0.872347, 0.942107] |
| task9:effect0:in white_yellow_mug_1 microwave_1_heating_region | 207/218 | 0.949541 | [0.911920, 0.971594] |

soup 的 held-out LCB=0.872347 也未达到 0.90；mug 的 LCB=0.911920 达到该条。三个缺阈值 effect 的 TPR/FPR 和完整 receipt 假升级数均记 null，不能写成 0。

## B4 D2 软聚合

完整曲线见 B4_SOFT.json，不强制零假阳。

| Split | 聚合 | ROC AUC | 曲线点数 | 有 effect 级判决 |
| --- | --- | ---: | ---: | --- |
| estimation | min | 0.983107107 | 2322 | True |
| estimation | sum_log_p | 0.983147961 | 2322 | False |
| qualification | min | 0.983705949 | 1487 | True |
| qualification | sum_log_p | 0.984274346 | 1487 | False |

min(scores) >= t 严格等价于每个 score >= t，可给出同量纲 effect 判决。sum(log p) 只给出 receipt 级分数，已排除出 G1' 候选；其略高 AUC 不能被当作 per-effect TPR 提升或认证成功。

## B5 原始选型与可复用产物

实际原始选择规则为 max(qualified, macro_auroc, -ece, prefer linear)，并非 loss/accuracy。两个候选原始 qualified 都为 false；small_mlp 的 macro AUC 0.991667 高于 linear 0.982177，因此被保存。

Phase-5 只序列化 selected small_mlp（6644 bytes；SHA256 edba3ff23ae71c5768760af161b601b3d1a9f65d37d1ddeb8304b2510e938005）。linear 只有历史聚合指标，无 w/b 或逐样本分数。因禁止重训，无法恢复其固定 FPR 排名或换模型后的 B2 结果。这部分明确为 BLOCKED_MISSING_LINEAR_CHECKPOINT，未伪称完成。

可用 small_mlp 在 calibration+pilot、receipt FPR<=0.02 下的 min effect TPR=0.666667，假升级 38/2277。这是选型准则的描述值；B2 Wilson 认证结果仍不变。linear 同准则值为 null，二者无法排序。

## B6 与 B7

按冻结规则扫描 0.80–0.99 共20个目标，所有目标都缺三个稀有 effect 的认证阈值。qualification 前沿条目全部保留，假升级率与 oracle 一致率为 null；不能画出不存在的完整向量曲线。formal 前沿同样不可定义且 selection_eligible=false。未创建 B6_OPERATING_POINT_SEAL.json。

B7 不具备封存工作点前提，未执行新重放。增益上下界、中性假设和分歧 unit 都为 null；B7_S2_UNITS.txt 是空占位文件，配套状态明确 candidate_count=null、execution_allowed=false，不能解释为“零分歧”或执行空集合。没有继承 S1 的18个/93个分歧集合，也没有使用 F 阈值。

## 机制反解（从代码和复算证据，不生成新 idea）

S1 固定分数复算：1680 行 mismatch=0；B/C oracle 一致率精确复现 0.733333/0.957143，C 一致 unit=102/120；一致 unit 的 oracle 增益分子为 197/227=86.784141%，对应原先约87%的结论。原始 S1 结论和文件未更改。

已确认的实现机制：原始全局阈值把不同 effect 的分数尺度混合，低分的稀有 effect 被 AND 规则直接否决；per-effect 阈值不改模型权重即可降低这些假阴性。B1 中 tomato 的上界阈值约0.02039，与 S1 calibration 阈值0.16626差距大；这只是 formal 诊断，不可反向写入选择。少量正样本能决定经验最小分，却无法认证总体召回率，解释了“经验判决改善”和“认证仍失败”同时出现。

已确认的统计区别：64/76 的 Wilson 区间 [0.743986,0.907305] 包含0.90，不能用点估计断言总体TPR<0.90；但同一区间下界也没有达到新G1'的认证要求。n=72–80 所需最少成功数是 n−2，经验TPR范围0.972222–0.975；用户计划的0.973是近似值。B0在分析前已写明按精确Wilson公式认证，规则未事后改变。

已确认的聚合机制：算术平均允许一个高分补偿另一个低分，而 AND 要求每一项都满足；min保留这个逻辑。sum(log p)使用联合低分惩罚，ROC略有变化，但不产生独立effect判决。未把微小AUC差异解释成已验证泛化收益。

待证机制：稀有正样本覆盖不足可能导致阈值尾部不稳定；现有相关帧和重用qualification不足以做独立因果验证。没有新rollout、没有新idea。

## 验证与交接

独立求解器审查与7项CPU测试见 TEST_REPORT.txt 和 CODE_REVIEW.md；测试含全局暴力枚举对照、Wilson反演、阈值tie、访问阻断、AUC ties及缺失值不伪装成0。

交接以 B_DECISION.md / B_DECISION.json 为准。停止在本轮 checkpoint，不启动S2、不提交PAI、不写Phase-6/7最终报告。
