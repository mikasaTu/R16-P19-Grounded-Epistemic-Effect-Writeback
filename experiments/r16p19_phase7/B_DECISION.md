# B_DECISION — step8 / S1b 人工交接

**UNCERTIFIABLE_NO_OPERATING_POINT；G1' 未获认证。** 这不是 Phase-6/7 最终报告。

1. 每 effect formal TPR LCB≥0.90：无所选变体，不能判通过。B2三个稀有effect正样本14/14/15，可达最高LCB仅0.784689/0.784689/0.796117；B6全部20个目标无完整认证向量。B3可估计的soup LCB=0.872347也未过0.90，mug LCB=0.911920。
2. formal假升级UCB≤0.05且显著低于B3的50%：未评估，无所选变体；UCB与p值均为null。F的10/612只用于诊断，不作为门证据。
3. 选择过程formal访问：已执行的B2/B5/B6选择阶段为0，B2 seal及读路径审计已保存；没有伪造B6工作点seal。
4. Phase-5判决复现：1680行、mismatch=0，PASS。

B1 在FPR≤2%和≤5%下均可达min effect TPR=1.0，假升级10/612（1.633987%），因此没有走容量不足的早停分支，继续执行了后续可行步骤。

B4的min与sum(log p)完整曲线已保存；B5已算可用small_mlp的同准则值，但linear权重/逐样本分数未保存，禁止重训下无法完成双模型重选。qualification曾参与原始选型，且和pilot共享15个环境seed，不能声称全流程独立held-out。

B7因缺少封存工作点未执行；S2候选没有有效集合。空B7_S2_UNITS.txt仅为交付占位，必须结合B7_S2_UNITS_STATUS.json（execution_allowed=false、candidate_count=null）读取，不能解释为零分歧。

交回：B1_CEILING.json、B3_HELDOUT.json、B7_REPLAY.json及候选状态。详见STEP8_CHECKPOINT.md。等待人工确认是否另行修订数据/候选保存条件；本任务不擅自改B0、不重训、不新增rollout、不启动S2、不提交PAI。
