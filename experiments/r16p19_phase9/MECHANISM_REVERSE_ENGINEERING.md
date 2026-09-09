# step9a 机制反解（诊断记录，不生成新 idea）

## 保形秩与样本单位

`phase9_nonformal.order_threshold` 使用 ceil((n+1)*(1-alpha))。每 effect 只有15个负episode，四个alpha均得到秩16；有限阈值至少需要 alpha>=1/16=.0625，而网格最大值只有.05。因此全部阈值确定性退化为null/reject-all。这是冻结的有限样本规则造成的，不是从formal推断检测器容量不足。

D3零错误的episode CP上界为.181036（逐effect95%）/.264358（联合95%）。receipt次口径虽然负样本521/878/596/437/803更多，经验CP上界仍超过对应alpha。边际split-conformal保证和额外的经验二项上界认证不是同一结论；校准选择后的CP值不能冒充独立验证。

## Base-rate 重采样

`phase9_baserate.fit_platt_positive` 在标量logit上拟合正斜率Platt及L2=1e-6，固定正样本、重采样负样本，不读取/更新verifier参数。稳定logit排序保持AUC；概率数值饱和不被用来改变ceiling判决。

|目标正类率|极差中位数|CV中位数|
|---|---:|---:|
|0.02|0.9996924003|1.3864184208|
|0.05|0.9995583338|1.1761331592|
|0.10|0.9996007414|1.0525944963|
|0.25|0.9997283783|0.9464534353|
|0.50|0.9998311491|0.8882804074|

未均衡Platt极差=0.9998241453。5水平中4个极差中位数下降，满足冻结的方向性支持判据，但最佳下降仅0.0002658115（约0.026586%），极差仍接近1；50%水平还略增大。因此不能解释成尺度失配得到实质解决，更不是因果证明。两档FPR cap结论相同。

## Receipt-AND ledger 升降的同一原因

`PersistentPerEffectUpgradeLedger.process` 先重放原核事件，保留witness state、proof validity、attempt attribution；升级需所有detector通过且所有witness verified。null阈值令detector全失败，原有REALIZED witness可被外层保持为UNKNOWN/REOBSERVE。该机制同时令假升级降为0和五effect TPR降为0。

D4相对只读B1诊断参考：TPR 1→0，假升级10/612→0/612；不能把0假升级当作有用检测性能提升。D5实际重放1400 effect流/8000事件，reference kernel mismatch0，单位一致6/120，低于C的102/120。全量oracle增益683−456=227，当前一致unit保留4/227，历史C保留197/227。197/227是增益保留比例，不是cell一致率。

增益包络：对抗下界-0.944444444；继承oracle上界0.315277778；中性假设0.005555556。后二者依赖明示假设，不是新的rollout点估计。

## 证据限制

D1的最大描述性FPR/nominal=3.846154，但all-frame与injection-frame样本不匹配，不能单独认定split转移失败。D6假升级unit为0/120，无法检验A5/C0集中性或时序/归属混淆，相关量为null；空集合不证明机制不存在。

formal发生两次加载尝试，且前期哈希扫描的逐文件打开次数未完整保存。恢复缓存修复不抹去该违规；所有formal派生结果均claim_eligible=false，不能称为有效的一次formal认证实验。
