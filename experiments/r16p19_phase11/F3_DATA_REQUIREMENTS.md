# F3 数据权限盘点与缺失数据规格

QPILOTS代码仓库可读：`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS`，HEAD `b30397a348d7b6bb4ddc5045ff59255ca2ce7ca1`。代码仓库有已有未提交改动，本轮未修改。`/mnt/cpfs/zbl-cpfs-new/dataset/leon/qpilots`可读，但有界搜索未找到arrange_3_flowers、arrange-3-flowers或arrange3flowers；x2robot_data顶层亦无arrange*。现有simple_pi05/wrc数据没有被当作目标数据。

仓库读取权限存在，但目标接管数据位置未知。故episode数、接管率、策略/人工段是否可判、每花入瓶正负数量、可自动导出的provenance数量均为null。本结果不等于证明目标数据全局不存在。

|需求|最小内容|标注粒度|
|---|---|---|
|身份|episode/unit/task/condition/seed、花身份、瓶身份、策略checkpoint/config SHA|episode及对象|
|时钟|单调frame/action/sensor timestamp、同步误差、相机曝光时刻|每帧/每动作|
|视觉|同步base/wrist原生RGB，内外参、有效深度（如有）|帧及时间窗口|
|执行|建议动作、实际执行动作、控制权来源、接管开始/结束、动作完成|事件|
|谓词|每支花是否入瓶、发生/失效时间、稳定性、遮挡/不确定|effect×frame/区间|
|provenance|realized-by-policy/external/imagined或unknown、证据链接、标注来源|effect×事件|
|质量|双标注一致率、争议复核、缺失率、时间抖动、边界样本|episode及effect|

初始探索预算为每谓词97正+97负独立episode（最保守p=0.5、95%半宽约0.10）；三个谓词联合95%约144正+144负/谓词。真实正例发生率、ICC与可标注率未知，换算所需总episode数为null。F5给出设计效应与split要求。该预算是规划假设，不是采集结果。
