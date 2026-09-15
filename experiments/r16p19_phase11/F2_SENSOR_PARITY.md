# F2 逐维传感器审计

下表分类描述原则上的真机可观测性；当前没有目标真机硬件清单，所以真实平台可用性一律为 null，未把仿真记录当作硬件证据。

|维度|名称|分类|来源/真机替代|
|---:|---|---|---|
|0|base_mean_R|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|1|base_mean_G|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|2|base_mean_B|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|3|base_std_R|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|4|base_std_G|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|5|base_std_B|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|6|wrist_mean_R|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|7|wrist_mean_G|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|8|wrist_mean_B|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|9|wrist_std_R|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|10|wrist_std_G|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|11|wrist_std_B|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|12|base_temporal_delta_mean_R|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|13|base_temporal_delta_mean_G|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|14|base_temporal_delta_mean_B|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|15|base_temporal_delta_std_R|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|16|base_temporal_delta_std_G|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|17|base_temporal_delta_std_B|onboard_observable|RGB camera (matching base/wrist view), hardware and synchronization unverified; |
|18|proprio_0|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|19|proprio_1|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|20|proprio_2|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|21|proprio_3|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|22|proprio_4|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|23|proprio_5|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|24|proprio_6|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|25|proprio_7|onboard_observable|observation/state channel; encoder/FK semantic mapping requires verification; |
|26|log1p(contact_count)|privileged|simulator trace contact_count; force/torque or tactile contact estimator; not semantically equivalent without validation|
|27|linspace_episode_time|shortcut|future-dependent complete episode length normalization; no deployable equivalent; exclude from effect detector|
|28|task_onehot_0|onboard_observable|query/task metadata, not physical evidence; |
|29|task_onehot_5|onboard_observable|query/task metadata, not physical evidence; |
|30|task_onehot_9|onboard_observable|query/task metadata, not physical evidence; |
|31|effect_onehot_0|onboard_observable|query predicate metadata, not ground-truth label; |
|32|effect_onehot_1|onboard_observable|query predicate metadata, not ground-truth label; |
|33|effect_onehot_2|onboard_observable|query predicate metadata, not ground-truth label; |
|34|effect_onehot_3|onboard_observable|query predicate metadata, not ground-truth label; |

contact_count 的力/触觉替代是否已可用：null。proprio 仍可能携带位姿/进度捷径；可观测不等于可迁移。task/effect one-hot 是查询条件，不是效果发生证据。
原始 `_feature_rows` 18:26 的状态通道语义与真实机器人8维状态的对应关系待确认；不臆造关节数量或硬件。
