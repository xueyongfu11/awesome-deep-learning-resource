[TOC]



# 0. 总体框架（全链路分层）

**RAG 系统 = Knowledge Pipeline（离线/在线） + Retrieval & Orchestration（查询时） + Grounded Generation（生成） + Evaluation & Governance（闭环） + Engineering（工程化）**



# 1. 知识与数据层（Knowledge Sources & Data Modeling）

## 1.1 知识源类型

* **非结构化**：PDF/Word/HTML/Markdown/邮件/聊天记录/日志
* **半结构化**：JSON/YAML/API 文档/配置/表单
* **结构化**：SQL 表、数据仓库、指标系统
* **图结构**：知识图谱（实体-关系）、业务图（组织/权限/依赖）
* **多模态**：图像（截图/扫描件）、表格图片、音频/视频转写 + 视觉内容

## 1.2 语义建模与本体

* 主题/标签体系、实体抽取（人/组织/产品/时间/地点）
* 事件建模（时间线、版本、变更记录）
* 规范化字段（来源、时间戳、权限域、可信度、文档类型）

# 2. 数据处理与文档构建层（Ingestion & Preprocessing）

## 2.1 清洗与质量控制

* 去重（精确/近似）、去噪（模板/脚注/导航栏）、纠错
* 版面恢复（标题层级、目录、表格结构）
* 语言检测/分句/术语统一

## 2.2 分块（Chunking）体系

* **长度驱动**：固定 token / 固定字符
* **结构驱动**：标题感知、段落/列表/代码块感知
* **语义驱动**：embedding 相似度断点、主题漂移断点
* **滑窗**：overlap、multi-granularity（多尺度 chunk：句/段/节/文档）
* **任务驱动**：FAQ chunk、定义 chunk、步骤 chunk、表格行/列 chunk

## 2.3 元数据与版本管理

* 文档版本、发布日期/生效期、所属系统、作者、权限标签
* “黄金段落/关键段落”标注（可人工/弱监督）

# 3. 表示与索引层（Representation & Indexing）

## 3.1 表示（Representation）

* **稀疏检索**：BM25/倒排、稀疏向量（如 SPLADE 类）
* **稠密检索**：Dense Embedding（双塔）
* **混合检索**：Sparse + Dense 融合（加权/学习融合）
* **多向量/晚交互**：ColBERT 类（token-level matching）
* **结构化表示**：表格 embedding、schema embedding、KG triple embedding
* **多模态表示**：图文 joint embedding、OCR token embedding

## 3.2 索引结构

* 向量索引：HNSW/IVF/Flat（近似/精确）
* 倒排索引：关键词检索、字段过滤
* 分层索引：文档→段落→句子（coarse-to-fine）
* 图索引：实体索引、关系索引、社区/子图索引（GraphRAG 常用）
* 多租户索引：tenant/shard、权限分区

## 3.3 索引更新策略

* 离线全量构建
* 在线增量（append/merge）
* 近实时（CDC / streaming ingestion）
* 冷热分层（hot index + cold archive）

# 4. 查询理解与意图层（Query Understanding）

* 意图识别：问答/查流程/查定义/查对比/生成文案/代码/诊断
* 实体识别与消歧（产品名、版本号、组织名）
* 结构化改写：把自然语言→检索 DSL（过滤字段、时间范围、权限域）
* 多轮对话状态：指代消解（“这个/上面那个”）、用户偏好、上下文限制

# 5. 检索策略层（Retrieval Strategies）

## 5.1 单跳检索

* Top-K dense / sparse / hybrid
* 字段过滤（时间、部门、权限、语言）

## 5.2 查询增强（Query Augmentation）

* Query Rewriting（更贴近文档语料）
* Query Expansion（同义词/术语扩展）
* Multi-query（生成多条查询并聚合）
* HyDE（先生成“假想答案/文档”再检索）
* Step-back（先问更抽象的问题再检索）

## 5.3 多跳与迭代检索（Reasoning-aware Retrieval）

* **Decomposition**：问题拆子问题→逐步检索
* **Multi-hop**：根据证据提出下一跳查询
* **Self-ask / Self-refine**：回答-反思-再检索
* **Coverage-driven**：以“证据覆盖率”为目标的检索循环

## 5.4 路由（Routing）

* 多库路由（政策库/产品库/代码库/数据仓库）
* 多模态路由（需要图/表就走对应索引）
* 工具路由（需要计算/SQL/API 就转 Tool）

# 6. 重排、过滤与证据选择层（Rerank & Evidence Selection）

## 6.1 Reranker 类型

* Cross-encoder rerank（语义精排）
* LLM rerank / judge（用指令判断相关性/可用性）
* 多目标 rerank：相关性 + 新鲜度 + 权威性 + 多样性

## 6.2 证据过滤

* 权限过滤（ACL/RBAC/ABAC）
* 新鲜度/版本过滤（只取有效版本）
* 去冗余（clustering、MMR、多样性约束）
* 可信度过滤（来源白名单/引用链）

## 6.3 证据粒度选择

* doc-level / section-level / chunk-level / sentence-level / span-level
* 表格：行/列/单元格证据抽取
* 代码：函数/调用栈片段抽取

# 7. 上下文构造层（Context Construction / Augmentation）

## 7.1 组织方式

* 直接拼接（baseline）
* 结构化拼接：按主题/子问题分组、标题层级、bullet
* 证据图谱：把证据节点和关系放入 prompt（GraphRAG 风格）
* 引用友好：每段证据带 source_id/段落号

## 7.2 压缩与选择

* Extractive compression（抽句/抽span）
* Abstractive summarization（摘要压缩）
* Map-Reduce summarization（多文档汇总）
* Query-focused compression（只保留与 query 相关信息）
* 动态预算：给“关键证据”更高 token 配额

## 7.3 冲突处理

* 证据冲突检测（版本不一致/来源不一致）
* 冲突解释策略：展示多来源 + 给出判据（时间、权威性）

# 8. 生成与对齐层（Grounded Generation）

## 8.1 生成范式

* Extract-then-Answer（先抽证据再回答）
* Cite-then-Explain（先列引用后解释）
* Plan-then-Write（先大纲再写作）
* 多答案候选 + 选择（self-consistency / verifier）

## 8.2 Grounding 约束

* 强制引用（每个关键断言必须有 citation）
* 引用一致性检查（claim ↔ evidence 对齐）
* 不足则拒答（insufficient evidence → ask clarification）

## 8.3 结构化输出

* JSON schema / function calling 输出
* 可执行查询（SQL/API）+ 解释
* 面向业务：工单模板、排障步骤、风险提示

# 9. 评测层（Evaluation & Benchmarking）

## 9.1 离线评测维度

* 检索质量：Recall@K、MRR、nDCG
* 证据质量：Evidence precision/coverage、重复率、多样性
* 生成质量：正确性、完整性、可读性、可操作性
* 可信性：Faithfulness/groundedness（是否被证据支持）
* 引用质量：citation accuracy（引用是否真的支撑该句）
* 鲁棒性：对噪声 query、对抗 query、长问题、多轮对话

## 9.2 在线评测与 A/B

* 任务成功率、用户满意度、人工纠错率
* 回答时延、成本、缓存命中率
* 失败归因：检索失败/证据不足/生成幻觉/权限拦截

# 10. 学习与优化闭环（Learning & Optimization）

* 数据闭环：用户反馈、点击/复制、人工标注
* 负例挖掘：hard negatives、混淆实体
* 检索微调：embedding 领域适配、对比学习
* reranker 微调：pairwise/listwise
* 端到端 RAG 优化：把“最终正确性/可信性”作为训练目标（含 RL/偏好优化）
* 自改进：失败样例聚类→自动生成新测试集/新规则

# 11. 治理、安全与合规层（Governance, Security, Compliance）

* 权限控制：行级/段级权限、租户隔离
* 数据泄露防护：prompt 注入防护、敏感信息脱敏、输出过滤
* 溯源审计：回答-证据-来源链路日志
* 合规模型：数据保留策略、删除权（Right to be Forgotten）
* 安全策略：工具调用白名单、域名白名单、执行沙盒

# 12. 工程化与系统层（Engineering & Ops）

* 缓存体系：query cache、embedding cache、retrieval cache、context cache
* 性能：并行检索、多路召回、流式生成、降级策略（无 rerank/低K）
* 可观测性：trace（检索→重排→生成）、指标、告警、回放
* 成本控制：动态选择模型（小模型检索/大模型生成）、自适应K、预算分配
* 部署形态：本地知识库/云知识库、边缘/端侧、离线/在线混合

# 13. 按“RAG 系统形态”的高层分类

1. **Naive RAG**：单次检索 + 拼接 + 生成
2. **Advanced RAG**：混合召回 + rerank + 压缩 + 引用约束
3. **Iterative / Reasoning RAG**：分解问题、多跳检索、自反思循环
4. **Agentic RAG**：Planner-Executor + 工具调用 + 多库路由
5. **GraphRAG**：实体/关系/社区子图检索 + 图结构上下文
6. **Multimodal RAG**：图文表混合检索 + 结构化证据对齐
7. **Trustworthy / Governed RAG**：权限、安全、审计、引用一致性为核心
8. **RAG + Long-Context 融合**：长上下文直接读 + 检索做“聚焦/压缩/路由”（不是替代，而是组合）
