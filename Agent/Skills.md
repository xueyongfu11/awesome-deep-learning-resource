[TOC]


## Skills

- [AI agent to evaluate and score resumes](https://github.com/interviewstreet/hiring-agent)

- [AI infra/ agent (vllm/sglang/torch等) skills整理](https://zhuanlan.zhihu.com/p/2023063871259177169)

- [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)

- [hermes tools](https://github.com/NousResearch/hermes-agent/tree/main/tools)
  - Hermes 的 tools 目录提供约 71 个可调用工具，涵盖终端与文件操作、浏览器和网络搜索、子 Agent 委派、代码执行、Skills 与记
  忆、定时任务、消息发送、多媒体生成分析、Kanban 多 Agent 协作，以及 Home Assistant、飞书、Discord、Spotify 和腾讯元宝等
  集成；其中部分 Python 文件只是注册、安全、状态管理等内部实现模块，并非 Agent 可直接调用的工具。

- [hermes skills](https://github.com/NousResearch/hermes-agent/tree/main/skills)
  - Hermes 的 skills 目录是一套按领域分类的流程知识库，覆盖 Apple 生态、自治 Agent、创作、数据科学、DevOps、邮件、GitHub、
  媒体、MLOps、笔记、效率、科研、智能家居、社交媒体和软件开发等场景；具体包括 Codex/Claude Code 使用、任务规划、系统化调
  试、测试驱动开发、代码审查、ArXiv 检索、论文写作等 Skill。它们主要通过 SKILL.md 告诉 Agent“何时以及按什么步骤完成任
  务”，本身不是底层执行工具，而是指导 Agent 组合调用 tools 中的能力。