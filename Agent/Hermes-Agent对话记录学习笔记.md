[TOC]

# 从一次 Hermes Agent 会话看 System Prompt、Tools 和 Skills

最近我看了一次 Hermes Agent 的会话导出。用户只问了一句话：调研近期大模型知识注入相关论文。

这个任务本身不复杂。有意思的是过程：Hermes 没有直接凭模型记忆写一篇综述，而是先加载 arXiv 相关 skill，再用工具多轮检索论文，最后把结果按技术路线整理出来。

所以这篇笔记不重点讨论知识注入论文，而是借这次会话看一个 Agent 系统到底怎么工作：system prompt 在管什么，工具是怎么被调用的，skills 为什么有用，tools list 又说明了什么。

文中会保留几段原始英文片段，方便直接看 Hermes 给模型的指令是什么。涉及本机路径和个人环境的内容已经省略。

## 1. 先看整体流程

普通聊天模型的路径大概是：用户输入，模型回答。

Hermes 这种 Agent 更像下面这样：

```text
用户任务
  -> system prompt 给出规则、工具边界和执行要求
  -> 模型判断是否需要查资料、读文件、跑命令
  -> 调用 skill / terminal / execute_code / web 等工具
  -> 工具结果回到上下文
  -> 模型继续判断下一步
  -> 最终回答
```

这次会话里，第一步工具调用是：

```text
skill_view(name="arxiv")
```

后面多次调用 `execute_code`，用 Python 脚本访问 arXiv API 和 Semantic Scholar API。最后才生成论文综述。

最终答案不是模型拍脑袋写出来的，后面有一条可以回看的执行轨迹。

## 2. System Prompt：Agent 开工前拿到的说明书

很多人看 Agent，会先看工具列表。但我觉得 system prompt 更值得先看。工具决定 Agent 能做什么，system prompt 决定它应该怎么做、什么时候做、做到什么程度。

这次导出的 system prompt 不是一句简单的角色设定，而是分成了好几块：persona、memory、工具使用规则、执行纪律、澄清策略、上下文缺失处理、skill 加载规则、可用 skill 索引、运行环境提示。每一块都会影响后面的行为。

先看最前面的 persona。它表面上是在定义 Agent 的语气和身份，但里面也可以放工作流要求。比如这段：

```text
If the user asks about configuring, setting up, or using Hermes Agent itself,
load the `hermes-agent` skill with skill_view(name='hermes-agent') before answering.
```

这说明 persona 不只是风格配置。它也可以告诉 Agent，遇到某类问题时必须先做什么。这里的规则就是：只要聊 Hermes 自己，就先加载 `hermes-agent` skill。

接下来是 memory。Hermes 会把一些长期有用的信息保存下来，比如用户偏好、环境信息、项目习惯。但 prompt 里也明确提醒，不要把临时任务进度、PR 编号、issue 编号这类东西写进长期记忆。这个限制很有必要。否则 memory 很快会变成一堆过期信息，反而干扰后面的会话。

然后是工具使用规则。这一段是 Agent 味最重的地方：

```text
# Tool-use enforcement
You MUST use your tools to take action — do not describe what you would do or plan to do without actually doing it. When you say you will perform an action (e.g. 'I will run the tests', 'Let me check the file', 'I will create the project'), you MUST immediately make the corresponding tool call in the same response.

Keep working until the task is actually complete. Do not stop with a summary of what you plan to do next time. If you have tools available that can accomplish the task, use them instead of telling the user what you would do.
```

这段话的意思很直接：不要只说要做，真的去做。普通聊天助手经常会停在建议层，比如你可以运行这个命令。Agent 则应该直接调用工具，拿到结果，再继续下一步。

紧接着是 execution discipline，也就是执行习惯。它要求 Agent 不要过早停止；工具返回空结果或不完整结果时，要换一种方式重试；只有任务完成并验证后，才给最终回复。这块看起来像管理细节，但对长任务很重要。没有这类规则，Agent 很容易跑两步就开始总结。

mandatory tool use 是另一块硬规则：

```text
<mandatory_tool_use>
NEVER answer these from memory or mental computation — ALWAYS use a tool:
- Arithmetic, math, calculations → use terminal or execute_code
- Hashes, encodings, checksums → use terminal (e.g. sha256sum, base64)
- Current time, date, timezone → use terminal (e.g. date)
- System state: OS, CPU, memory, disk, ports, processes → use terminal
- File contents, sizes, line counts → use read_file, search_files, or terminal
- Git history, branches, diffs → use terminal
- Current facts (weather, news, versions) → use web_search
</mandatory_tool_use>
```

这解释了为什么这次论文调研不能靠模型记忆硬答。近期论文属于当前事实，必须查外部来源。类似地，文件内容、系统状态、Git diff、时间日期，也都不能靠猜。

再往下是 act_dont_ask。它要求 Agent 在默认含义很明显时直接行动，不要把简单问题反复丢回给用户。比如用户问 443 端口开了吗，默认就检查当前机器，而不是先追问是哪台机器。这个规则能减少很多没必要的来回。

prerequisite_checks 则是在提醒 Agent：动手前先看有没有前置条件。比如要改代码，先看项目结构；要跑测试，先确认测试命令；要写文件，先确认目标路径是否存在。它解决的是另一类常见问题：模型一上来就执行，结果漏掉前置信息。

verification 这块也很关键。它要求最终回答前检查四件事：结果是否满足需求，事实有没有依据，格式是否符合要求，是否涉及危险操作。换句话说，Agent 不能只完成动作，还要检查动作有没有做对。

missing_context 处理缺失上下文。它的原则是：能用工具查到的就不要问用户；工具查不到、又会影响结果时再澄清；如果只能带着假设继续，就把假设写清楚。这条规则能让 Agent 少问废话，也少瞎编。

然后是 skills mandatory：

```text
## Skills (mandatory)
Before replying, scan the skills below. If a skill matches or is even partially relevant to your task, you MUST load it with skill_view(name) and follow its instructions.

Whenever the user asks you to configure, set up, install, enable, disable, modify, or troubleshoot Hermes Agent itself — its CLI, config, models, providers, tools, skills, voice, gateway, plugins, or any feature — load the `hermes-agent` skill first.
```

这就是为什么这次论文调研一开始会调用 `skill_view(name="arxiv")`。用户没有要求使用 arXiv skill，但任务本身是论文调研，Hermes 就应该主动加载。

system prompt 里还放了一份 available_skills 索引。它不是完整 skill 内容，只是目录。原始片段大概是这样：

```text
<available_skills>
  autonomous-ai-agents:
    - hermes-agent: Configure, extend, or contribute to Hermes Agent.

  research:
    - arxiv: Search arXiv papers by keyword, author, category, or ID.

  software-development:
    - systematic-debugging: 4-phase root cause debugging: understand bugs before fixing.
    - test-driven-development: TDD: enforce RED-GREEN-REFACTOR, tests before code.
    - writing-plans: Write implementation plans: bite-sized tasks, paths, code.
</available_skills>
```

这个设计比较省上下文。system prompt 只放索引，等任务真的相关时，再通过 `skill_view` 读取完整说明。

最后是运行环境提示。公开文章里不需要暴露本机路径，这里只保留关键结构：

```text
Host: Windows (10)
User home directory: [redacted]
Current working directory: [redacted]

Shell: on this Windows host your `terminal` tool runs commands through bash (git-bash / MSYS), NOT PowerShell or cmd.exe. Use POSIX shell syntax (`ls`, `$HOME`, `&&`, `|`, single-quoted strings) inside terminal calls.
```

这类信息看起来不起眼，但能避免很多低级错误。同样是 Windows，终端可能是 PowerShell，也可能是 Git Bash。模型如果不知道 shell 是哪一个，生成的命令就可能完全不能跑。

所以我更愿意把 system prompt 看成一份开工说明书：它不只告诉模型扮演什么角色，还写清楚了怎么用工具、什么时候查证、什么时候加载 skill、如何处理不确定信息，以及当前到底运行在什么环境里。

## 3. Tool calls：先看过程，再看答案

这次导出里有一个 `tool_calls_audit.json`，它记录了工具调用。里面能看到调用了什么工具、传了什么参数、查询了哪些关键词、访问了哪些 URL。

第一条是加载 arXiv skill：

```json
{
  "message_index": 2,
  "tool": "skill_view",
  "args": {
    "name": "arxiv"
  }
}
```

后面主要是 `execute_code`。audit 里会记录 imports、queries、urls，大概长这样：

```json
{
  "tool": "execute_code",
  "imports": [
    "from hermes_tools import terminal",
    "import urllib.parse, xml.etree.ElementTree as ET, json, time"
  ],
  "queries": [
    "knowledge injection large language models",
    "knowledge editing large language models",
    "continual pretraining knowledge injection LLM"
  ],
  "urls": [
    "https://export.arxiv.org/api/query?search_query=...",
    "https://api.semanticscholar.org/graph/v1/paper/search?query=..."
  ]
}
```

这里能看出检索过程：先用宽泛关键词找论文，再补充知识编辑、持续预训练、GraphRAG、ROME、MEMIT、AlphaEdit、EasyEdit 等方向。

这比最终答案更有价值。最终答案只能告诉你写得好不好；工具轨迹能告诉你它为什么这么写。

`execute_code` 在这类任务里很合适。它可以循环多个查询词，解析 XML 和 JSON，去重、筛选，再把整理后的结果交回给模型。如果每个 HTTP 请求都单独展开，反而会让上下文很乱。

## 4. Tools list：Agent 的行动边界

下面是这次会话导出的原始 tools list：

```text
✓ enabled   web              Web Search & Scraping
✓ enabled   browser          Browser Automation
✓ enabled   terminal         Terminal & Processes
✓ enabled   file             File Operations
✓ enabled   code_execution   Code Execution
✓ enabled   vision           Vision / Image Analysis
✗ disabled  video            Video Analysis
✓ enabled   image_gen        Image Generation
✗ disabled  video_gen        Video Generation
✗ disabled  moa              Mixture of Agents
✓ enabled   tts              Text-to-Speech
✓ enabled   skills           Skills
✓ enabled   todo             Task Planning
✓ enabled   memory           Memory
✓ enabled   session_search   Session Search
✓ enabled   clarify          Clarifying Questions
✓ enabled   delegation       Task Delegation
✓ enabled   cronjob          Cron Jobs
✓ enabled   messaging        Cross-Platform Messaging
✗ disabled  homeassistant    Home Assistant
✗ disabled  spotify          Spotify
✗ disabled  yuanbao          Yuanbao
✓ enabled   computer_use     Computer Use (macOS)
```

这份列表很像 Agent 的能力边界。

`web` 和 `browser` 负责外部信息获取。`terminal`、`file`、`code_execution` 负责实际执行和文件操作。`skills`、`memory`、`todo`、`delegation`、`cronjob` 则让 Hermes 有了长期记忆、任务管理、子 Agent 和定时任务能力。

禁用项也要看。比如 video、spotify、homeassistant 没启用，就意味着当前会话不能碰这些能力。Agent 系统不能只看模型多强，还要看它被允许接触什么工具。

这次论文调研没有直接用 web 或 browser，而是用 `execute_code` 调 API。这个选择挺合理：论文检索 API 比网页搜索更容易批量处理，也更容易留下清楚的查询记录。

## 5. Skill：给 Agent 看的操作手册

Hermes 的 skill 不是模型本身的能力，更像一份可加载的操作手册。比如查论文、做代码审查、处理 PDF、操作 GitHub PR，都可以写成 skill。

一个 skill 通常会包含：触发条件、推荐工具、操作步骤、常用命令、API、输出格式、注意事项，有时还会带脚本或模板。

system prompt 里会放一份 skill 索引。原始片段大概是这样：

```text
<available_skills>
  autonomous-ai-agents: Skills for spawning and orchestrating autonomous AI coding agents and multi-agent workflows.
    - hermes-agent: Configure, extend, or contribute to Hermes Agent.

  research: Skills for academic research, paper discovery, literature review, domain reconnaissance, market data, content monitoring, and scientific knowledge retrieval.
    - arxiv: Search arXiv papers by keyword, author, category, or ID.

  software-development:
    - systematic-debugging: 4-phase root cause debugging: understand bugs before fixing.
    - test-driven-development: TDD: enforce RED-GREEN-REFACTOR, tests before code.
    - writing-plans: Write implementation plans: bite-sized tasks, paths, code.
</available_skills>
```

这里有个设计挺好：system prompt 里只放索引，不把所有 skill 全塞进去。任务相关时，再用 `skill_view` 读取完整 skill。这样既省上下文，也避免模型被无关信息干扰。

Skill 和 memory 容易混在一起，但差别不小。

Memory 适合放长期稳定事实，比如用户偏好、项目约定、环境细节。Skill 适合放流程。比如用户喜欢中文回答，这是 memory；如何查 arXiv 论文，这是 skill。

这次任务里，Hermes 先加载 arXiv skill，相当于先看一眼查论文的流程，再动手检索。

## 6. 这次论文调研是怎么跑起来的

把轨迹压缩一下，大概是这样：

```text
用户：调研近期大模型知识注入相关论文

Hermes：
1. 判断这是研究任务，不能只靠模型记忆
2. 加载 arXiv skill
3. 用 execute_code 查询 arXiv / Semantic Scholar
4. 多轮调整关键词
5. 汇总论文元信息
6. 按方向整理成综述
```

查询词也能看出思路，从宽到窄：

```text
knowledge injection large language models
knowledge editing large language models
knowledge infusion large language models
continual pretraining knowledge injection LLM
knowledge graph enhanced large language models survey
```

后面又补了几个代表方法：

```text
ROME
MEMIT
GRACE
AlphaEdit
EasyEdit
```

最后答案按几个方向组织：参数内知识注入、知识编辑、RAG / GraphRAG、结构化知识、安全和攻击风险。

这就是工具和模型各自做的事：工具负责拿事实，模型负责归纳、分类和表达。

## 7. 几点观察

第一，system prompt 不能只写角色。

你是一个有帮助的助手这种句子没什么用。对 Agent 来说，要写清楚：什么时候必须用工具，什么时候不要问用户，怎么处理缺失上下文，怎么验证结果，怎么加载 skill。

第二，工具列表决定了 Agent 能做什么。

模型再强，没有文件工具就不能读写项目；没有 terminal 就不能跑测试；没有 web 或 API 检索，就不能查当前事实。Agent 能力不是模型单独决定的，而是模型、prompt、tools、skills、memory 和权限共同决定的。

第三，skill 是经验复用。

如果一个任务经常出现，就不应该每次靠模型临场发挥。把流程写成 skill，下次直接加载，稳定性会好很多。

第四，过程要能审计。

研究任务要能看检索源，编程任务要能看改了哪些文件、跑了哪些测试，运维任务要能看执行了哪些命令。否则最终答案写得再顺，也很难判断它靠不靠谱。

## 8. 小结

这次会话让我更直观地看到 Hermes Agent 的几块东西：

```text
System Prompt：规定行为规则
Tools：提供外部行动能力
Skills：保存可复用流程
Memory：保存长期稳定信息
Session Export：留下可审计轨迹
Tools List：说明当前能力边界
```

对于论文调研这种任务，Hermes 的路线是：先判断需要当前信息，再加载 arXiv skill，用工具检索论文，最后整理成综述。

好的 Agent 不只是会写答案。它要知道什么时候该查、该用哪个工具、查完怎么整理，还要把过程留下来。这样最终结果才不只是看起来像那么回事，而是有迹可循。