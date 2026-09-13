# Shadow Music Generator

一个 MCP 服务器：为 Shadow Producers 排队并执行**音乐生成任务**（YuE 及兼容流程）—— 除非你明确要求，
它不会下载任何权重、也不会启动模型。

[English](README.md) · 许可证：[AGPL-3.0](LICENSE)

## 功能

- **默认 dry-run**：提交任务只做校验、记录模型许可证并写出任务文件，不会真的跑模型。
- **带历史的任务队列**：每个任务都是一个 JSON 文件，含状态、流水线阶段、耗时、产物与错误。
- **自带流水线**：本地执行通过 `SHADOW_PIPELINE_FACTORY` 指向的适配器完成，插件只负责按顺序调用。
- **许可证透明**：每个任务结果都会记录模型许可证；插件不下载、不打包任何权重。

## 环境要求

| | |
| --- | --- |
| 系统 | macOS / Linux / Windows |
| Python | 3.9 或更新 |
| 运行时依赖 | 无（模型栈由你的适配器提供） |

## 安装

```sh
# 作为 Codex 插件
codex plugin marketplace add shadowroommusic/shadow-music-generator
codex plugin add shadow-music-generator@shadowroom

# 只用命令行
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/shadow-music-generator --help
```

MCP 客户端配置：

```json
{
  "mcpServers": {
    "shadow-music-generator": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/shadow-music-generator"
    }
  }
}
```

## 配置项

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `SHADOW_JOB_DIR` | `./.shadow-jobs` | 任务文件存放目录 |
| `--job-dir` | 同上 | 单次命令覆盖任务目录 |
| `SHADOW_PIPELINE_FACTORY` | – | 本地执行的适配器：`module:callable` 或 `/path/file.py:callable` |
| `--mode dry-run \| local` | `dry-run` | 只校验，或排队等待本地执行 |

## 工具

| 工具 | 作用 |
| --- | --- |
| `submit_generation` | 排入一个任务（`prompt`、`mode`、`model`、`lyrics`、`source_audio`、`output_dir`、`job_dir`、`run`） |
| `run_job` | 执行已排队任务，可用 `factory` 临时覆盖适配器 |
| `job_status` | 读取某个任务的状态、阶段、产物与错误 |

命令行等价：`shadow-music-generator submit`、`run`、`status`、`list`。

## 用法

```sh
# dry-run：校验请求并写出任务文件
.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno, 128 bpm'

# 排队做本地执行，然后运行
.venv/bin/shadow-music-generator submit --prompt 'techno' --mode local
.venv/bin/shadow-music-generator run --job JOB_ID

# 查询
.venv/bin/shadow-music-generator status --job JOB_ID
.venv/bin/shadow-music-generator list
```

本地运行需要适配器，例如：

```sh
export SHADOW_PIPELINE_FACTORY=/path/to/my_yue_adapter.py:make_pipeline
.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno' --mode local --run
```

适配器契约（阶段、context、产物）见 [docs/internals.md](docs/internals.md)。

## 模型许可证

本插件**不含任何模型权重**。YuE 代码是 Apache-2.0，当前 YuE2 权重单独采用 **CC BY-NC 4.0** 许可 ——
未经单独授权不要用于商业产品；每个任务结果里都会记录它运行时的许可证。

## 安全说明

- 只有在你指定 `--mode local` **并且**配置了适配器时，才会有本地执行；插件不会下载任何东西。
- 任务只写入 `output_dir` 与 `job_dir`；失败时已成功的阶段仍然保留。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `status: failed` 并指出某阶段 | 看任务文件里的 `stages`/`error`，失败阶段会明确标出。 |
| 本地执行不启动 | 设置 `SHADOW_PIPELINE_FACTORY`；没有适配器就没有可执行的东西。 |
| 适配器导入失败 | 用绝对路径写法 `/path/file.py:callable`，或确认模块在 `PYTHONPATH` 上。 |

## 参与开发

见 [CONTRIBUTING.md](CONTRIBUTING.md)；实现细节在 [docs/internals.md](docs/internals.md)。

## 许可证

AGPL-3.0，见 [LICENSE](LICENSE)。上游 YuE 代码与模型权重保留各自许可证。
