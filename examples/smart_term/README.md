# Smart Terminal Session

一个功能丰富的Python终端Session类，支持彩色输出、实时命令执行和自定义前置处理，现在集成了AI智能功能！

## 主要特性

- 🎨 **彩色输出**: 支持丰富的终端颜色和格式
- ⚡ **实时执行**: 实时显示命令输出，就像真正的终端一样
- 🔧 **自定义前置处理**: 支持命令别名、参数补全、安全检查等
- 📊 **后置处理**: 命令执行统计、错误分析、智能建议
- 🏠 **内置命令**: cd, pwd, history, help, clear等
- 🛡️ **安全特性**: 危险命令检测和阻止
- 📝 **命令历史**: 自动记录和查看命令历史
- 🤖 **AI智能命令生成**: 使用自然语言生成命令 (以`?`开头)
- 🧠 **AI智能结果解释**: 自动解释命令执行结果 (以`?`结尾)

## 新增AI功能

### 🤖 智能命令生成
输入以`?`开头的自然语言描述，AI会自动生成对应的命令：

```bash
# 输入
?查看当前目录的文件

# AI生成
ls -lha --color=auto

# 系统会询问是否执行
是否执行此命令? (y/N): y
```

### 🧠 智能结果解释
在命令后加上`?`，AI会自动解释命令的执行结果：

```bash
# 输入
ps aux?

# 执行命令后，AI会解释进程列表的含义
🧠 [AI解释] 这个命令显示了系统中所有正在运行的进程...
```

### 组合使用
可以同时使用两个功能：

```bash
# 输入
?显示系统信息?

# AI先生成命令，执行后再解释结果
```

## 快速开始

### 基本用法

```python
from smart_term import TerminalSession

# 创建基本的终端session
session = TerminalSession()
session.run()
```

### 高级用法

```python
from smart_term import TerminalSession

def my_pre_processor(command: str) -> str:
    """自定义前置处理"""
    # 命令别名
    if command == 'll':
        return 'ls -la --color=auto'
    return command

def my_post_processor(command: str, return_code: int, command_output: str):
    """自定义后置处理"""
    if return_code != 0:
        print(f"命令失败: {command}")

# 创建带有自定义处理的session
session = TerminalSession(
    prompt="MyTerm> ",
    pre_processor=my_pre_processor,
    post_processor=my_post_processor
)
session.run()
```

## 运行示例

### 1. 基本终端
```bash
python smart_term.py
```

### 2. AI增强演示版本（推荐）
```bash
python demo.py
```

**注意**: AI功能需要配置`../provider.json`文件来设置LLM接口。

## AI功能配置

确保在上级目录有正确配置的`provider.json`文件：

```json
{
  "volc_engine": {
    "deepseek-v3-250324": {
      "api_key": "your-api-key",
      "base_url": "https://ark.cn-beijing.volces.com/api/v3",
      "model": "ep-xxx"
    }
  }
}
```

## 内置命令

- `help` - 显示帮助信息
- `cd <目录>` - 切换工作目录
- `pwd` - 显示当前目录
- `history` - 显示命令历史
- `clear` - 清屏
- `exit` / `quit` - 退出终端

## 快捷键

- `Ctrl+C` - 中断当前命令（不退出终端）
- `Ctrl+D` - 退出终端
- `回车` - 执行命令

## 智能功能演示

### AI命令生成
```bash
# 自然语言生成命令
?查看当前目录下的文件

# AI自动转换为
ls -lha --color=auto
```

### AI结果解释
```bash
# 在命令后加?获得AI解释
df -h?

# AI会解释磁盘使用情况的含义
```

### 组合使用
```bash
# 同时使用生成和解释
?检查网络连接状态?

# AI生成命令 → 执行 → AI解释结果
```

### 命令别名
```bash
# 输入
ll

# 自动转换为
ls -la --color=auto
```

### 参数补全
```bash
# 输入
grep test

# 自动补全为
grep --color=auto -n test
```

### 安全检查
```bash
# 危险命令会被自动阻止
rm -rf /
# ⚠️ 警告: 检测到危险命令，已阻止执行！
```

### 智能建议
```bash
# 命令失败时提供建议
cd nonexistent
# ❌ 命令执行失败 (返回码: 1)
# 💡 建议: 目录不存在或无权限，使用 'ls' 查看可用目录
```

## 文件结构

```
smart_term/
├── smart_term.py          # 主要的TerminalSession类
├── demo.py                # 高级功能演示
├── README.md              # 本文档
├── terminal_session.log   # 命令执行日志（运行后生成）
└── failed_commands.log    # 失败命令日志（运行后生成）
```

## API 文档

### TerminalSession 类

#### 构造函数参数

- `prompt` (Optional[str]): 自定义提示符
- `pre_processor` (Optional[Callable]): 前置处理函数
- `post_processor` (Optional[Callable]): 后置处理函数  
- `working_dir` (Optional[str]): 工作目录

#### 主要方法

- `run()`: 启动终端主循环
- `process_command(command: str)`: 处理单个命令
- `execute_command(command: str) -> int`: 执行命令并返回状态码

### Colors 类

提供终端颜色常量：

- `RED`, `GREEN`, `YELLOW`, `BLUE`, `MAGENTA`, `CYAN`, `WHITE`
- `BOLD`, `UNDERLINE`, `END`
- `colorize(text: str, color: str) -> str`: 给文本添加颜色

## 扩展示例

### 自定义主题

```python
def create_custom_session():
    session = TerminalSession(
        prompt=f"{Colors.BOLD}{Colors.BLUE}[MyApp]{Colors.END} {Colors.CYAN}➜{Colors.END} "
    )
    return session
```

### 命令日志记录

```python
def logging_post_processor(command: str, return_code: int, command_output: str):
    import logging
    logging.basicConfig(filename='commands.log', level=logging.INFO)
    logging.info(f"Command: {command}, Return Code: {return_code}")
```

### 环境变量处理

```python
def env_pre_processor(command: str) -> str:
    import os
    # 自动展开环境变量
    return os.path.expandvars(command)
```

## 依赖要求

- Python 3.6+
- 标准库模块：os, sys, subprocess, threading, pty, select 等

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
