# 智能终端配置系统

## 🎯 概述

智能终端现在支持完全可配置的交互式命令管理！你可以通过配置文件来自定义：

- 🎮 **交互式命令列表**：哪些命令需要完全终端控制
- 🔗 **命令别名**：创建便捷的命令快捷方式  
- ⚙️ **自动参数**：为特定命令自动添加参数
- ⚠️ **危险命令检测**：防止误执行危险操作
- 🎨 **个性化界面**：自定义提示符和欢迎消息

## 📁 配置文件结构

配置文件为JSON格式，包含以下字段：

```json
{
  "full_control_commands": [
    "vim", "nano", "less", "top", "python3", "git"
  ],
  "command_aliases": {
    "ll": "ls -la --color=auto",
    "gs": "git status",
    "py": "python3"
  },
  "auto_args": {
    "ls": "--color=auto",
    "grep": "--color=auto"
  },
  "dangerous_commands": [
    "rm -rf /",
    "dd if=/dev/zero"
  ],
  "enable_suggestions": true,
  "log_failed_commands": true,
  "failed_commands_log": "failed_commands.log",
  "prompt_template": "{cwd}$ ",
  "welcome_message": "欢迎使用智能终端"
}
```

## 🚀 快速开始

### 1. 使用默认配置

```bash
python3 smart_term.py
```

第一次运行时会自动创建 `terminal_config.json` 配置文件。

### 2. 查看和管理配置

在智能终端中使用内置命令：

```bash
# 查看所有交互式命令
config list

# 查看所有别名
config aliases

# 添加新的交互式命令
config add htop

# 移除交互式命令
config remove htop

# 重新加载配置
config reload
```

### 3. 使用预设配置

我们提供了几个预设配置：

```bash
# 开发者配置演示
python3 config_demo.py
```

选项：
- **基础配置**：最小化的命令集
- **开发者配置**：包含Git、Docker、Python等工具
- **系统管理员配置**：系统监控和管理工具
- **数据科学配置**：Python、R、Jupyter等

## 🔧 配置详解

### 交互式命令 (full_control_commands)

这些命令会获得完全的终端控制，支持：
- 全屏界面（vim, nano, less）
- 实时更新（top, htop）
- 交互式输入（python, mysql）
- 键盘快捷键（所有vim命令）

**默认包含**：
- **编辑器**：vim, nano, emacs
- **查看器**：less, more, man
- **监控工具**：top, htop, watch  
- **解释器**：python, node, R
- **数据库**：mysql, psql
- **工具**：git, docker, ssh

### 命令别名 (command_aliases)

创建便捷的命令快捷方式：

```json
{
  "ll": "ls -la --color=auto",
  "gs": "git status", 
  "py": "python3",
  "dps": "docker ps --format 'table {{.Names}}\\t{{.Status}}'"
}
```

### 自动参数 (auto_args)

为命令自动添加常用参数：

```json
{
  "ls": "--color=auto",
  "grep": "--color=auto",
  "tree": "-C"
}
```

### 个性化设置

```json
{
  "prompt_template": "[DEV] {cwd} ⚡ ",
  "welcome_message": "🚀 开发环境已启动"
}
```

## 📝 配置示例

### 开发者配置
```json
{
  "full_control_commands": [
    "vim", "nano", "less", "python3", "node", 
    "git", "docker", "kubectl"
  ],
  "command_aliases": {
    "gs": "git status",
    "ga": "git add", 
    "gc": "git commit",
    "dps": "docker ps",
    "k": "kubectl"
  },
  "prompt_template": "[DEV] {cwd} ⚡ ",
  "welcome_message": "🚀 开发环境已启动"
}
```

### 系统管理员配置
```json
{
  "full_control_commands": [
    "vim", "less", "top", "htop", "systemctl"
  ],
  "command_aliases": {
    "services": "systemctl list-units --type=service",
    "ports": "netstat -tlnp",
    "disk": "df -h"
  },
  "dangerous_commands": [
    "rm -rf /", "mkfs", "fdisk", "systemctl stop"
  ],
  "prompt_template": "[ADMIN] {cwd}# "
}
```

## 🎮 实际使用

### 1. 配置管理

```bash
# 启动智能终端
python3 smart_term.py

# 查看当前配置
config list
config aliases

# 添加自定义命令
config add htop
config add mycustomtool

# 重新加载配置（修改JSON文件后）
config reload
```

### 2. 使用别名

```bash
# 使用 ll 别名 (等同于 ls -la --color=auto)
ll

# 使用 gs 别名 (等同于 git status)  
gs

# 使用 py 别名 (等同于 python3)
py
```

### 3. 交互式程序

```bash
# 这些命令会获得完全终端控制
vim file.txt    # 完整的vim功能
less file.txt   # 支持上下翻页
top             # 实时系统监控
python3         # 交互式Python解释器
```

## 🛠️ 高级用法

### 创建多个配置文件

```python
from smart_term import TerminalSession
from terminal_config import ConfigManager

# 使用特定配置文件
session = TerminalSession(config_file="my_custom_config.json")
session.run()
```

### 编程方式修改配置

```python
from terminal_config import ConfigManager

config_manager = ConfigManager()

# 添加新命令
config_manager.add_full_control_command("neovim")

# 添加别名
config_manager.add_alias("nv", "neovim")

# 保存配置
config_manager.save_config()
```

### 导入导出配置

```python
config_manager = ConfigManager()

# 导出配置
config_manager.export_config("backup_config.json")

# 导入配置
config_manager.import_config("shared_config.json")
```

## 🔍 故障排除

### 配置文件损坏
如果配置文件格式错误，系统会自动使用默认配置并提示错误信息。

### 命令不工作
1. 检查命令是否在 `full_control_commands` 列表中
2. 使用 `config list` 查看当前配置
3. 使用 `config reload` 重新加载配置

### 别名冲突
别名会覆盖原命令，确保别名名称不与重要命令冲突。

## 📚 更多信息

- 运行 `python3 config_demo.py` 查看配置演示
- 查看 `terminal_config.py` 了解配置管理API
- 编辑 `terminal_config.json` 直接修改配置

## 🎯 最佳实践

1. **备份配置**：重要配置要备份
2. **测试新命令**：添加新的交互式命令后要测试
3. **合理使用别名**：避免覆盖重要的系统命令
4. **定期更新**：根据使用习惯调整配置
5. **团队共享**：可以导出配置给团队成员使用
