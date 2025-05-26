#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能终端配置文件
支持可配置的交互式命令、别名、前置处理规则等
"""

import json
import os
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TerminalConfig:
    """终端配置类"""
    
    # 需要完全终端控制的命令
    full_control_commands: List[str]
    
    # 命令别名
    command_aliases: Dict[str, str]
    
    # 自动添加参数的规则
    auto_args: Dict[str, str]
    
    # 安全检查的危险命令
    dangerous_commands: List[str]
    
    # 是否启用命令建议
    enable_suggestions: bool
    
    # 是否记录失败命令
    log_failed_commands: bool
    
    # 日志文件路径
    failed_commands_log: str
    
    # 提示符模板
    prompt_template: Optional[str]
    
    # 欢迎消息
    welcome_message: Optional[str]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "terminal_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def get_default_config(self) -> TerminalConfig:
        """获取默认配置"""
        return TerminalConfig(
            full_control_commands=[
                # 文本编辑器
                'vim', 'vi', 'nvim', 'nano', 'emacs', 'micro',
                
                # 分页器和查看器
                'less', 'more', 'most', 'man', 'info',
                
                # 系统监控
                'top', 'htop', 'btop', 'atop', 'iotop', 'watch',
                
                # 交互式解释器
                'python', 'python3', 'ipython', 'python2',
                'node', 'nodejs', 'deno', 'bun',
                'ruby', 'irb', 'php', 'lua',
                'R', 'octave', 'matlab',
                
                # 数据库客户端
                'mysql', 'psql', 'sqlite3', 'mongo', 'redis-cli',
                
                # 终端复用器
                'tmux', 'screen',
                
                # 其他交互式工具
                'gdb', 'lldb', 'pdb', 'pudb',
                'ssh', 'telnet', 'nc', 'netcat',
                'ftp', 'sftp',
                'docker', 'kubectl',
                'git' # git 有时需要交互式操作
            ],
            
            command_aliases={
                # 常用ls别名
                'll': 'ls -la --color=auto',
                'la': 'ls -a --color=auto',
                'l': 'ls -CF --color=auto',
                'lt': 'ls -lt --color=auto',  # 按时间排序
                'lh': 'ls -lh --color=auto',  # 人类可读大小
                
                # grep别名
                'grep': 'grep --color=auto',
                'egrep': 'egrep --color=auto',
                'fgrep': 'fgrep --color=auto',
                'rg': 'rg --color=auto',  # ripgrep
                
                # 系统信息
                'df': 'df -h',  # 人类可读的磁盘使用情况
                'du': 'du -h',  # 人类可读的目录大小
                'free': 'free -h',  # 人类可读的内存信息
                
                # 网络工具
                'ping': 'ping -c 4',  # 默认ping 4次
                'wget': 'wget -c',  # 支持断点续传
                'curl': 'curl -L',  # 跟随重定向
                
                # Git别名
                'gs': 'git status',
                'ga': 'git add',
                'gc': 'git commit',
                'gp': 'git push',
                'gl': 'git log --oneline',
                'gd': 'git diff',
                
                # 其他有用的别名
                'cls': 'clear',
                'md': 'mkdir -p',
                'rd': 'rmdir',
                'cp': 'cp -i',  # 交互式复制
                'mv': 'mv -i',  # 交互式移动
                'rm': 'rm -i',  # 交互式删除
            },
            
            auto_args={
                'ls': '--color=auto',
                'grep': '--color=auto',
                'egrep': '--color=auto',
                'fgrep': '--color=auto',
                'tree': '-C',  # 彩色输出
                'diff': '--color=auto',
                'ip': '--color=auto',
            },
            
            dangerous_commands=[
                'rm -rf /',
                'dd if=/dev/zero',
                'mkfs',
                'fdisk',
                'parted',
                'shutdown',
                'reboot',
                'halt',
                'init 0',
                'init 6',
                ':(){ :|:& };:',  # fork bomb
            ],
            
            enable_suggestions=True,
            log_failed_commands=True,
            failed_commands_log="failed_commands.log",
            prompt_template=None,  # 使用默认
            welcome_message=None   # 使用默认
        )
    
    def load_config(self) -> TerminalConfig:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                    # 将字典转换为TerminalConfig对象
                    return TerminalConfig(**config_dict)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"配置文件格式错误，使用默认配置: {e}")
                return self.get_default_config()
        else:
            # 创建默认配置文件
            default_config = self.get_default_config()
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: TerminalConfig = None):
        """保存配置到文件"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(config), f, ensure_ascii=False, indent=2)
            print(f"配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def add_full_control_command(self, command: str):
        """添加需要完全终端控制的命令"""
        if command not in self.config.full_control_commands:
            self.config.full_control_commands.append(command)
            self.save_config()
            print(f"已添加交互式命令: {command}")
        else:
            print(f"命令 {command} 已存在于交互式命令列表中")
    
    def remove_full_control_command(self, command: str):
        """移除需要完全终端控制的命令"""
        if command in self.config.full_control_commands:
            self.config.full_control_commands.remove(command)
            self.save_config()
            print(f"已移除交互式命令: {command}")
        else:
            print(f"命令 {command} 不在交互式命令列表中")
    
    def add_alias(self, alias: str, command: str):
        """添加命令别名"""
        self.config.command_aliases[alias] = command
        self.save_config()
        print(f"已添加别名: {alias} -> {command}")
    
    def remove_alias(self, alias: str):
        """移除命令别名"""
        if alias in self.config.command_aliases:
            del self.config.command_aliases[alias]
            self.save_config()
            print(f"已移除别名: {alias}")
        else:
            print(f"别名 {alias} 不存在")
    
    def list_full_control_commands(self):
        """列出所有需要完全终端控制的命令"""
        print("需要完全终端控制的命令:")
        for i, cmd in enumerate(sorted(self.config.full_control_commands), 1):
            print(f"  {i:2d}. {cmd}")
    
    def list_aliases(self):
        """列出所有别名"""
        print("命令别名:")
        for alias, command in sorted(self.config.command_aliases.items()):
            print(f"  {alias:<10} -> {command}")
    
    def is_dangerous_command(self, command: str) -> bool:
        """检查是否为危险命令"""
        command_lower = command.lower().strip()
        return any(dangerous in command_lower for dangerous in self.config.dangerous_commands)
    
    def export_config(self, export_file: str):
        """导出配置到指定文件"""
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
            print(f"配置已导出到: {export_file}")
        except Exception as e:
            print(f"导出配置失败: {e}")
    
    def import_config(self, import_file: str):
        """从指定文件导入配置"""
        try:
            with open(import_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                self.config = TerminalConfig(**config_dict)
                self.save_config()
            print(f"配置已从 {import_file} 导入")
        except Exception as e:
            print(f"导入配置失败: {e}")


def create_sample_configs():
    """创建示例配置文件"""
    
    # 基础配置
    basic_config = TerminalConfig(
        full_control_commands=['vim', 'nano', 'less', 'top'],
        command_aliases={'ll': 'ls -la', 'la': 'ls -a'},
        auto_args={'ls': '--color=auto'},
        dangerous_commands=['rm -rf /'],
        enable_suggestions=True,
        log_failed_commands=True,
        failed_commands_log="failed_commands.log",
        prompt_template=None,
        welcome_message=None
    )
    
    # 开发者配置
    developer_config = TerminalConfig(
        full_control_commands=[
            'vim', 'nvim', 'nano', 'emacs',
            'less', 'more', 'man',
            'top', 'htop',
            'python', 'python3', 'ipython', 'node',
            'mysql', 'psql',
            'git', 'docker', 'kubectl'
        ],
        command_aliases={
            'll': 'ls -la --color=auto',
            'la': 'ls -a --color=auto',
            'gs': 'git status',
            'ga': 'git add',
            'gc': 'git commit',
            'gp': 'git push',
            'docker-ps': 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"',
            'k': 'kubectl'
        },
        auto_args={
            'ls': '--color=auto',
            'grep': '--color=auto',
            'tree': '-C',
            'diff': '--color=auto'
        },
        dangerous_commands=[
            'rm -rf /',
            'dd if=/dev/zero',
            'docker system prune -a',
            'kubectl delete namespace'
        ],
        enable_suggestions=True,
        log_failed_commands=True,
        failed_commands_log="dev_failed_commands.log",
        prompt_template=None,
        welcome_message="🚀 开发者终端环境已启动"
    )
    
    # 保存示例配置
    configs = {
        'basic_terminal_config.json': basic_config,
        'developer_terminal_config.json': developer_config
    }
    
    for filename, config in configs.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        print(f"示例配置已保存: {filename}")


if __name__ == "__main__":
    # 示例用法
    config_manager = ConfigManager()
    
    print("=== 终端配置管理器 ===")
    print("1. 列出交互式命令")
    config_manager.list_full_control_commands()
    
    print("\n2. 列出别名")
    config_manager.list_aliases()
    
    print("\n3. 创建示例配置文件")
    create_sample_configs()
    
    print(f"\n配置文件位置: {config_manager.config_file}")
    print("你可以直接编辑JSON文件或使用配置管理器的方法来修改配置")
