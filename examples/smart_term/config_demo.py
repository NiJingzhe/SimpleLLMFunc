#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置化智能终端演示脚本
展示如何使用配置文件来自定义终端行为
"""

import os
import sys
from smart_term import TerminalSession, create_smart_pre_processor, smart_post_processor, Colors
from terminal_config import ConfigManager, TerminalConfig


def demo_basic_config():
    """演示基础配置的使用"""
    print(f"{Colors.BOLD}{Colors.CYAN}=== 基础配置演示 ==={Colors.END}")
    
    # 创建基础配置
    basic_config = TerminalConfig(
        full_control_commands=['vim', 'nano', 'less', 'top'],
        command_aliases={
            'll': 'ls -la --color=auto',
            'la': 'ls -a --color=auto'
        },
        auto_args={
            'ls': '--color=auto',
            'grep': '--color=auto'
        },
        dangerous_commands=['rm -rf /'],
        enable_suggestions=True,
        log_failed_commands=True,
        failed_commands_log="basic_failed_commands.log",
        prompt_template=f"{Colors.BLUE}[基础]{Colors.END} {{cwd}}$ ",
        welcome_message="🔧 基础配置终端已启动"
    )
    
    # 保存配置
    config_manager = ConfigManager("basic_config.json")
    config_manager.config = basic_config
    config_manager.save_config()
    
    # 创建终端session
    session = TerminalSession(
        config_file="basic_config.json",
        pre_processor=create_smart_pre_processor(basic_config),
        post_processor=smart_post_processor
    )
    
    print(f"{Colors.GREEN}基础配置已创建并保存到 basic_config.json{Colors.END}")
    print(f"{Colors.YELLOW}支持的交互式命令: {', '.join(basic_config.full_control_commands)}{Colors.END}")
    print(f"{Colors.YELLOW}可用别名: {', '.join(basic_config.command_aliases.keys())}{Colors.END}")
    print("试试输入 'help' 或 'config list' 查看更多信息")
    
    session.run()


def demo_developer_config():
    """演示开发者配置的使用"""
    print(f"{Colors.BOLD}{Colors.CYAN}=== 开发者配置演示 ==={Colors.END}")
    
    # 创建开发者配置
    developer_config = TerminalConfig(
        full_control_commands=[
            # 编辑器
            'vim', 'nvim', 'nano', 'emacs', 'code',
            # 查看器
            'less', 'more', 'man',
            # 系统监控
            'top', 'htop', 'btop',
            # 开发工具
            'python', 'python3', 'ipython', 'node', 'npm',
            'git', 'docker', 'kubectl',
            # 数据库
            'mysql', 'psql', 'redis-cli'
        ],
        command_aliases={
            # 文件操作
            'll': 'ls -la --color=auto',
            'la': 'ls -a --color=auto',
            'lt': 'ls -lt --color=auto',
            
            # Git 别名
            'gs': 'git status',
            'ga': 'git add',
            'gc': 'git commit',
            'gp': 'git push',
            'gl': 'git log --oneline --graph',
            'gd': 'git diff',
            'gb': 'git branch',
            'gco': 'git checkout',
            
            # Docker 别名
            'dps': 'docker ps',
            'di': 'docker images',
            'dc': 'docker-compose',
            'dcu': 'docker-compose up',
            'dcd': 'docker-compose down',
            
            # Python 别名
            'py': 'python3',
            'pip': 'pip3',
            
            # 其他工具
            'k': 'kubectl',
            'tf': 'terraform',
            'code': 'code .',
        },
        auto_args={
            'ls': '--color=auto',
            'grep': '--color=auto',
            'tree': '-C',
            'diff': '--color=auto',
            'docker ps': '--format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
        },
        dangerous_commands=[
            'rm -rf /',
            'dd if=/dev/zero',
            'docker system prune -a',
            'kubectl delete namespace',
            'terraform destroy'
        ],
        enable_suggestions=True,
        log_failed_commands=True,
        failed_commands_log="dev_failed_commands.log",
        prompt_template=f"{Colors.GREEN}[DEV]{Colors.END} {Colors.CYAN}{{cwd}}{Colors.END} ⚡ ",
        welcome_message="🚀 开发者终端环境已启动 - 支持 Git, Docker, Python, K8s 等工具"
    )
    
    # 保存配置
    config_manager = ConfigManager("developer_config.json")
    config_manager.config = developer_config
    config_manager.save_config()
    
    # 创建终端session
    session = TerminalSession(
        config_file="developer_config.json",
        pre_processor=create_smart_pre_processor(developer_config),
        post_processor=smart_post_processor
    )
    
    print(f"{Colors.GREEN}开发者配置已创建并保存到 developer_config.json{Colors.END}")
    print(f"{Colors.YELLOW}支持 {len(developer_config.full_control_commands)} 个交互式命令{Colors.END}")
    print(f"{Colors.YELLOW}包含 {len(developer_config.command_aliases)} 个便捷别名{Colors.END}")
    print("试试输入: gs, ga, gc, dps, py, k get pods 等命令")
    
    session.run()


def demo_custom_config():
    """演示自定义配置的创建"""
    print(f"{Colors.BOLD}{Colors.CYAN}=== 自定义配置演示 ==={Colors.END}")
    print("这里演示如何创建完全自定义的配置...")
    
    # 询问用户偏好
    print("\n请选择你想要的配置类型:")
    print("1. 最小化配置（只包含基本命令）")
    print("2. 系统管理员配置（系统监控和管理工具）")
    print("3. 数据科学配置（Python、R、Jupyter等）")
    
    choice = input("请输入选项 (1-3): ").strip()
    
    if choice == "1":
        config = TerminalConfig(
            full_control_commands=['vim', 'less', 'top'],
            command_aliases={'ll': 'ls -la'},
            auto_args={'ls': '--color=auto'},
            dangerous_commands=['rm -rf /'],
            enable_suggestions=False,
            log_failed_commands=False,
            failed_commands_log="",
            prompt_template=f"{Colors.WHITE}minimal:{Colors.END} ",
            welcome_message="最小化终端配置"
        )
        config_name = "minimal_config.json"
        
    elif choice == "2":
        config = TerminalConfig(
            full_control_commands=[
                'vim', 'nano', 'less', 'man',
                'top', 'htop', 'iotop', 'nethogs',
                'systemctl', 'journalctl'
            ],
            command_aliases={
                'll': 'ls -la --color=auto',
                'services': 'systemctl list-units --type=service',
                'ports': 'netstat -tlnp',
                'processes': 'ps aux',
                'disk': 'df -h',
                'memory': 'free -h'
            },
            auto_args={'ls': '--color=auto'},
            dangerous_commands=[
                'rm -rf /',
                'dd if=/dev/zero',
                'mkfs',
                'fdisk',
                'systemctl stop',
                'systemctl disable'
            ],
            enable_suggestions=True,
            log_failed_commands=True,
            failed_commands_log="sysadmin_failed_commands.log",
            prompt_template=f"{Colors.RED}[ADMIN]{Colors.END} {{cwd}}# ",
            welcome_message="🔧 系统管理员终端 - 谨慎操作"
        )
        config_name = "sysadmin_config.json"
        
    elif choice == "3":
        config = TerminalConfig(
            full_control_commands=[
                'vim', 'nano', 'less',
                'python', 'python3', 'ipython', 'jupyter',
                'R', 'Rscript',
                'mysql', 'psql'
            ],
            command_aliases={
                'll': 'ls -la --color=auto',
                'py': 'python3',
                'ipy': 'ipython',
                'jup': 'jupyter notebook',
                'lab': 'jupyter lab',
                'pip': 'pip3',
                'conda-env': 'conda env list',
                'r': 'R --no-save'
            },
            auto_args={
                'ls': '--color=auto',
                'jupyter': '--ip=0.0.0.0'
            },
            dangerous_commands=['rm -rf /'],
            enable_suggestions=True,
            log_failed_commands=True,
            failed_commands_log="datascience_failed_commands.log",
            prompt_template=f"{Colors.MAGENTA}[DS]{Colors.END} {{cwd}} 📊 ",
            welcome_message="📊 数据科学终端环境 - Python, R, Jupyter Ready!"
        )
        config_name = "datascience_config.json"
        
    else:
        print("无效选择，使用默认配置")
        config_manager = ConfigManager()
        config = config_manager.config
        config_name = "terminal_config.json"
    
    # 保存并使用配置
    config_manager = ConfigManager(config_name)
    config_manager.config = config
    config_manager.save_config()
    
    session = TerminalSession(
        config_file=config_name,
        pre_processor=create_smart_pre_processor(config),
        post_processor=smart_post_processor
    )
    
    print(f"\n{Colors.GREEN}自定义配置已保存到 {config_name}{Colors.END}")
    session.run()


def main():
    """主菜单"""
    print(f"{Colors.BOLD}{Colors.BLUE}智能终端配置演示{Colors.END}")
    print("这个演示展示了如何使用配置文件来自定义终端行为\n")
    
    while True:
        print(f"{Colors.BOLD}请选择演示类型:{Colors.END}")
        print("1. 基础配置演示")
        print("2. 开发者配置演示") 
        print("3. 自定义配置演示")
        print("4. 退出")
        
        choice = input(f"\n{Colors.CYAN}请输入选项 (1-4): {Colors.END}").strip()
        
        if choice == "1":
            demo_basic_config()
        elif choice == "2":
            demo_developer_config()
        elif choice == "3":
            demo_custom_config()
        elif choice == "4":
            print(f"{Colors.GREEN}再见！{Colors.END}")
            break
        else:
            print(f"{Colors.RED}无效选择，请重试{Colors.END}")


if __name__ == "__main__":
    main()
