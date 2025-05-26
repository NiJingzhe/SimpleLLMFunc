#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能终端Session演示脚本
展示各种自定义前置处理功能
"""

from smart_term import TerminalSession, Colors
from datetime import datetime
import re
from SimpleLLMFunc import llm_function, tool
from SimpleLLMFunc import OpenAICompatible

dsv3_interface = OpenAICompatible.load_from_json_file("../provider.json")["volc_engine"]["deepseek-v3-250324"]


@llm_function(
    llm_interface=dsv3_interface        
)
def smart_command_generator(command: str, os: str = "mac") -> str: # type: ignore
    """
    智能命令生成器
    根据用户输入生成命令, 直接返回raw string，不要包裹任何markdown语法。

    例如：
    用以下参数
    command: "查看当前目录下的文件"
    os: "mac"

    返回：
    ls -lha --color=auto

    Args:
        command (str): 用户输入的命令
        os (str): 操作系统类型，默认为"mac"
    Returns:
        str: 生成的智能命令
    """

@llm_function(
    llm_interface=dsv3_interface
)
def explain_result(command: str, command_result: str, return_code: int) -> str: # type: ignore
    """
    智能结果解释器
    根据命令执行结果和返回码提供对于命令执行结果的详细解释或者建议。
    例如：
    用以下参数
    command: "ls nonexistent"
    command_result: "ls: cannot access 'nonexistent': No such file or directory"
    return_code: 2
    返回：
    "命令执行失败，可能是因为指定的文件或目录不存在。请检查路径是否正确，或者使用 'ls' 查看当前目录下的文件。"

    例如：
    用以下参数
    command: "ping baidu.com -c 4"
    command_result: PING baidu.com (110.242.68.66): 56 data bytes
64 bytes from 110.242.68.66: icmp_seq=0 ttl=50 time=44.187 ms
64 bytes from 110.242.68.66: icmp_seq=1 ttl=50 time=40.203 ms
64 bytes from 110.242.68.66: icmp_seq=2 ttl=50 time=40.759 ms
64 bytes from 110.242.68.66: icmp_seq=3 ttl=50 time=40.362 ms

--- baidu.com ping statistics ---
4 packets transmitted, 4 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 40.203/41.378/44.187/1.635 ms

    return_code: 0

    返回：
    "网络连接测试成功！ping命令向baidu.com发送了4个ICMP数据包，全部成功接收。关键指标解析：
    
    📡 目标解析：baidu.com解析为IP地址110.242.68.66，域名解析正常
    📦 数据包：每个包64字节，发送4个包，接收4个包，0%丢包率表示网络连接稳定
    ⏱️ 延迟分析：平均往返时间41.378ms，属于正常范围（<100ms为良好）
    🔄 TTL值：50表示数据包经过了14跳路由（初始64-50=14）
    
    网络状态：连接质量良好，可以正常访问互联网。如果延迟>100ms可能表示网络较慢，丢包>5%则需要检查网络问题。"

    Args:
        command (str): 执行的命令
        command_result (str): 命令执行结果
        return_code (int): 命令返回码
    Returns:
        str: 解释和建议，请给出具有丰富知识和教育意义的解释，帮助用户更好的理解输出或者错误。
    """


def advanced_pre_processor(command: str) -> str:
    """
    高级前置处理示例
    包含更多智能功能，集成LLM智能命令生成
    """
    original_command = command
    
    # 检查是否需要AI解释后缀（保存状态）
    needs_explanation = command.rstrip().endswith("?")
    
    # 0. LLM智能命令生成 - 检查以"?"开头的命令
    if command.startswith("?"):
        print(f"{Colors.CYAN}🤖 [AI助手] 正在生成智能命令...{Colors.END}")
        try:
            # 移除开头的"?"并获取用户的自然语言描述
            user_request = command[1:].strip()
            
            # 如果用户请求也以"?"结尾，移除它（这是为了AI解释，不是生成请求的一部分）
            if user_request.endswith("?"):
                user_request = user_request[:-1].strip()
                needs_explanation = True  # 确保标记需要解释
            
            # 调用LLM生成智能命令
            generated_command = smart_command_generator(user_request, os="mac")
            
            # 显示生成结果
            print(f"{Colors.GREEN}🎯 [AI生成] {user_request} → {generated_command}{Colors.END}")
            
            # 询问用户是否执行
            try:
                confirm = input(f"{Colors.YELLOW}是否执行此命令? (y/N): {Colors.END}").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    command = generated_command
                    # 如果原始命令需要AI解释，在生成的命令后添加"?"标记
                    if needs_explanation:
                        command += "?"
                    print(f"{Colors.MAGENTA}✨ [执行AI命令] {command}{Colors.END}")
                    
                    # AI生成的命令直接返回，不再进行后续处理
                    return command
                else:
                    print(f"{Colors.YELLOW}❌ 已取消执行{Colors.END}")
                    return "echo '用户取消执行AI生成的命令'"
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Colors.YELLOW}❌ 已取消执行{Colors.END}")
                return "echo '用户取消执行AI生成的命令'"
                
        except Exception as e:
            print(f"{Colors.RED}❌ [AI错误] 智能命令生成失败: {e}{Colors.END}")
            return "echo 'AI命令生成失败'"
    
    # 1. 命令别名替换
    aliases = {
        'll': 'ls -la --color=auto',
        'la': 'ls -la --color=auto',
        'l': 'ls -CF',
        'grep': 'grep --color=auto',
        'tree': r'find . -type d | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"',
        'ports': 'lsof -i -P -n | grep LISTEN',
        'myip': 'curl -s ifconfig.me',
        'weather': 'curl -s "wttr.in?format=3"',
    }
    
    parts = command.split()
    if parts and parts[0] in aliases:
        parts[0] = aliases[parts[0]]
        command = ' '.join(parts)
    
    # 2. 智能参数补全 - 跳过以"?"结尾的命令（用于AI解释）
    if parts and not needs_explanation:
        cmd = parts[0]
        
        # 为ls命令自动添加颜色
        if cmd == 'ls' and '--color' not in command:
            command += ' --color=auto'
        
        # 为grep命令自动添加颜色和行号
        elif cmd == 'grep' and '--color' not in command:
            command += ' --color=auto -n'
        
        # 为ps命令自动添加用户友好格式
        elif cmd == 'ps' and len(parts) == 1:
            command = 'ps aux'
        
        # 为curl命令自动添加进度条和User-Agent
        elif cmd == 'curl' and '-s' not in command and '--silent' not in command:
            command += ' --progress-bar -A "Smart Terminal Session"'
    
    # 3. 安全检查
    dangerous_commands = ['rm -rf /', 'sudo rm -rf', 'dd if=', 'mkfs', 'fdisk']
    for dangerous in dangerous_commands:
        if dangerous in command:
            print(f"{Colors.RED}⚠️  警告: 检测到危险命令，已阻止执行！{Colors.END}")
            return "echo '命令被安全检查阻止'"
    
    # 4. 智能路径补全提示
    if 'cd ' in command and len(parts) >= 2:
        path = parts[1]
        if '~' not in path and not path.startswith('/'):
            print(f"{Colors.YELLOW}💡 提示: 使用相对路径 '{path}'{Colors.END}")
    
    # 5. 显示处理结果
    if command != original_command:
        print(f"{Colors.MAGENTA}🔄 [智能处理] {original_command} → {command}{Colors.END}")
    
    return command


def analytics_post_processor(command: str, return_code: int, command_output: str):
    """
    分析型后置处理
    提供命令执行统计和建议，集成LLM智能结果解释
    """
    original_command = command
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 检查是否需要AI解释 - 以"?"结尾的命令
    need_ai_explanation = original_command.rstrip().endswith("?")
    
    # 记录所有命令到日志文件
    log_entry = f"[{timestamp}] RC:{return_code} | {command}\n"
    with open("terminal_session.log", "a", encoding="utf-8") as f:
        f.write(log_entry)
    
    # 失败命令的特殊处理
    if return_code != 0 and return_code != 130:
        print(f"{Colors.RED}❌ 命令执行失败 (返回码: {return_code}){Colors.END}")
        
        # 如果需要AI解释
        if need_ai_explanation:
            print(f"{Colors.CYAN}🤖 [AI助手] 正在分析失败原因...{Colors.END}")
            try:
                # 清理输出中的颜色代码用于AI分析
                clean_output = re.sub(r'\x1b\[[0-9;]*m', '', command_output)
                ai_explanation = explain_result(command, clean_output, return_code)
                print(f"{Colors.MAGENTA}🧠 [AI分析] {ai_explanation}{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}❌ [AI错误] 智能分析失败: {e}{Colors.END}")
        
        # 提供智能建议
        parts = command.split()
        if parts:
            cmd = parts[0]
            
            if cmd == 'ls' and return_code == 2:
                print(f"{Colors.YELLOW}💡 建议: 目录可能不存在，尝试 'ls -la' 查看当前目录{Colors.END}")
            
            elif cmd == 'cd' and return_code == 1:
                print(f"{Colors.YELLOW}💡 建议: 目录不存在或无权限，使用 'ls' 查看可用目录{Colors.END}")
            
            elif cmd in ['grep', 'find'] and return_code == 1:
                print(f"{Colors.YELLOW}💡 建议: 没有找到匹配结果，检查搜索模式或路径{Colors.END}")
            
            elif cmd == 'python' and return_code == 127:
                print(f"{Colors.YELLOW}💡 建议: Python未安装或不在PATH中，尝试 'python3'{Colors.END}")
            
            elif return_code == 127:
                print(f"{Colors.YELLOW}💡 建议: 命令未找到，检查拼写或使用 'which {cmd}' 查找{Colors.END}")
        
        # 记录失败命令
        with open("failed_commands.log", "a", encoding="utf-8") as f:
            f.write(log_entry)
    
    else:
        print(f"{Colors.GREEN}✅ 命令执行成功{Colors.END}")
        
        # 如果需要AI解释成功的结果
        if need_ai_explanation:
            print(f"{Colors.CYAN}🤖 [AI助手] 正在解释执行结果...{Colors.END}")
            try:
                # 清理输出中的颜色代码用于AI分析
                clean_output = re.sub(r'\x1b\[[0-9;]*m', '', command_output)
                ai_explanation = explain_result(command, clean_output, return_code)
                print(f"{Colors.MAGENTA}🧠 [AI解释] {ai_explanation}{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}❌ [AI错误] 智能解释失败: {e}{Colors.END}")
        
        # 成功命令的统计和建议
        parts = command.split()
        if parts:
            cmd = parts[0]
            
            if cmd == 'git' and len(parts) > 1:
                git_cmd = parts[1]
                if git_cmd == 'clone':
                    print(f"{Colors.CYAN}💡 提示: 克隆完成后，别忘了 'cd' 进入项目目录{Colors.END}")
                elif git_cmd == 'add':
                    print(f"{Colors.CYAN}💡 提示: 别忘了执行 'git commit' 提交更改{Colors.END}")
            
            elif cmd == 'npm' and 'install' in command:
                print(f"{Colors.CYAN}💡 提示: 依赖安装完成，可以运行 'npm start' 或 'npm run dev'{Colors.END}")
            
            elif cmd == 'pip' and 'install' in command:
                print(f"{Colors.CYAN}💡 提示: Python包安装完成，现在可以在代码中导入使用{Colors.END}")


def demo_session():
    """演示不同配置的terminal session"""
    print(f"{Colors.BOLD}{Colors.BLUE}=== AI增强智能终端Session演示 ==={Colors.END}")
    print()
    print("这个演示展示了以下功能：")
    print(f"  {Colors.GREEN}✓{Colors.END} 彩色输出和实时命令执行")
    print(f"  {Colors.GREEN}✓{Colors.END} 智能命令别名和参数补全")
    print(f"  {Colors.GREEN}✓{Colors.END} 安全检查和危险命令阻止")
    print(f"  {Colors.GREEN}✓{Colors.END} 命令执行统计和智能建议")
    print(f"  {Colors.GREEN}✓{Colors.END} 内置命令支持 (cd, pwd, history等)")
    print(f"  {Colors.MAGENTA}🤖 AI智能命令生成 (以?开头){Colors.END}")
    print(f"  {Colors.MAGENTA}🧠 AI智能结果解释 (以?结尾){Colors.END}")
    print()
    print(f"{Colors.YELLOW}可以尝试的命令：{Colors.END}")
    print("  ll                    # 别名演示")
    print("  grep test             # 自动参数补全")
    print("  ps                    # 智能格式化")
    print("  help                  # 查看内置命令")
    print("  history               # 查看命令历史")
    print("  cd /tmp               # 目录切换")
    print("  pwd                   # 显示当前目录")
    print()
    print(f"{Colors.MAGENTA}🤖 AI功能演示：{Colors.END}")
    print("  ?查看当前目录的文件      # AI生成命令")
    print("  ?找出占用磁盘空间最多的文件 # AI生成复杂命令")
    print("  ls -la?              # 执行后AI解释结果")
    print("  ps aux?              # 执行后AI解释进程信息")
    print("  ?显示系统信息?          # 同时使用AI生成和解释")
    print()
    
    # 创建带有高级功能的session
    session = TerminalSession(
        config_file="./terminal_config.json",
        pre_processor=advanced_pre_processor,
        post_processor=analytics_post_processor
    )
    
    print(f"{Colors.CYAN}启动AI增强智能终端... (输入 'exit' 退出){Colors.END}")
    print()
    session.run()


if __name__ == "__main__":
    demo_session()
