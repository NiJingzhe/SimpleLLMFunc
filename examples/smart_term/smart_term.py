#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能终端Session类
支持彩色输出、实时命令执行和自定义前置处理
"""

import os
import sys
import subprocess
import threading
import shlex
import signal
from typing import Callable, Optional, Dict, Any, Tuple
from datetime import datetime
import pty
import select
import termios
import tty
import struct
import fcntl
import shutil
import select
import termios
import tty
from terminal_config import ConfigManager, TerminalConfig


class Colors:
    """终端颜色常量"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """给文本添加颜色"""
        return f"{color}{text}{cls.END}"


class TerminalSession:
    """智能终端Session类"""
    
    def __init__(self, 
                 prompt: Optional[str] = None,
                 pre_processor: Optional[Callable[[str], str]] = None,
                 post_processor: Optional[Callable[[str, int, str], None]] = None,
                 working_dir: Optional[str] = None,
                 config_file: Optional[str] = None):
        """
        初始化终端Session
        
        Args:
            prompt: 自定义提示符，默认为用户@主机名:工作目录$
            pre_processor: 前置处理函数，接收命令字符串，返回处理后的命令
            post_processor: 后置处理函数，接收命令和返回码
            working_dir: 工作目录，默认为当前目录
            config_file: 配置文件路径，默认为 terminal_config.json
        """
        # 加载配置
        self.config_manager = ConfigManager(config_file or "terminal_config.json")
        self.config = self.config_manager.config
        
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.working_dir = working_dir or os.getcwd()
        self.running = False
        self.history = []
        
        # 设置提示符
        if prompt is None:
            if self.config.prompt_template:
                self.prompt_template = self.config.prompt_template
            else:
                username = os.getenv('USER', 'user')
                hostname = os.uname().nodename
                self.prompt_template = f"{Colors.GREEN}{username}@{hostname}{Colors.END}:{Colors.BLUE}{{cwd}}{Colors.END}$ "
        else:
            self.prompt_template = prompt
    
    def get_prompt(self) -> str:
        """获取当前提示符"""
        if '{cwd}' in self.prompt_template:
            cwd = os.path.basename(self.working_dir) or self.working_dir
            return self.prompt_template.format(cwd=cwd)
        return self.prompt_template
    
    def print_colored(self, text: str, color: Optional[str] = None):
        """打印彩色文本"""
        if color:
            print(Colors.colorize(text, color))
        else:
            print(text)
    
    def print_header(self):
        """打印欢迎头部"""
        if self.config.welcome_message:
            # 使用自定义欢迎消息
            print(f"\n{Colors.CYAN}{'='*50}{Colors.END}")
            print(f"{Colors.BOLD}{self.config.welcome_message}{Colors.END}")
            print(f"{Colors.CYAN}{'='*50}{Colors.END}\n")
        else:
            # 使用默认欢迎消息
            header = f"""
{Colors.CYAN}╔══════════════════════════════════════╗
║          智能终端 Session            ║
║        Smart Terminal Session        ║
╚══════════════════════════════════════╝{Colors.END}

{Colors.YELLOW}提示：{Colors.END}
- 输入命令并按回车执行
- 输入 'exit' 或 'quit' 退出
- 使用 Ctrl+C 中断当前命令
- 输入 'help' 查看内置命令

"""
            print(header)
    
    def builtin_commands(self, command: str) -> bool:
        """
        处理内置命令
        
        Returns:
            True: 命令已处理
            False: 不是内置命令，需要继续处理
        """
        parts = command.strip().split()
        if not parts:
            return True
            
        cmd = parts[0].lower()
        
        if cmd in ['exit', 'quit']:
            self.print_colored("再见！", Colors.GREEN)
            return True
            
        elif cmd == 'help':
            help_text = f"""
{Colors.BOLD}内置命令：{Colors.END}
  {Colors.GREEN}cd <目录>{Colors.END}     - 改变工作目录
  {Colors.GREEN}pwd{Colors.END}          - 显示当前目录
  {Colors.GREEN}history{Colors.END}      - 显示命令历史
  {Colors.GREEN}clear{Colors.END}        - 清屏
  {Colors.GREEN}help{Colors.END}         - 显示此帮助
  {Colors.GREEN}exit/quit{Colors.END}    - 退出终端

{Colors.BOLD}配置管理命令：{Colors.END}
  {Colors.GREEN}config list{Colors.END}         - 列出交互式命令
  {Colors.GREEN}config aliases{Colors.END}      - 列出命令别名
  {Colors.GREEN}config add <cmd>{Colors.END}    - 添加交互式命令
  {Colors.GREEN}config remove <cmd>{Colors.END} - 移除交互式命令
  {Colors.GREEN}config reload{Colors.END}       - 重新加载配置

{Colors.BOLD}支持的功能：{Colors.END}
  {Colors.CYAN}• 统一交互式框架{Colors.END} - 所有命令都支持完整的终端功能
  {Colors.CYAN}• 全屏程序支持{Colors.END}   - vim, nano, less, man, top 等
  {Colors.CYAN}• 彩色输出{Colors.END}       - 自动保持命令的颜色输出
  {Colors.CYAN}• 信号处理{Colors.END}       - 正确处理 Ctrl+C, Ctrl+Z 等
  {Colors.CYAN}• 实时输出{Colors.END}       - 命令输出实时显示

{Colors.BOLD}示例命令：{Colors.END}
  {Colors.YELLOW}vim file.txt{Colors.END}  - 编辑文件（完整vim功能）
  {Colors.YELLOW}less file.txt{Colors.END} - 查看文件（支持翻页）
  {Colors.YELLOW}top{Colors.END}           - 系统监控（实时更新）
  {Colors.YELLOW}python3{Colors.END}       - Python交互式解释器
"""
            print(help_text)
            return True
            
        elif cmd == 'cd':
            if len(parts) > 1:
                try:
                    new_dir = os.path.expanduser(parts[1])
                    if os.path.isdir(new_dir):
                        self.working_dir = os.path.abspath(new_dir)
                        os.chdir(self.working_dir)
                        self.print_colored(f"已切换到: {self.working_dir}", Colors.GREEN)
                    else:
                        self.print_colored(f"目录不存在: {new_dir}", Colors.RED)
                except Exception as e:
                    self.print_colored(f"切换目录失败: {e}", Colors.RED)
            else:
                # 无参数时切换到家目录
                home_dir = os.path.expanduser("~")
                self.working_dir = home_dir
                os.chdir(self.working_dir)
                self.print_colored(f"已切换到: {self.working_dir}", Colors.GREEN)
            return True
            
        elif cmd == 'pwd':
            self.print_colored(self.working_dir, Colors.BLUE)
            return True
            
        elif cmd == 'history':
            if self.history:
                print(f"{Colors.BOLD}命令历史：{Colors.END}")
                for i, hist_cmd in enumerate(self.history[-20:], 1):  # 显示最近20条
                    print(f"{Colors.CYAN}{i:2d}.{Colors.END} {hist_cmd}")
            else:
                self.print_colored("暂无命令历史", Colors.YELLOW)
            return True
            
        elif cmd == 'clear':
            os.system('clear' if os.name != 'nt' else 'cls')
            return True
            
        elif cmd == 'config':
            # 配置管理命令
            if len(parts) < 2:
                self.print_colored("用法: config <子命令>", Colors.YELLOW)
                self.print_colored("可用子命令: list, aliases, add, remove, reload", Colors.YELLOW)
                return True
                
            subcmd = parts[1].lower()
            
            if subcmd == 'list':
                self.config_manager.list_full_control_commands()
            elif subcmd == 'aliases':
                self.config_manager.list_aliases()
            elif subcmd == 'add' and len(parts) > 2:
                cmd_to_add = parts[2]
                self.config_manager.add_full_control_command(cmd_to_add)
                # 重新加载配置
                self.config = self.config_manager.config
            elif subcmd == 'remove' and len(parts) > 2:
                cmd_to_remove = parts[2]
                self.config_manager.remove_full_control_command(cmd_to_remove)
                # 重新加载配置
                self.config = self.config_manager.config
            elif subcmd == 'reload':
                self.config_manager = ConfigManager(self.config_manager.config_file)
                self.config = self.config_manager.config
                self.print_colored("配置已重新加载", Colors.GREEN)
            else:
                self.print_colored(f"未知的配置子命令: {subcmd}", Colors.RED)
                self.print_colored("可用子命令: list, aliases, add <cmd>, remove <cmd>, reload", Colors.YELLOW)
            
            return True
            
        return False
    
    def requires_full_terminal_control(self, command: str) -> bool:
        """检查是否需要完全的终端控制（全屏、光标控制等）"""
        cmd_parts = command.strip().split()
        if cmd_parts:
            cmd_name = os.path.basename(cmd_parts[0])
            return cmd_name in self.config.full_control_commands
        return False
    
    def execute_command(self, command: str) -> Tuple[int, str]:
        """
        统一的命令执行器 - 支持所有类型的命令
        
        Returns:
            (命令的返回码, 命令输出)
        """
        captured_output = ""
        requires_raw_mode = self.requires_full_terminal_control(command)
        old_tty = None
        
        try:
            # 如果需要完全终端控制，保存当前终端属性
            if requires_raw_mode:
                old_tty = termios.tcgetattr(sys.stdin)
            
            # 创建pty
            master, slave = pty.openpty()
            
            # 设置pty的终端尺寸以匹配当前终端
            if requires_raw_mode:
                try:
                    # 获取当前终端的尺寸
                    cols, rows = shutil.get_terminal_size()
                    
                    # 设置pty的窗口尺寸
                    winsize = struct.pack('HHHH', rows, cols, 0, 0)
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, winsize)
                    
                    # 设置环境变量
                    env = os.environ.copy()
                    env['COLUMNS'] = str(cols)
                    env['LINES'] = str(rows)
                    env['TERM'] = 'xterm-256color'  # 确保支持颜色
                except Exception as e:
                    # 如果获取尺寸失败，使用默认值
                    env = os.environ.copy()
                    env['COLUMNS'] = '80'
                    env['LINES'] = '24'
                    env['TERM'] = 'xterm-256color'
            else:
                env = os.environ.copy()
                env['TERM'] = 'xterm-256color'
            
            # 启动子进程
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=self.working_dir,
                preexec_fn=os.setsid,
                env=env
            )
            
            os.close(slave)
            
            # 设置终端模式
            if requires_raw_mode:
                tty.setraw(sys.stdin.fileno())
                
                # 设置信号处理器来处理窗口尺寸变化
                def handle_sigwinch(signum, frame):
                    try:
                        cols, rows = shutil.get_terminal_size()
                        winsize = struct.pack('HHHH', rows, cols, 0, 0)
                        fcntl.ioctl(master, termios.TIOCSWINSZ, winsize)
                    except:
                        pass
                
                # 保存原始信号处理器
                old_winch_handler = signal.signal(signal.SIGWINCH, handle_sigwinch)
            else:
                old_winch_handler = None
            
            try:
                stdin_eof = False
                while True:
                    # 检查进程是否还在运行
                    if process.poll() is not None:
                        break
                    
                    # 确定要监听的文件描述符
                    read_fds = [master]
                    if not stdin_eof:
                        read_fds.append(sys.stdin.fileno())
                    
                    # 选择可读的文件描述符
                    ready, _, _ = select.select(read_fds, [], [], 0.1)
                    
                    # 处理用户输入
                    if sys.stdin.fileno() in ready and not stdin_eof:
                        try:
                            data = os.read(sys.stdin.fileno(), 1024)
                            if data:
                                os.write(master, data)
                            else:
                                # EOF
                                stdin_eof = True
                        except OSError:
                            stdin_eof = True
                    
                    # 处理程序输出
                    if master in ready:
                        try:
                            data = os.read(master, 1024)
                            if data:
                                if requires_raw_mode:
                                    # 对于需要完全终端控制的程序，直接输出原始数据
                                    sys.stdout.buffer.write(data)
                                    sys.stdout.flush()
                                else:
                                    # 对于普通程序，解码后输出并捕获
                                    decoded_data = data.decode('utf-8', errors='replace')
                                    sys.stdout.write(decoded_data)
                                    sys.stdout.flush()
                                    captured_output += decoded_data
                            else:
                                break
                        except OSError:
                            break
                            
            except KeyboardInterrupt:
                if requires_raw_mode:
                    # 对于全屏程序，将Ctrl+C传递给子进程
                    try:
                        os.write(master, b'\x03')
                    except OSError:
                        pass
                else:
                    # 对于普通程序，显示中断信息并终止
                    self.print_colored("\n^C", Colors.YELLOW)
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=2)
                    except:
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except:
                            pass
                    return 130, captured_output
            
            finally:
                # 恢复终端属性
                if requires_raw_mode and old_tty is not None:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
                
                # 恢复信号处理器
                if requires_raw_mode and old_winch_handler is not None:
                    signal.signal(signal.SIGWINCH, old_winch_handler)
                
                # 关闭master
                try:
                    os.close(master)
                except OSError:
                    pass
                
                # 等待进程结束
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            
            # 对于全屏程序，不返回捕获的输出（因为包含控制字符）
            if requires_raw_mode:
                return process.returncode, ""
            else:
                return process.returncode, captured_output
            
        except Exception as e:
            self.print_colored(f"执行命令时出错: {e}", Colors.RED)
            return 1, f"Error: {e}"
    
    def process_command(self, raw_command: str):
        """处理单个命令"""
        # 去除首尾空白
        command = raw_command.strip()
        original_command = command  # 保存原始命令用于后置处理
        
        # 跳过空命令
        if not command:
            return
        
        # 添加到历史记录
        self.history.append(command)
        
        # 处理内置命令
        if self.builtin_commands(command):
            if command.lower() in ['exit', 'quit']:
                self.running = False
            return
        
        # 前置处理
        if self.pre_processor:
            try:
                processed_command = self.pre_processor(command)
                if processed_command != command:
                    self.print_colored(f"[预处理] {command} -> {processed_command}", Colors.MAGENTA)
                command = processed_command
            except Exception as e:
                self.print_colored(f"前置处理失败: {e}", Colors.RED)
                return
        
        # 检查命令是否以"?"结尾（用于AI解释），如果是则在执行前移除"?"
        execution_command = command
        if command.rstrip().endswith("?"):
            execution_command = command.rstrip()[:-1].rstrip()
        
        # 执行命令
        start_time = datetime.now()
        return_code, command_output = self.execute_command(execution_command)
        end_time = datetime.now()
        
        # 显示执行信息
        duration = (end_time - start_time).total_seconds()
        if return_code == 0:
            status_color = Colors.GREEN
            status = "成功"
        else:
            status_color = Colors.RED
            status = "失败"
        
        print(f"\n{Colors.BOLD}[{status_color}{status}{Colors.END}{Colors.BOLD}] 返回码: {return_code}, 耗时: {duration:.2f}s{Colors.END}")
        
        # 后置处理 - 传递原始命令而不是处理后的命令
        if self.post_processor:
            try:
                self.post_processor(original_command, return_code, command_output)
            except Exception as e:
                self.print_colored(f"后置处理失败: {e}", Colors.RED)
    
    def run(self):
        """启动终端Session主循环"""
        self.running = True
        self.print_header()
        
        try:
            while self.running:
                try:
                    # 显示提示符并获取输入
                    prompt = self.get_prompt()
                    command = input(prompt)
                    
                    # 处理命令
                    self.process_command(command)
                    
                except KeyboardInterrupt:
                    # Ctrl+C 时不退出，只是显示新的提示符
                    print(f"\n{Colors.YELLOW}^C{Colors.END}")
                    continue
                    
                except EOFError:
                    # Ctrl+D 退出
                    print(f"\n{Colors.GREEN}再见！{Colors.END}")
                    break
                    
        except Exception as e:
            self.print_colored(f"终端Session出错: {e}", Colors.RED)
        
        finally:
            self.running = False


# 示例用法和自定义前置处理函数
def smart_pre_processor(command: str, config: Optional[TerminalConfig] = None) -> str:
    """
    智能前置处理示例
    - 自动添加常用参数
    - 命令别名替换
    - 安全检查等
    """
    if config is None:
        # 如果没有传入配置，创建一个临时的配置管理器
        config_manager = ConfigManager()
        config = config_manager.config
    
    # 分割命令
    parts = command.split()
    if not parts:
        return command
        
    cmd = parts[0]
    
    # 应用别名
    if cmd in config.command_aliases:
        parts[0] = config.command_aliases[cmd]
        command = ' '.join(parts)
        # 重新分析命令（因为别名可能改变了命令结构）
        parts = command.split()
        if parts:
            cmd = parts[0]
    
    # 自动添加参数
    if cmd in config.auto_args:
        auto_arg = config.auto_args[cmd]
        if auto_arg not in command:
            command += f' {auto_arg}'
    
    return command


def create_smart_pre_processor(config: TerminalConfig):
    """创建一个绑定配置的前置处理器"""
    def processor(command: str) -> str:
        return smart_pre_processor(command, config)
    return processor


def smart_post_processor(command: str, return_code: int, command_output: str):
    """
    智能后置处理示例
    - 记录失败命令
    - 性能统计
    - 自动建议等
    """
    if return_code != 0:
        # 记录失败的命令
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("failed_commands.log", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] 返回码: {return_code}, 命令: {command}\n")
        
        # 提供建议
        if command.startswith('ls') and return_code == 2:
            print(f"{Colors.YELLOW}提示: 可能是目录不存在，尝试 'ls -la' 查看详细信息{Colors.END}")


def main():
    """主函数 - 演示如何使用TerminalSession"""
    print(f"{Colors.BOLD}启动智能终端Session...{Colors.END}")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 创建终端Session实例
    session = TerminalSession(
        pre_processor=create_smart_pre_processor(config_manager.config),
        post_processor=smart_post_processor
    )
    
    # 启动Session
    session.run()


if __name__ == "__main__":
    main()