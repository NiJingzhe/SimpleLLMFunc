from SimpleLLMFunc import llm_chat, llm_function
from SimpleLLMFunc import ZhipuAI_glm_4_flash_Interface
from SimpleLLMFunc import tool
import os
import sys
from typing import List, Dict

@tool(
    name="calculator", description="A calculator that can perform basic arithmetic operations."
)
def calc(expression: str) -> float:
    """计算器

    Args:
        expression: 一个数学表达式，例如：1+2*3-4/5
    Returns:
        计算结果
    """
    
    return float(eval(expression))

@tool(
    name="get_current_time_and_date", description="获取当前时间和日期"
)
def get_current_time_and_date() -> str:
    """获取当前时间和日期

    Returns:
        当前时间和日期
    """
    
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface
)
def auto_merge(prev_content: str, content_need_merge: str) -> str: # type: ignore
    """自动合并函数
    
    你需要将 content_need_merge 代表的内容和 prev_content 进行智能合并，并返回合并后的新结果。

    Args:
        prev_content: 旧的内容
        content_need_merge: 需要和旧内容和并的新的内容，可能包含关于如何正确合并的信息
    Returns:
        合并后的内容
    """
    pass
    

@tool(
    name="file_operator", description="文件操作工具"
)
def file_operator(file_path: str, operation: str, content: str = "") -> str:
    """文件操作工具

    Args:
        file_path: 文件路径，请直接给出一个相对路径，不要以 /, ./, ../等开头
        operation: 操作类型，例如：read(读取文件), write(写入文件), delete（删除文件）, auto_merge（自动合并内容）, mkdir（创建目录）, tree（列出目录树）, ls（列出当前目录内容）
        content: 写入内容，不要在写入内容中包含任何markdown包裹（仅在操作为write或auto_merge时使用）, auto_merge时请提供关于merge行为的更多信息在content中。
    Returns:
        操作结果
    """
    
    import os
    import shutil
    
    # 安全处理文件路径
    # 如果file path以/开头，去掉
    if file_path.startswith("/"):
        file_path = file_path[1:]
    # 如果file path以./开头，去掉
    if file_path.startswith("./"):
        file_path = file_path[2:] 
    # 如果file path以../开头，去掉
    if file_path.startswith("../"):
        file_path = file_path[3:]
    
    # 去除路径末尾的斜杠，避免误创建为目录
    if file_path.endswith("/") and operation != "mkdir" and operation != "tree" and operation != "ls":
        file_path = file_path.rstrip("/")
        
    # 在sandbox下创建文件
    sandbox_path = os.path.join(os.getcwd(), "sandbox")
    # 确保sandbox目录存在
    os.makedirs(sandbox_path, exist_ok=True)
    
    full_path = os.path.join(sandbox_path, file_path)
    
    # 创建目录操作
    if operation == "mkdir":
        try:
            # 确保是一个有效的目录路径
            if os.path.isfile(full_path):
                return f"错误: 无法创建目录 '{file_path}'，因为同名文件已存在"
            
            os.makedirs(full_path, exist_ok=True)
            return f"创建目录成功: '{file_path}'"
        except Exception as e:
            return f"创建目录失败: {str(e)}"
    
    # 列出目录内容操作
    if operation == "ls":
        try:
            # 获取目录路径
            dir_path = full_path if os.path.isdir(full_path) else os.path.dirname(full_path)
            
            if not os.path.exists(dir_path):
                return f"错误: 目录 '{os.path.relpath(dir_path, sandbox_path)}' 不存在"
            
            # 列出目录内容
            items = os.listdir(dir_path)
            result = []
            for item in items:
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    result.append(f"{item}/")
                else:
                    result.append(item)
            
            relative_path = os.path.relpath(dir_path, sandbox_path)
            return f"目录 '{relative_path if relative_path != '.' else ''}' 的内容:\n" + "\n".join(result)
        except Exception as e:
            return f"列出目录内容失败: {str(e)}"
    
    # 生成目录树操作
    if operation == "tree":
        try:
            # 获取目录路径
            dir_path = full_path if os.path.isdir(full_path) else os.path.dirname(full_path)
            
            if not os.path.exists(dir_path):
                return f"错误: 目录 '{os.path.relpath(dir_path, sandbox_path)}' 不存在"
            
            # 返回递归展开的目录树
            def list_files(startpath):
                result = []
                for root, dirs, files in os.walk(startpath):
                    level = root.replace(startpath, "").count(os.sep)
                    indent = " " * 4 * level
                    result.append(f"{indent}{os.path.basename(root)}/")
                    subindent = " " * 4 * (level + 1)
                    for f in files:
                        result.append(f"{subindent}{f}")
                return result
            
            relative_path = os.path.relpath(dir_path, sandbox_path)
            tree_content = "\n".join(list_files(dir_path))
            return f"目录 '{relative_path if relative_path != '.' else ''}' 的树状结构:\n{tree_content}"
        except Exception as e:
            return f"生成目录树失败: {str(e)}"
    
    # 读取文件操作
    if operation == "read":
        try:
            # 检查路径是否是目录
            if os.path.isdir(full_path):
                return f"错误: '{file_path}' 是一个目录，无法进行读取操作"
            
            # 检查文件是否存在
            if not os.path.exists(full_path):
                return f"错误: 文件 '{file_path}' 不存在"
            
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(">"*50, "\n", f"SYSTEM: 文件: {file_path} 被读取\n", "<"*50)
                return content
        except Exception as e:
            return f"读取文件失败: {str(e)}"
    
    # 删除文件或目录操作
    elif operation == "delete":
        try:
            if not os.path.exists(full_path):
                return f"错误: 文件或目录 '{file_path}' 不存在"
            
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(">"*50, "\n", f"SYSTEM: 目录: {file_path} 及其内容被删除\n", "<"*50)
                return f"删除成功: 目录 '{file_path}' 及其内容已被删除"
            else:
                os.remove(full_path)
                print(">"*50, "\n", f"SYSTEM: 文件: {file_path} 被删除\n", "<"*50)
                return f"删除成功: 文件 '{file_path}' 已被删除"
        except Exception as e:
            return f"删除操作失败: {str(e)}"
    
    # 写入文件操作
    elif operation == "write":
        try:
            # 检查路径是否是目录
            if os.path.isdir(full_path):
                return f"错误: '{file_path}' 是一个目录，无法进行写入操作。请使用不同的文件名或先删除同名目录。"
            
            # 确保父目录存在
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"SYSTEM: 创建父目录: {os.path.relpath(parent_dir, sandbox_path)}")
            
            # 写入文件
            if not os.path.exists(full_path):
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(">"*50, "\n", f"SYSTEM: 文件: {file_path} 被创建并写入\n", "<"*50)
                return f"写入成功: 文件 '{file_path}' 已创建并写入内容，可以使用read操作查看内容"
            else:
                # 文件已存在，执行合并
                with open(full_path, "r", encoding="utf-8") as f:
                    prev_content = f.read()
                
                merged_content = auto_merge(prev_content, content)
                
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(merged_content)
                
                print(">"*50, "\n", f"SYSTEM: 文件: {file_path} 内容被合并\n", "<"*50)
                return f"写入并合并成功: 文件 '{file_path}' 的内容已更新，可以使用read操作查看内容并检查合并是否正确"
        except Exception as e:
            return f"写入文件失败: {str(e)}"
    
    # 自动合并操作 (显式调用)
    elif operation == "auto_merge":
        try:
            # 检查路径是否是目录
            if os.path.isdir(full_path):
                return f"错误: '{file_path}' 是一个目录，无法进行合并操作"
            
            # 检查文件是否存在
            if not os.path.exists(full_path):
                return f"错误: 文件 '{file_path}' 不存在，无法进行合并操作"
            
            with open(full_path, "r", encoding="utf-8") as f:
                prev_content = f.read()
            
            merged_content = auto_merge(prev_content, content)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(merged_content)
            
            print(">"*50, "\n", f"SYSTEM: 文件: {file_path} 内容被合并\n", "<"*50)
            return f"合并成功: 文件 '{file_path}' 的内容已更新，可以使用read操作查看内容并检查合并是否正确"
        except Exception as e:
            return f"合并文件失败: {str(e)}"
    
    else:
        return f"不支持的操作: '{operation}'。支持的操作包括: read, write, delete, auto_merge, mkdir, tree, ls"
    
@tool(
    name="execute_command", description="执行系统命令"
)    
def execute_command(command: str) -> str:
    """执行系统命令

    Args:
        command: 系统命令，例如：ls -l
    Returns:
        命令输出
    """
    
    import subprocess
    
    try:
        print(">"*50, "\n", f"SYSTEM: 执行命令: {command}\n", "<"*50)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(">"*50, "\n", f"SYSTEM: 命令: {command} 执行完成，结果是: {result}\n", "<"*50)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as e:
        return f"执行命令失败: {str(e)}"


@llm_chat(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    toolkit=[calc, get_current_time_and_date, file_operator, execute_command],
)
def chat_bot_1(history: List[Dict[str, str]], query: str) -> str: # type: ignore
    """
    你是Cortana，是一位具有丰富经验的程序员，擅长于使用Python代码实现各种功能，推荐使用file_operator工具来帮助你撰写代码。
    推荐你和GLaDos一起合作，他是你的上司。
    GLaDos是擅长于进行总体规划的游戏设计师。    
    你务必听从GLaDos给你的任务。
    
    - calculator: 计算器
    
    - get_current_time_and_date: 获取当前时间和日期
    
    - file_operator: 文件操作工具，不要只说操作文件，要实际上调用工具
    
    - execute_command: 执行系统命令
    """
    pass


@llm_chat(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    toolkit=[calc, get_current_time_and_date, file_operator, execute_command],
)
def chat_bot_2(history: List[Dict[str, str]], query: str) -> str:  # type: ignore
    """
    你是GLaDos，一位擅长于进行总体规划的游戏设计师，推荐你和Cortana一起合作，她是你的下属。同时推荐你使用file_operator工具记录你的设计和想法
    Cortana是具有丰富经验的程序员。你可以命令她实现代码，写完后一定要读取检查一下。
    
    - calculator: 计算器
    
    - get_current_time_and_date: 获取当前时间和日期
    
    - file_operator: 文件操作工具，不要只说操作文件，要实际上调用工具
    
    - execute_command: 执行系统命令
    """
    pass


if __name__ == "__main__":
    # 创建空的历史记录列表
    history_cb1 = []
    history_cb2 = []
    
    # 获取用户输入作为初始化对话的内容
    user_input = input("请输入您的消息以开始对话: ")
    
    # 使用用户输入初始化对话，由GLaDos先发起
    initial_message = f"用户说: {user_input}" if user_input.strip() else "你要和Cortana一起合作，制作一款在终端中运行的文字RPG游戏."
    
    print("=========================================="*3)
    print(f"用户: {initial_message}")

    # 初始消息添加到历史中
    history_cb1.append({"role": "user", "content": initial_message})
    
    # 调用chat_bot_1，获取返回的内容和更新的历史
    response_cb1, history_cb1 = chat_bot_1(history_cb1, initial_message)
    # 只截取"回答："后面的内容
    message_from_cb1 = response_cb1.split("回答：")[-1].strip()
    print("=========================================="*3)
    print(f"GLaDos: {message_from_cb1}")
    
    # 将GLaDos的回复添加到另一个聊天机器人的历史中
    history_cb2.append({"role": "user", "content": message_from_cb1})
    
    # 调用chat_bot_2，获取返回的内容和更新的历史
    response_cb2, history_cb2 = chat_bot_2(history_cb2, message_from_cb1)
    # 只截取"回答："后面的内容
    message_from_cb2 = response_cb2.split("回答：")[-1].strip()
    print("=========================================="*3)
    print(f"小娜: {message_from_cb2}")
    
    # 简单的双Agent循环
    try:
        while True:
            # 将小娜的回复添加到GLaDos的历史中
            history_cb1.append({"role": "user", "content": message_from_cb2})
            
            # 调用chat_bot_1，获取返回的内容和更新的历史
            response_cb1, history_cb1 = chat_bot_1(history_cb1, message_from_cb2)
            # 只截取"回答："后面的内容
            message_from_cb1 = response_cb1.split("回答：")[-1].strip()
            print("=========================================="*3)
            print(f"GLaDos: {message_from_cb1}")
            
            # 将GLaDos的回复添加到小娜的历史中
            history_cb2.append({"role": "user", "content": message_from_cb1})
            
            # 调用chat_bot_2，获取返回的内容和更新的历史
            response_cb2, history_cb2 = chat_bot_2(history_cb2, message_from_cb1)
            # 只截取"回答："后面的内容
            message_from_cb2 = response_cb2.split("回答：")[-1].strip()
            print("=========================================="*3)
            print(f"小娜: {message_from_cb2}")
    except KeyboardInterrupt:
        print("\n检测到键盘中断，程序已退出")