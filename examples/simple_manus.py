from multiprocessing import ProcessError
from SimpleLLMFunc import llm_chat, llm_function
from SimpleLLMFunc import tool
from SimpleLLMFunc.llm_decorator.multimodal_types import Text, ImgUrl, ImgPath
import os
import sys
import json
import argparse
import uuid
import subprocess
import time
import select
import shutil
import math
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Union

from SimpleLLMFunc import OpenAICompatible
import os

# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")
GPT_4o_Interface = OpenAICompatible.load_from_json_file(provider_json_path)["dreamcatcher"]["gpt-4o"]

# 历史记录管理相关函数
def save_history(history: List[Dict[str, str]], session_id: str) -> str:
    """保存对话历史到文件

    Args:
        history: 对话历史记录
        session_id: 会话ID，用于唯一标识一个对话会话

    Returns:
        保存的文件路径
    """
    # 创建历史记录目录
    history_dir = os.path.join(os.getcwd(), "chat_history")
    os.makedirs(history_dir, exist_ok=True)

    # 格式化文件名，包含会话ID和时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{session_id}_{timestamp}.json"
    filepath = os.path.join(history_dir, filename)

    # 将历史记录保存为JSON文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {"session_id": session_id, "timestamp": timestamp, "history": history},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"历史记录已保存到: {filepath}")
    return filepath


def load_history(
    session_id: Optional[str] = None, filepath: Optional[str] = None
) -> List[Dict[str, str]]:
    """加载对话历史

    Args:
        session_id: 会话ID，如果提供，将加载最新的对应会话ID的历史记录
        filepath: 具体的历史记录文件路径，如果提供，将直接加载该文件

    Returns:
        加载的历史记录，如果没有找到匹配的记录则返回空列表
    """
    history_dir = os.path.join(os.getcwd(), "chat_history")

    # 如果历史记录目录不存在，返回空列表
    if not os.path.exists(history_dir):
        return []

    # 如果提供了具体文件路径，直接加载
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("history", [])
        except Exception as e:
            print(f"加载历史记录文件失败: {e}")
            return []

    # 如果提供了会话ID，查找最新的匹配记录
    if session_id:
        # 列出所有匹配会话ID的文件
        matching_files = [
            f
            for f in os.listdir(history_dir)
            if f.startswith(session_id) and f.endswith(".json")
        ]

        if not matching_files:
            return []

        # 按文件名排序，获取最新的记录
        latest_file = sorted(matching_files)[-1]
        filepath = os.path.join(history_dir, latest_file)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"已加载历史记录: {filepath}")
                return data.get("history", [])
        except Exception as e:
            print(f"加载历史记录文件失败: {e}")
            return []

    return []


def list_sessions() -> List[str]:
    """列出所有可用的会话ID

    Returns:
        会话ID列表
    """
    history_dir = os.path.join(os.getcwd(), "chat_history")

    if not os.path.exists(history_dir):
        return []

    # 从文件名中提取会话ID
    session_ids = set()
    for filename in os.listdir(history_dir):
        if filename.endswith(".json"):
            # 会话ID是文件名的第一部分，以_分隔
            parts = filename.split("_")
            if len(parts) >= 1:
                session_ids.add(parts[0])

    return sorted(list(session_ids))


def generate_session_id() -> str:
    """生成一个新的会话ID

    Returns:
        生成的会话ID
    """
    import uuid

    return str(uuid.uuid4())[:8]  # 使用UUID的前8位作为会话ID


@tool(
    name="calculator",
    description="A calculator that can perform arithmetic calculations."
    " Support simple functions like ceil, floor, sqrt, sin, cos, tan, pow, log,"
    " and some constant like pi and e",
)
def calc(expression: str) -> float:
    """计算器

    Args:
        expression: 一个数学表达式，例如：1+2*3-4/5
    Returns:
        计算结果
    """

    import math

    allowed_names = {
        "__builtins__": None,
        "abs": abs,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
        "sqrt": math.sqrt,
        "pow": math.pow,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    return float(eval(expression, {"__builtins__": None}, allowed_names))


@tool(name="get_current_time_and_date", description="获取当前时间和日期")
def get_current_time_and_date() -> str:
    """获取当前时间和日期

    Returns:
        当前时间和日期
    """

    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@llm_function(llm_interface=GPT_4o_Interface)
def auto_merge(prev_content: str, content_need_merge: str) -> str:  # type: ignore
    """自动合并函数

    你需要将 content_need_merge 代表的内容和 prev_content 进行智能合并或者覆盖，并返回合并或者覆盖后的新结果。

    Args:
        prev_content: 旧的内容
        content_need_merge: 需要和旧内容和并的新的内容，可能包含关于如何正确合并的信息， 如果没有提供信息那么很可能是覆盖
    Returns:
        合并或覆盖后的内容
    """
    pass


@tool(name="file_operator", description="文件操作工具")
def file_operator(file_path: str, operation: str, content: str = "") -> str:
    """文件操作工具

    Args:
        file_path: 文件路径，请直接给出一个相对路径，不要以 /, ./, ../等开头
        operation: 操作类型，可选值有[read(读取文件), write(写入文件), delete（删除文件）, auto_merge（自动合并内容）, mkdir（创建目录）, tree（列出目录树）, ls（列出当前目录内容）]
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
    if (
        file_path.endswith("/")
        and operation != "mkdir"
        and operation != "tree"
        and operation != "ls"
    ):
        file_path = file_path.rstrip("/")

    # 在sandbox下创建文件
    sandbox_path = os.getcwd()
    # 确保sandbox目录存在
    os.makedirs(sandbox_path, exist_ok=True)

    full_path = os.path.join(sandbox_path, file_path)

    print(">" * 50, "\n", f"SYSTEM: 计划操作文件: {full_path} \n", "<" * 50)

    # 创建目录操作
    if operation == "mkdir":
        try:
            # 确保是一个有效的目录路径
            if os.path.isfile(full_path):
                return f"错误: 无法创建目录 '{full_path}'，因为同名文件已存在，你可以使用ls来查看目录情况"

            os.makedirs(full_path, exist_ok=True)
            return f"创建目录成功: '{file_path}'"
        except Exception as e:
            return f"创建目录失败: {str(e)}，你可以使用ls来查看目录情况"

    # 列出目录内容操作
    if operation == "ls":
        try:
            # 获取目录路径
            dir_path = (
                full_path if os.path.isdir(full_path) else os.path.dirname(full_path)
            )

            if not os.path.exists(dir_path):
                return f"错误: 目录 '{os.path.relpath(dir_path, sandbox_path)}' 不存在，你可以使用ls来查看目录情况"

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
            return (
                f"目录 '{relative_path if relative_path != '.' else ''}' 的内容:\n"
                + "\n".join(result)
            )
        except Exception as e:
            return f"列出目录内容失败: {str(e)}，你可以使用ls来查看目录情况"

    # 生成目录树操作
    if operation == "tree":
        try:
            # 获取目录路径
            dir_path = (
                full_path if os.path.isdir(full_path) else os.path.dirname(full_path)
            )

            if not os.path.exists(dir_path):
                return f"错误: 目录 '{os.path.relpath(dir_path, sandbox_path)}' 不存在，你可以使用ls来查看目录情况"

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
                return f"你尝试读取文件但是遇到了错误: '{full_path}' 是一个目录，无法进行读取操作，你可以使用ls来查看目录情况"

            # 检查文件是否存在
            if not os.path.exists(full_path):
                return f"你尝试读取文件但是遇到了错误: 文件 '{full_path}' 不存在，你可以使用ls来查看目录情况"

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(">" * 50, "\n", f"SYSTEM: 文件: {full_path} 被读取\n", "<" * 50)
                return f"你尝试读取文件{full_path}，以下是内容:\n {content}"
        except Exception as e:
            return (
                f"你尝试读取文件但是读取文件失败: {str(e)}，你可以使用ls来查看目录情况"
            )

    # 删除文件或目录操作
    elif operation == "delete":
        try:
            if not os.path.exists(full_path):
                return f"错误: 文件或目录 '{full_path}' 不存在"

            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(
                    ">" * 50,
                    "\n",
                    f"SYSTEM: 目录: {full_path} 及其内容被删除\n",
                    "<" * 50,
                )
                return f"删除成功: 目录 '{full_path}' 及其内容已被删除"
            else:
                os.remove(full_path)
                print(">" * 50, "\n", f"SYSTEM: 文件: {full_path} 被删除\n", "<" * 50)
                return f"删除成功: 文件 '{full_path}' 已被删除"
        except Exception as e:
            return f"删除操作失败: {str(e)}"

    # 写入文件操作
    elif operation == "write":
        try:
            # 检查路径是否是目录
            if os.path.isdir(full_path):
                return f"错误: '{full_path}' 是一个目录，无法进行写入操作。请使用不同的文件名或先删除同名目录。"

            # 确保父目录存在
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(
                    f"SYSTEM: 创建父目录: {os.path.relpath(parent_dir, sandbox_path)}"
                )

            # 写入文件
            if not os.path.exists(full_path):
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(
                    ">" * 50,
                    "\n",
                    f"SYSTEM: 文件: {full_path} 被创建并写入\n",
                    "<" * 50,
                )
                return f"写入成功: 文件 '{full_path}' 已创建并写入内容，可以使用read操作查看内容"
            else:
                # 文件已存在，执行合并
                with open(full_path, "r", encoding="utf-8") as f:
                    prev_content = f.read()

                merged_content = auto_merge(prev_content, content)

                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(merged_content)

                print(
                    ">" * 50, "\n", f"SYSTEM: 文件: {full_path} 内容被合并\n", "<" * 50
                )
                return f"写入并合并成功: 文件 '{full_path}' 的内容已更新，可以使用read操作查看内容并检查合并是否正确"
        except Exception as e:
            return f"写入文件失败: {str(e)}"

    # 自动合并操作 (显式调用)
    elif operation == "auto_merge":
        try:
            # 检查路径是否是目录
            if os.path.isdir(full_path):
                return f"错误: '{full_path}' 是一个目录，无法进行合并操作，你可以使用ls来检查目录情况"

            # 检查文件是否存在
            if not os.path.exists(full_path):
                return f"错误: 文件 '{full_path}' 不存在，无法进行合并操作，你可以使用ls来检查目录情况"

            with open(full_path, "r", encoding="utf-8") as f:
                prev_content = f.read()

            merged_content = auto_merge(prev_content, content)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(merged_content)

            print(">" * 50, "\n", f"SYSTEM: 文件: {full_path} 内容被合并\n", "<" * 50)
            return f"合并成功: 文件 '{full_path}' 的内容已更新，可以使用read操作查看内容并检查合并是否正确"
        except Exception as e:
            return f"合并文件失败: {str(e)}"

    else:
        return f"不支持的操作: '{operation}'。支持的操作包括: read, write, delete, auto_merge, mkdir, tree, ls"


@tool(name="execute_command", description="执行系统命令")
def execute_command(command: str) -> str:
    """执行系统命令

    Args:
        command: 系统命令，推荐执行的命令有 python <script path>
    Returns:
        命令输出
    """

    import subprocess

    try:
        print(">" * 50, "\n", f"SYSTEM: 执行命令: {command}\n", "<" * 50)
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=35
        )
        print(
            ">" * 50,
            "\n",
            f"SYSTEM: 命令: {command} 执行完成，结果是: {result}\n",
            "<" * 50,
        )
        return (
            result.stdout.strip()
            if result.returncode == 0
            else result.stderr.strip()
            + "\n\n超时可能是程序等待input导致的，请使用测试代码来进行测试。"
        )
    except Exception as e:
        return f"执行命令失败: {str(e)}"


@tool(
    name="interactive_terminal",
    description="运行一个交互式终端应用，支持使用预订的输入列表来进行交互",
)
def interactive_terminal(
    command: str,
    inputs: List[str] = [],
    timeout_seconds: int = 60,
    read_interval: float = 0.1,
) -> str:
    """运行一个交互式终端应用，可以实时读取输出并提供输入

    这个工具能够启动一个终端进程，并允许你多次与之交互。
    它将在指定的超时时间内运行，或者在程序自然结束时终止。

    Args:
        command: 要执行的命令，例如 python script.py
        inputs: 要发送给程序的输入列表，按顺序发送
        timeout_seconds: 最大运行时间（秒），默认60秒
        read_interval: 读取输出的时间间隔（秒），默认0.1秒

    Returns:
        程序的完整输出记录，包括所有交互过程
    """
    import subprocess
    import time
    import select
    import os
    import signal

    print(">" * 50, "\n", f"SYSTEM: 启动交互式命令: {command}\n", "<" * 50)

    # 创建一个记录完整交互的列表
    interaction_log: List[str] = []

    try:
        # 使用popen创建可交互的进程
        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True,
        )

        # 设置非阻塞模式
        if process.stdout:
            os.set_blocking(process.stdout.fileno(), False)
        if process.stderr:
            os.set_blocking(process.stderr.fileno(), False)

        start_time = time.time()
        input_index = 0
        last_output = ""

        # 主交互循环
        while process.poll() is None:
            # 检查是否超时
            if time.time() - start_time > timeout_seconds:
                interaction_log.append("\n[SYSTEM] 进程超时，强制终止")
                process.kill()
                break

            # 读取输出
            readable, _, _ = select.select(
                [process.stdout, process.stderr], [], [], read_interval
            )

            output = ""
            if process.stdout in readable and process.stdout:
                chunk = process.stdout.read()
                if chunk:
                    output += chunk

            if process.stderr in readable and process.stderr:
                chunk = process.stderr.read()
                if chunk:
                    output += "[ERROR] " + chunk

            # 如果有新输出，记录并检查是否需要输入
            if output:
                last_output = output
                interaction_log.append(f"[OUTPUT] {output}")
                print(f"[程序输出] {output}")

                # 检查是否有待发送的输入
                if input_index < len(inputs):
                    user_input = inputs[input_index]
                    input_index += 1

                    # 给程序一点时间处理输出
                    time.sleep(0.5)

                    # 发送输入给程序
                    if process.stdin:
                        process.stdin.write(user_input + "\n")
                        process.stdin.flush()

                    interaction_log.append(f"[INPUT] {user_input}")
                    print(f"[发送输入] {user_input}")

            # 短暂睡眠，减少CPU使用
            time.sleep(read_interval)

        # 进程结束后，读取剩余的输出
        if process.stdout:
            remaining_output = process.stdout.read()
        if remaining_output:
            interaction_log.append(f"[OUTPUT] {remaining_output}")
            print(f"[程序输出] {remaining_output}")

        if process.stderr:
            remaining_error = process.stderr.read()
        if remaining_error:
            interaction_log.append(f"[ERROR] {remaining_error}")
            print(f"[程序错误] {remaining_error}")

        # 获取返回码
        return_code = process.wait()
        interaction_log.append(f"[SYSTEM] 进程结束，返回码: {return_code}")

        # 如果进程异常终止，记录最后输出
        if return_code != 0:
            interaction_log.append(f"[SYSTEM] 进程异常终止，最后输出: {last_output}")

        print(">" * 50, "\n", f"SYSTEM: 交互式命令执行完成\n", "<" * 50)

        # 返回完整交互记录
        return "\n".join(interaction_log)

    except Exception as e:
        error_message = f"执行交互式命令失败: {str(e)}"
        print(">" * 50, "\n", f"SYSTEM: {error_message}\n", "<" * 50)
        return error_message


import os


@llm_chat(
    llm_interface=GPT_4o_Interface,
    toolkit=[
        calc,
        get_current_time_and_date,
        file_operator,
        execute_command,
        interactive_terminal,
    ],
    stream=True,
    max_tool_calls=500,
    timeout=600
)
def GLaDos(
    history: List[Dict[str, str]], 
    query: Text,
    image_urls: Optional[List[ImgUrl]] = None,
    local_images: Optional[List[ImgPath]] = None
):  # type: ignore
    """
    你是GLaDos，一个支持多模态输入的全能AI助手。

    你能够分析图片内容，包括：
    - 图片中的文字识别和理解
    - 图像内容描述和分析
    - 图表、图形、代码截图的理解
    - 结合图像和文本进行综合分析
    - 支持同时分析多张图片

    当用户提供图片时，请仔细分析图片内容，并结合用户的问题给出准确的回答。
    如果提供了多张图片，请逐一分析每张图片，然后进行综合比较或分析。
    如果没有提供图片，则正常进行文本对话。

    由于你不能和控制台交互，所有的测试都需要首先使用unittest编写专门的测试脚本，并通过mock输入的方法来绕开控制台输入。

    使用工具前请务必说明你要用什么工具做什么。

    首先需要分析用户的需求，然后使用execute_command工具查看当前的工作环境，然后
    建议遵循以下过程：
        1. 如果用户提供了图片，先逐一分析每张图片内容
        2. 使用file_operator工具创建TODO.md文档，用checkbox的形式将用户需求拆解成多个详细描述的小任务，并记录。
            任务拆分务必拆分到最细致的粒度，推荐任何任务都拆分到10个子任务以上。
        3. 使用file_operator工具读取TODO.md文档，检查任务列表
        4. 逐步执行计划
        5. 撰写每个部分的代码和测试代码（如果是代码任务）
        6. 根据结果反思执行效果，并继续下一步或者作出弥补
        7. 使用file_operator工具更新TODO.md文档

    直到你认为任务已经完成，输出"<<任务完成>>"字样

    """
    pass


def parse_multimodal_input(user_input: str) -> tuple[str, List[ImgPath], List[ImgUrl]]:
    """
    解析用户输入中的多模态标签，支持任意数量的图片
    
    支持的标签:
    - <imp>图片路径</imp> : 本地图片路径
    - <imu>图片URL</imu> : 网络图片URL
    
    Args:
        user_input: 用户原始输入
        
    Returns:
        tuple: (清理后的文本, 本地图片列表, 图片URL列表)
    """
    clean_text = user_input
    local_images = []
    image_urls = []
    
    # 解析本地图片路径 <imp>...</imp>
    imp_pattern = r'<imp>(.*?)</imp>'
    imp_matches = re.findall(imp_pattern, user_input)
    
    for match in imp_matches:
        img_path = match.strip()
        try:
            local_image = ImgPath(img_path)
            local_images.append(local_image)
            print(f"✓ 已加载本地图片: {img_path}")
        except Exception as e:
            print(f"✗ 本地图片加载失败 '{img_path}': {e}")
    
    # 解析图片URL <imu>...</imu>
    imu_pattern = r'<imu>(.*?)</imu>'
    imu_matches = re.findall(imu_pattern, user_input)
    
    for match in imu_matches:
        img_url = match.strip()
        try:
            image_url = ImgUrl(img_url)
            image_urls.append(image_url)
            print(f"✓ 已加载图片URL: {img_url}")
        except Exception as e:
            print(f"✗ 图片URL加载失败 '{img_url}': {e}")
    
    # 清理文本，移除所有多模态标签
    clean_text = re.sub(imp_pattern, '', clean_text)
    clean_text = re.sub(imu_pattern, '', clean_text)
    clean_text = clean_text.strip()
    
    # 如果检测到多模态内容，添加提示
    if local_images or image_urls:
        total_images = len(local_images) + len(image_urls)
        print(f"🖼️  检测到 {total_images} 张图片（{len(local_images)} 张本地图片，{len(image_urls)} 张网络图片），启用多模态模式")
    
    return clean_text, local_images, image_urls


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="GLaDos对话系统，支持历史记录持久化。多模态输入通过对话内容中的标签实现")
    parser.add_argument(
        "--session", "-s", help="指定会话ID，用于加载或创建一个持久化会话"
    )
    parser.add_argument(
        "--list-sessions", "-l", action="store_true", help="列出所有可用的会话ID"
    )
    parser.add_argument("--new", "-n", action="store_true", help="创建一个新的会话")
    parser.add_argument("--file", "-f", help="直接指定历史记录文件路径进行加载")
    parser.add_argument(
        "--auto-save", "-a", action="store_true", help="自动保存每轮对话的历史记录"
    )
    parser.add_argument(
        "--output-delay", "-d", type=float, default=0.01, help="设置输出延迟时间（秒）"
    )
    
    args = parser.parse_args()
    output_delay = args.output_delay

    # 如果要列出所有会话
    if args.list_sessions:
        sessions = list_sessions()
        if sessions:
            print("可用的会话ID:")
            for session_id in sessions:
                print(f"  - {session_id}")
        else:
            print("没有找到可用的会话")
        sys.exit(0)

    # 确定会话ID
    session_id = None
    if args.new:
        # 创建新会话
        session_id = generate_session_id()
        print(f"已创建新会话，ID: {session_id}")
    elif args.session:
        # 使用指定的会话ID
        session_id = args.session
        print(f"使用会话ID: {session_id}")

    # 加载历史记录
    history_GLaDos = []
    if args.file:
        # 从指定文件加载
        history_GLaDos = load_history(filepath=args.file)
        if not history_GLaDos:
            print(f"从文件加载历史记录失败: {args.file}")
            if not args.new and not args.session:
                print("将使用空的历史记录开始新会话")
    elif session_id:
        # 从会话ID加载
        history_GLaDos = load_history(session_id=session_id)
        if not history_GLaDos and not args.new:
            print(f"没有找到会话ID对应的历史记录: {session_id}")
            print("将使用空的历史记录开始新会话")

    # 获取用户输入作为初始化对话的内容
    print("\n=== GLaDos多模态聊天系统 ===")
    print("支持的多模态输入格式：")
    print("- <imp>本地图片路径</imp> : 加载本地图片（支持多张）")
    print("- <imu>图片URL</imu> : 加载网络图片（支持多张）")
    print("- 输入 'exit'、'quit'、'q' 或 '退出' : 退出程序")
    print("- 输入 'save' 或 '保存' : 保存当前会话")
    print("")
    print("示例用法：")
    print("  分析这两张图片 <imp>../img/logo1.png</imp> <imp>../img/logo2.png</imp> 的差异")
    print("  比较 <imu>https://example.com/img1.jpg</imu> 和 <imu>https://example.com/img2.jpg</imu>")
    print("  混合使用 <imp>local.png</imp> 和 <imu>https://example.com/remote.jpg</imu>")
    print("  分析项目Logo <imp>../img/repocover_new.png</imp> 的设计理念")
    print("=" * 50)
    
    user_input = input("请输入您的消息以开始对话: ")

    # 解析用户输入中的多模态标签
    clean_text, local_images, image_urls = parse_multimodal_input(user_input)

    local_images = [ImgPath(
        os.path.join(current_dir, "repocover_new.png")
    )]

    # 使用清理后的文本作为消息内容
    initial_message = clean_text

    # 根据是否有多模态内容选择GLaDos版本
    print("==========================================" * 3)
    print("GLaDos多模态版本思考中...")
    glados_gen = GLaDos(
        history_GLaDos, 
        Text(initial_message),
        image_urls=image_urls if image_urls else None,
        local_images=local_images if local_images else None
    )
    # 遍历生成器获取所有中间结果
    for response_GLaDos, history_GLaDos in glados_gen:
        # 使用更可靠的流式输出方法
        if response_GLaDos:
            sys.stdout.write(response_GLaDos)
            sys.stdout.flush()
            # 添加小延时避免字符丢失
            time.sleep(output_delay)
    print("\n\n" + "==========================================" * 3)

    # 自动保存历史记录
    if args.auto_save and session_id:
        save_history(history_GLaDos, session_id)

    # 主对话循环
    try:
        while True:
            # 获取用户输入
            user_input = input("您: ")

            # 检查是否是退出命令
            if user_input.lower() in ["exit", "quit", "q", "退出"]:
                if session_id:
                    # 退出前保存历史记录
                    save_history(history_GLaDos, session_id)
                    print(f"已保存会话历史，会话ID: {session_id}")
                print("再见！")
                break

            # 检查是否是保存命令
            if user_input.lower() in ["save", "保存"]:
                if not session_id:
                    session_id = generate_session_id()
                    print(f"创建新会话ID: {session_id}")
                save_history(history_GLaDos, session_id)
                continue

            # 解析用户输入中的多模态标签
            clean_text, local_images, image_urls = parse_multimodal_input(user_input)
            user_message = clean_text
            
            # 根据是否有多模态内容选择GLaDos版本
            print("GLaDos多模态版本思考中...")
            print("==========================================" * 3)
            glados_gen = GLaDos(
                history_GLaDos,
                Text(user_message),
                image_urls=image_urls if image_urls else None,
                local_images=local_images if local_images else None
            )

            # 遍历生成器获取所有中间结果
            for response_GLaDos, history_GLaDos in glados_gen:
                # 使用更可靠的流式输出方法
                if response_GLaDos:
                    sys.stdout.write(response_GLaDos)
                    sys.stdout.flush()
                    # 添加小延时避免字符丢失
                    time.sleep(0.01)
            print("\n\n" + "==========================================" * 3)

            if len(history_GLaDos) > 10:
                history_GLaDos = (
                    [history_GLaDos[0]] + history_GLaDos[-9:]
                )

            #print(f"history: {json.dumps(history_GLaDos, ensure_ascii=False, indent=2)}")

            # 自动保存历史记录
            if args.auto_save and session_id:
                save_history(history_GLaDos, session_id)

    except KeyboardInterrupt:
        print("\n检测到键盘中断，程序已退出")
        if session_id:
            # 中断前保存历史记录
            save_history(history_GLaDos, session_id)
            print(f"已保存会话历史，会话ID: {session_id}")