#!/usr/bin/env python3
"""
动态模板参数演示

展示如何使用_template_params在函数调用时动态设置DocString模板参数。
一个函数定义，多种使用场景。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from SimpleLLMFunc import llm_function, app_log
from SimpleLLMFunc import OpenAICompatible

# 加载LLM接口配置
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")

try:
    llm_interface = OpenAICompatible.load_from_json_file(provider_json_path)["dreamcatcher"]["gpt-4o"]
    print("✅ 成功加载LLM接口配置")
except (FileNotFoundError, KeyError) as e:
    print(f"⚠️  警告: 无法加载LLM接口配置 ({e})")
    print("请确保provider.json文件存在且配置正确")
    llm_interface = None  # type: ignore

# 万能的代码分析函数
@llm_function(llm_interface=llm_interface)  # type: ignore
def analyze_code(code: str) -> str:
    """以{style}的方式分析{language}代码，重点关注{focus}。"""
    return ""

# 万能的文本处理函数  
@llm_function(llm_interface=llm_interface)  # type: ignore
def process_text(text: str) -> str:
    """作为{role}，请{action}以下文本，输出风格为{style}。"""
    return ""


def main():
    """主函数演示"""
    if llm_interface is None:
        print("由于缺少LLM接口配置，仅展示函数定义。")
        print("请参考examples/provider_template.json创建provider.json配置文件。")
        return
    
    app_log("开始运行动态模板参数演示")
    
    print("=== 动态模板参数演示 ===\n")
    
    # 示例1: 代码分析
    print("1. 代码分析功能演示:")
    
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    try:
        print("   Python性能分析:")
        result1 = analyze_code(
            python_code,
            _template_params={
                'style': '详细', 
                'language': 'Python', 
                'focus': '性能优化'
            }
        )
        print(f"   分析结果: {result1}\n")
    except Exception as e:
        print(f"   执行失败: {e}\n")
    
    try:
        print("   JavaScript规范检查:")
        js_code = "function test() { console.log('hello'); }"
        result2 = analyze_code(
            js_code,
            _template_params={
                'style': '简洁', 
                'language': 'JavaScript', 
                'focus': '代码规范'
            }
        )
        print(f"   分析结果: {result2}\n")
    except Exception as e:
        print(f"   执行失败: {e}\n")
    
    # 示例2: 文本处理
    print("2. 文本处理功能演示:")
    
    sample_text = "人工智能技术正在快速发展，对各行各业产生深远影响。"
    
    try:
        print("   编辑润色:")
        result3 = process_text(
            sample_text,
            _template_params={
                'role': '专业编辑', 
                'action': '润色', 
                'style': '学术'
            }
        )
        print(f"   处理结果: {result3}\n")
    except Exception as e:
        print(f"   执行失败: {e}\n")
    
    try:
        print("   翻译转换:")
        result4 = process_text(
            sample_text,
            _template_params={
                'role': '翻译专家', 
                'action': '翻译成英文', 
                'style': '商务'
            }
        )
        print(f"   处理结果: {result4}\n")
    except Exception as e:
        print(f"   执行失败: {e}\n")
    
    print("✨ 核心优势：")
    print("• 一个函数定义，多种使用场景")
    print("• 调用时动态指定角色和任务")
    print("• 代码复用性大大提高")
    print("• 更符合实际使用需求")
    print()
    print("💡 使用提示：")
    print("• 在DocString中使用{变量名}作为占位符")
    print("• 调用时通过_template_params传入变量值")
    print("• _template_params不会传递给LLM，仅用于模板处理")
    
    app_log("动态模板参数演示运行结束")


if __name__ == "__main__":
    main()
