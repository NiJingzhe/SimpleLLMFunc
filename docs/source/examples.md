# 示例代码

本章节提供了 SimpleLLMFunc 的实际使用示例，帮助你更好地理解框架的功能和用法。

## 基础示例

### 动态模板参数示例

这个示例演示了如何使用 `_template_params` 让同一个函数适应不同的使用场景：

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc import OpenAICompatible

# 创建LLM接口
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["provider_name"]["model_name"]

# 万能的代码分析函数
@llm_function(llm_interface=llm_interface)
def analyze_code(code: str) -> str:
    """以{style}的方式分析{language}代码，重点关注{focus}。"""
    pass

# 万能的文本处理函数
@llm_function(llm_interface=llm_interface)
def process_text(text: str) -> str:
    """作为{role}，请{action}以下文本，输出风格为{style}。"""
    pass

# 使用示例
if __name__ == "__main__":
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # 不同的分析方式
    performance_result = analyze_code(
        python_code,
        _template_params={
            'style': '详细',
            'language': 'Python',
            'focus': '性能优化'
        }
    )
    
    style_result = analyze_code(
        python_code,
        _template_params={
            'style': '简洁',
            'language': 'Python',
            'focus': '代码规范'
        }
    )
    
    # 不同的文本处理角色
    sample_text = "人工智能技术正在快速发展，对各行各业产生深远影响。"
    
    edited_result = process_text(
        sample_text,
        _template_params={
            'role': '专业编辑',
            'action': '润色',
            'style': '学术'
        }
    )
    
    translated_result = process_text(
        sample_text,
        _template_params={
            'role': '翻译专家',
            'action': '翻译成英文',
            'style': '商务'
        }
    )
    
    print("性能分析:", performance_result)
    print("代码规范:", style_result)
    print("编辑润色:", edited_result)
    print("翻译结果:", translated_result)
```

这个示例展示了动态模板参数的核心优势：
- **一个函数定义，多种使用场景**
- **调用时动态指定角色和任务**
- **代码复用性大大提高**

### 产品评论分析

这个示例演示了如何使用 `@llm_function` 来创建一个产品评论分析功能。

```python
from typing import Dict, List
from pydantic import BaseModel, Field

from SimpleLLMFunc import llm_function
from SimpleLLMFunc import OpenAICompatible

# 创建LLM接口
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["provider_name"]["model_name"]

# 定义返回类型模型
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")

# 创建LLM函数
@llm_function(llm_interface=llm_interface)
def analyze_product_review(product_name: str, review_text: str) -> ProductReview:
    """
    分析产品评论，提取关键信息并生成结构化评测报告

    Args:
        product_name: 产品名称
        review_text: 用户评论文本

    Returns:
        包含评分、优缺点和总结的产品评测报告
    """
    pass

# 使用函数
if __name__ == "__main__":
    product_name = "XYZ无线耳机"
    review_text = """
    我买了这款XYZ无线耳机已经使用了一个月。音质非常不错，尤其是低音部分表现出色，
    佩戴也很舒适，可以长时间使用不感到疲劳。电池续航能力也很强，充满电后可以使用约8小时。
    不过连接偶尔会有些不稳定，有时候会突然断开。另外，触控操作不够灵敏，经常需要点击多次才能响应。
    总的来说，这款耳机性价比很高，适合日常使用，但如果你需要用于专业音频工作可能还不够。
    """
    
    result = analyze_product_review(product_name, review_text)
    
    print(f"产品评分: {result.rating}/5")
    print(f"产品优点:")
    for pro in result.pros:
        print(f"  - {pro}")
    print(f"产品缺点:")
    for con in result.cons:
        print(f"  - {con}")
    print(f"总结: {result.summary}")
```

### 使用工具获取天气信息并给出建议

这个示例演示了如何定义和使用工具，以及如何将工具集成到 LLM 函数中。

```python
from typing import Dict
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, tool

# 定义工具函数
@tool(name="get_weather", description="获取指定城市的天气信息")
def get_weather(city: str) -> Dict[str, str]:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        包含温度、湿度和天气状况的字典
    """
    # 这里是模拟数据，实际应用中可以调用真实的天气API
    return {"temperature": "32°C", "humidity": "80%", "condition": "Raining"}

# 定义结构化输出模型
class WeatherInfo(BaseModel):
    city: str = Field(..., description="城市名称")
    temperature: str = Field(..., description="当前温度")
    humidity: str = Field(..., description="当前湿度")
    condition: str = Field(..., description="天气状况")
    recommendation: str = Field(..., description="推荐的活动")

# 创建使用工具的LLM函数
@llm_function(llm_interface=llm_interface, toolkit=[get_weather])
def get_daily_recommendation(city: str) -> WeatherInfo:
    """
    通过get_weather工具获取天气信息，并给出推荐的活动

    Args:
        city: 城市名称

    Returns:
        WeatherInfo对象，包含温度、湿度、天气状况和活动建议
    """
    pass

# 使用函数
if __name__ == "__main__":
    result = get_daily_recommendation("北京")
    print(f"城市: {result.city}")
    print(f"温度: {result.temperature}")
    print(f"湿度: {result.humidity}")
    print(f"天气状况: {result.condition}")
    print(f"推荐活动: {result.recommendation}")
```

## 高级示例

### 对话助手

这个示例演示了如何使用 `@llm_chat` 装饰器创建一个具有对话能力的助手，以及如何管理对话历史。

```python
from typing import List, Dict
from SimpleLLMFunc import llm_chat, tool
import os
import json
from datetime import datetime

# 工具函数定义
@tool(name="get_weather", description="获取指定城市的天气信息")
def get_weather(city: str) -> Dict[str, str]:
    return {"temperature": "25°C", "humidity": "60%", "condition": "Sunny"}

@tool(name="search_information", description="搜索特定主题的信息")
def search_information(query: str) -> str:
    # 模拟搜索功能
    return f"关于 '{query}' 的搜索结果: {...}"

# 历史记录管理函数
def save_history(history: List[Dict[str, str]], session_id: str) -> str:
    # 创建历史记录目录
    history_dir = os.path.join(os.getcwd(), "chat_history")
    os.makedirs(history_dir, exist_ok=True)

    # 格式化文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{session_id}_{timestamp}.json"
    filepath = os.path.join(history_dir, filename)

    # 保存历史记录
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {"session_id": session_id, "timestamp": timestamp, "history": history},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return filepath

def load_history(session_id: str = None, filepath: str = None) -> List[Dict[str, str]]:
    # 历史记录加载逻辑
    # ...省略具体实现...
    return []  # 返回加载的历史记录或空列表

# 创建对话助手
@llm_chat(llm_interface=llm_interface, toolkit=[get_weather, search_information])
def chat_assistant(message: str, history: List[Dict[str, str]] = None):
    """
    你是一个智能助手，可以回答用户的问题并提供帮助。
    你可以使用工具来获取实时信息，如天气状况和搜索结果。
    请保持友好、专业的态度，并尽可能提供准确、有用的信息。
    """
    pass

# 交互式聊天示例
def interactive_chat(session_id: str):
    # 加载历史记录
    history = load_history(session_id=session_id)
    
    print("====== 聊天助手 ======")
    print("输入 'exit' 退出对话")
    
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        
        # 调用聊天函数
        response, history = next(chat_assistant(user_input, history))
        print(f"助手: {response}")
        
        # 保存历史记录
        save_history(history, session_id)

if __name__ == "__main__":
    interactive_chat("user_12345")
```

### 复杂问题解决流程

这个示例展示了如何组合多个 LLM 函数来解决复杂问题，实现分步骤的推理和决策过程。

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function

# 步骤1：提取关键信息
class ProblemInfo(BaseModel):
    context: str = Field(..., description="问题背景")
    key_points: List[str] = Field(..., description="关键点列表")
    constraints: List[str] = Field(..., description="约束条件")
    
@llm_function(llm_interface=llm_interface)
def extract_problem_info(problem_description: str) -> ProblemInfo:
    """
    从问题描述中提取关键信息，包括背景、关键点和约束条件
    
    Args:
        problem_description: 详细的问题描述
        
    Returns:
        结构化的问题信息
    """
    pass

# 步骤2：生成解决方案选项
class SolutionOption(BaseModel):
    approach: str = Field(..., description="解决方案概述")
    pros: List[str] = Field(..., description="优势")
    cons: List[str] = Field(..., description="劣势")
    risk_level: str = Field(..., description="风险等级(低/中/高)")

@llm_function(llm_interface=llm_interface)
def generate_solution_options(problem_info: ProblemInfo) -> List[SolutionOption]:
    """
    基于问题信息生成多个可能的解决方案选项
    
    Args:
        problem_info: 结构化的问题信息
        
    Returns:
        可能的解决方案列表，每个方案包含概述、优缺点和风险评估
    """
    pass

# 步骤3：选择和详细化最佳方案
class DetailedSolution(BaseModel):
    solution_overview: str = Field(..., description="解决方案概述")
    implementation_steps: List[str] = Field(..., description="实施步骤")
    required_resources: List[str] = Field(..., description="所需资源")
    timeline: str = Field(..., description="预估时间线")
    success_metrics: List[str] = Field(..., description="成功指标")

@llm_function(llm_interface=llm_interface)
def select_and_detail_solution(
    problem_info: ProblemInfo, 
    solution_options: List[SolutionOption]
) -> DetailedSolution:
    """
    从多个方案中选择最佳解决方案，并提供详细的实施计划
    
    Args:
        problem_info: 结构化的问题信息
        solution_options: 可能的解决方案列表
        
    Returns:
        详细的解决方案实施计划
    """
    pass

# 使用组合函数流程
def solve_complex_problem(problem_description: str) -> DetailedSolution:
    # 步骤1：提取问题信息
    problem_info = extract_problem_info(problem_description)
    print("问题分析完成，提取了关键信息")
    
    # 步骤2：生成解决方案选项
    solution_options = generate_solution_options(problem_info)
    print(f"生成了 {len(solution_options)} 个可能的解决方案")
    
    # 步骤3：选择和详细化最佳方案
    detailed_solution = select_and_detail_solution(problem_info, solution_options)
    print("最佳解决方案已选定并详细化")
    
    return detailed_solution

# 执行示例
if __name__ == "__main__":
    problem = """
    我们公司是一家中型电商企业，目前面临客户流失率上升的问题。
    过去6个月，流失率从5%上升到12%。我们有约50,000名活跃客户，
    主要销售电子产品和家居用品。客户反馈显示，送货延迟和客服响应慢是主要抱怨点。
    我们的预算有限，需要在3个月内看到明显改善。如何解决这个问题？
    """
    
    solution = solve_complex_problem(problem)
    
    print("\n最终解决方案:")
    print(f"概述: {solution.solution_overview}\n")
    print("实施步骤:")
    for i, step in enumerate(solution.implementation_steps, 1):
        print(f"{i}. {step}")
    print("\n所需资源:")
    for resource in solution.required_resources:
        print(f"- {resource}")
    print(f"\n时间线: {solution.timeline}")
    print("\n成功指标:")
    for metric in solution.success_metrics:
        print(f"- {metric}")
```

## 实用工具示例

### 自定义提示模板

这个示例演示了如何使用自定义提示模板来控制 LLM 的行为和输出风格。

```python
from SimpleLLMFunc import llm_function

# 自定义系统提示模板
CUSTOM_SYSTEM_PROMPT = """
你是一个专业的{role}，擅长{specialty}。你的任务是：

{function_description}

请根据以下参数进行分析：
{parameters_description}

你需要输出：
{return_type_description}

请保持专业、简洁的风格，直接给出高质量的分析结果。
"""

# 自定义用户提示模板
CUSTOM_USER_PROMPT = """
请分析以下数据：
{parameters}

要求：
1. 分析必须基于数据
2. 给出明确的结论
3. 提供可行的建议
"""

@llm_function(
    llm_interface=llm_interface,
    system_prompt_template=CUSTOM_SYSTEM_PROMPT,
    user_prompt_template=CUSTOM_USER_PROMPT
)
def marketing_analysis(
    role: str = "市场分析师",
    specialty: str = "数字营销",
    sales_data: Dict[str, Any] = None,
    target_audience: str = "",
    campaign_history: List[Dict[str, Any]] = None
) -> MarketingReport:
    """
    分析销售数据和营销活动历史，提供市场策略建议
    
    Args:
        role: 分析角色
        specialty: 专业领域
        sales_data: 销售数据
        target_audience: 目标受众描述
        campaign_history: 过去营销活动的历史记录
        
    Returns:
        市场分析报告，包含趋势分析、效果评估和建议
    """
    pass
```
