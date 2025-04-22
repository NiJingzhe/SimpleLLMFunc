"""
使用LLM函数装饰器的示例
"""
from typing import Dict, List
from pydantic import BaseModel, Field

from SimpleLLMGraph.llm_decorator.llm_function_decorator import llm_function
from SimpleLLMGraph.interface import ZhipuAI_glm_4_flash_Interface

# 定义一个Pydantic模型作为返回类型
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")

# 使用装饰器创建一个LLM函数
@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    system_prompt="你是一个专业的产品评测专家，可以客观公正地评价各种产品。"
)
def analyze_product_review(product_name: str, review_text: str) -> ProductReview:
    """
    分析产品评论，提取关键信息并生成结构化评测报告
    
    Args:
        product_name: 产品名称
        review_text: 用户评论文本
        
    Returns:
        包含评分、优缺点和总结的产品评测报告
    """
    pass  # 函数体为空，实际执行由LLM完成

# 基础类型返回示例
@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface
)
def summarize_text(text: str, max_words: int) -> str:
    """
    将文本概括为指定字数的摘要
    
    Args:
        text: 需要概括的文本
        max_words: 摘要的最大字数
        
    Returns:
        文本摘要，请务必严格遵循字数要求！
    """
    pass

# 字典返回示例
@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface
)
def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    从文本中提取实体（人物、地点、组织等）
    
    Args:
        text: 需要分析的文本
        
    Returns:
        字典形式的实体分类列表，例如 {"people": ["张三", "李四"], "locations": ["北京", "上海"]}
    """
    pass

def main():
    # 测试产品评测分析
    product_name = "XYZ无线耳机"
    review_text = """
    我买了这款XYZ无线耳机已经使用了一个月。音质非常不错，尤其是低音部分表现出色，
    佩戴也很舒适，可以长时间使用不感到疲劳。电池续航能力也很强，充满电后可以使用约8小时。
    不过连接偶尔会有些不稳定，有时候会突然断开。另外，触控操作不够灵敏，经常需要点击多次才能响应。
    总的来说，这款耳机性价比很高，适合日常使用，但如果你需要用于专业音频工作可能还不够。
    """
    
    try:
        print("\n===== 产品评测分析 =====")
        result = analyze_product_review(product_name, review_text)
        print(f"评分: {result.rating}/5")
        print("优点:")
        for pro in result.pros:
            print(f"- {pro}")
        print("缺点:")
        for con in result.cons:
            print(f"- {con}")
        print(f"总结: {result.summary}")
    except Exception as e:
        print(f"产品评测分析失败: {e}")
    
    # 测试文本摘要
    long_text = """
    人工智能(AI)是计算机科学的一个分支，致力于开发能够执行通常需要人类智能的任务的系统。
    这些任务包括视觉感知、语音识别、决策制定和语言翻译等。AI的历史可以追溯到20世纪50年代，
    当时计算机科学家开始探索机器是否可以"思考"。如今，AI已经融入我们日常生活的方方面面，
    从智能手机上的虚拟助手到自动驾驶汽车，再到推荐系统和医疗诊断工具。
    机器学习是AI的一个子领域，它使系统能够从数据中学习和改进，而无需被明确编程。
    深度学习是机器学习的一种形式，它使用多层神经网络来分析数据的各个方面。
    随着计算能力的增加和数据可用性的提高，AI技术在近年来取得了显著进步。
    然而，随着AI的发展，也出现了关于隐私、安全和道德影响的担忧。
    """
    
    try:
        print("\n===== 文本摘要 =====")
        summary = summarize_text(long_text, 20)
        print(summary)
    except Exception as e:
        print(f"文本摘要失败: {e}")
    
    # 测试实体提取
    entity_text = """
    中国国家主席习近平周四在北京会见了美国总统拜登的特使克里。
    会议在人民大会堂举行，双方讨论了气候变化问题。
    克里表示，美国和中国作为世界上最大的两个经济体，应该共同努力应对全球气候危机。
    同时，阿里巴巴和腾讯等中国科技公司正积极投资绿色技术。
    """
    
    try:
        print("\n===== 实体提取 =====")
        entities = extract_entities(entity_text)
        for entity_type, entity_list in entities.items():
            print(f"{entity_type}:")
            for entity in entity_list:
                print(f"- {entity}")
    except Exception as e:
        print(f"实体提取失败: {e}")
        

if __name__ == "__main__":
    main()