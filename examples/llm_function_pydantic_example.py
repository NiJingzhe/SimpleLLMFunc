"""
llm_function 返回复杂 Pydantic 模型示例
"""

import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel, Field

from SimpleLLMFunc.llm_decorator import llm_function
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible


# 配置 LLM
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")

try:
    llm_interface = OpenAICompatible.load_from_json_file(
        provider_json_path
    )["volc_engine"]["deepseek-v3-250324"]
except Exception:
    llm_interface = None


# 定义复杂的 Pydantic 模型
class Address(BaseModel):
    """地址信息"""
    street: str = Field(..., description="街道地址")
    city: str = Field(..., description="城市名称")
    state: Optional[str] = Field(None, description="州/省名称")
    zip_code: str = Field(..., description="邮政编码")
    country: str = Field(default="中国", description="国家名称")


class Contact(BaseModel):
    """联系方式"""
    email: str = Field(..., description="电子邮箱")
    phone: Optional[str] = Field(None, description="电话号码")
    website: Optional[str] = Field(None, description="网站URL")


class Product(BaseModel):
    """商品信息"""
    name: str = Field(..., description="商品名称")
    price: float = Field(..., description="商品价格")
    category: str = Field(..., description="商品类别")
    tags: List[str] = Field(default_factory=list, description="标签列表")


class Company(BaseModel):
    """公司信息"""
    name: str = Field(..., description="公司名称")
    founded_year: int = Field(..., description="成立年份")
    employee_count: int = Field(..., description="员工数量")
    address: Address = Field(..., description="公司地址")
    contact: Contact = Field(..., description="联系方式")
    products: List[Product] = Field(default_factory=list, description="产品列表")
    is_public: bool = Field(default=False, description="是否上市公司")


class SearchResult(BaseModel):
    """搜索结果"""
    query: str = Field(..., description="搜索查询词")
    total_results: int = Field(..., description="总结果数")
    companies: List[Company] = Field(default_factory=list, description="公司列表")
    search_time_ms: float = Field(..., description="搜索耗时（毫秒）")


# 定义返回复杂 Pydantic 模型的 llm_function
@llm_function(llm_interface=llm_interface)
async def search_companies(query: str, max_results: int = 3) -> SearchResult:
    """搜索符合条件的公司信息"""
    pass


async def main():
    if not llm_interface:
        print("请配置 provider.json")
        return
    
    result = await search_companies("AI 科技公司", max_results=2)
    
    import json
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

