"""
llm_function 返回复杂 Pydantic 模型示例

本示例展示如何使用 @llm_function 装饰器返回复杂的 Pydantic 模型，
并验证系统能够正确生成 XML Schema 和解析 XML 响应。
"""

import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from SimpleLLMFunc.llm_decorator import llm_function
from SimpleLLMFunc.base.type_resolve.description import (
    build_type_description_xml,
    generate_example_xml,
)
from SimpleLLMFunc.base.post_process import process_response


# ============ 定义复杂的 Pydantic 模型 ============

class Address(BaseModel):
    """地址信息"""
    street: str = Field(..., description="街道地址")
    city: str = Field(..., description="城市名称")
    state: Optional[str] = Field(None, description="州/省名称，可选")
    zip_code: str = Field(..., description="邮政编码")
    country: str = Field(default="中国", description="国家名称")


class Contact(BaseModel):
    """联系方式"""
    email: str = Field(..., description="电子邮箱地址")
    phone: Optional[str] = Field(None, description="电话号码，可选")
    website: Optional[str] = Field(None, description="网站URL，可选")


class Product(BaseModel):
    """商品信息"""
    name: str = Field(..., description="商品名称")
    price: float = Field(..., description="商品价格，单位：元")
    category: str = Field(..., description="商品类别")
    in_stock: bool = Field(default=True, description="是否有库存")
    tags: List[str] = Field(default_factory=list, description="商品标签列表")


class Company(BaseModel):
    """公司信息模型"""
    name: str = Field(..., description="公司名称")
    founded_year: int = Field(..., description="成立年份")
    employee_count: int = Field(..., description="员工数量")
    address: Address = Field(..., description="公司地址")
    contact: Contact = Field(..., description="联系方式")
    products: List[Product] = Field(default_factory=list, description="产品列表")
    is_public: bool = Field(default=False, description="是否上市公司")


class SearchResult(BaseModel):
    """搜索结果模型"""
    query: str = Field(..., description="搜索查询词")
    total_results: int = Field(..., description="总结果数")
    companies: List[Company] = Field(default_factory=list, description="公司列表")
    search_time_ms: float = Field(..., description="搜索耗时（毫秒）")


# ============ 定义返回复杂 Pydantic 模型的 llm_function ============

# 注意：这里使用一个模拟的 LLM interface，实际使用时需要配置真实的 LLM
# 为了演示，我们直接测试 XML Schema 生成和解析功能

async def demonstrate_pydantic_return():
    """演示 Pydantic 返回类型的生成和解析"""
    
    print("=" * 80)
    print("llm_function 返回复杂 Pydantic 模型示例")
    print("=" * 80)
    print()
    
    # ============ 1. 生成 XML Schema ============
    print("1. 生成 XML Schema 描述")
    print("-" * 80)
    
    xml_schema = build_type_description_xml(SearchResult)
    print(xml_schema)
    print()
    
    # ============ 2. 生成 XML 示例 ============
    print("2. 生成 XML 示例")
    print("-" * 80)
    
    xml_example = generate_example_xml(SearchResult)
    print(xml_example)
    print()
    
    # ============ 3. 测试 XML 解析 ============
    print("3. 测试 XML 解析为 Pydantic 模型")
    print("-" * 80)
    
    # 模拟 LLM 返回的 XML（符合 Schema）
    sample_xml = """<SearchResult>
  <query>科技公司</query>
  <total_results>2</total_results>
  <search_time_ms>125.5</search_time_ms>
  <companies>
    <item>
      <name>示例科技公司</name>
      <founded_year>2010</founded_year>
      <employee_count>500</employee_count>
      <address>
        <street>科技大道123号</street>
        <city>北京</city>
        <state>北京</state>
        <zip_code>100000</zip_code>
        <country>中国</country>
      </address>
      <contact>
        <email>contact@example.com</email>
        <phone>010-12345678</phone>
        <website>https://www.example.com</website>
      </contact>
      <products>
        <item>
          <name>产品A</name>
          <price>99.99</price>
          <category>软件</category>
          <in_stock>true</in_stock>
          <tags>
            <item>热门</item>
            <item>推荐</item>
          </tags>
        </item>
        <item>
          <name>产品B</name>
          <price>199.99</price>
          <category>硬件</category>
          <in_stock>true</in_stock>
          <tags>
            <item>新品</item>
          </tags>
        </item>
      </products>
      <is_public>true</is_public>
    </item>
    <item>
      <name>另一家公司</name>
      <founded_year>2015</founded_year>
      <employee_count>200</employee_count>
      <address>
        <street>创新路456号</street>
        <city>上海</city>
        <zip_code>200000</zip_code>
        <country>中国</country>
      </address>
      <contact>
        <email>info@another.com</email>
      </contact>
      <is_public>false</is_public>
    </item>
  </companies>
</SearchResult>"""
    
    print("输入 XML:")
    print(sample_xml)
    print()
    
    # 创建一个模拟的响应对象
    class MockResponse:
        def __init__(self, content):
            class Choice:
                class Message:
                    def __init__(self, content):
                        self.content = content
                def __init__(self, content):
                    self.message = self.Message(content)
            self.choices = [Choice(content)]
    
    mock_response = MockResponse(sample_xml)
    
    # 解析为 Pydantic 模型
    try:
        result = process_response(mock_response, SearchResult)
        
        print("解析结果:")
        print(f"类型: {type(result)}")
        print(f"是否为 SearchResult 实例: {isinstance(result, SearchResult)}")
        print()
        
        print("解析后的数据:")
        print(f"查询词: {result.query}")
        print(f"总结果数: {result.total_results}")
        print(f"搜索耗时: {result.search_time_ms}ms")
        print(f"公司数量: {len(result.companies)}")
        print()
        
        print("第一个公司信息:")
        company1 = result.companies[0]
        print(f"  名称: {company1.name}")
        print(f"  成立年份: {company1.founded_year}")
        print(f"  员工数: {company1.employee_count}")
        print(f"  地址: {company1.address.street}, {company1.address.city}")
        print(f"  联系方式: {company1.contact.email}, {company1.contact.phone}")
        print(f"  产品数量: {len(company1.products)}")
        if company1.products:
            print(f"  第一个产品: {company1.products[0].name}, ¥{company1.products[0].price}")
        print(f"  是否上市: {company1.is_public}")
        print()
        
        print("第二个公司信息:")
        company2 = result.companies[1]
        print(f"  名称: {company2.name}")
        print(f"  地址: {company2.address.street}, {company2.address.city}")
        print(f"  联系方式: {company2.contact.email}")
        print(f"  产品数量: {len(company2.products)}")
        print()
        
        # 验证模型验证
        print("4. 验证 Pydantic 模型验证功能")
        print("-" * 80)
        
        # 测试模型验证
        json_data = result.model_dump()
        validated = SearchResult.model_validate(json_data)
        print(f"✓ 模型验证成功: {validated.query == result.query}")
        print()
        
        # 测试 JSON 序列化
        import json as json_lib
        json_str = json_lib.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        print("5. JSON 序列化结果（前500字符）:")
        print("-" * 80)
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        print()
        
        print("=" * 80)
        print("✓ 所有测试通过！复杂 Pydantic 模型可以正确生成和解析")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        import traceback
        traceback.print_exc()


# ============ 实际使用示例（需要配置 LLM） ============

# 这是一个示例函数，展示如何使用 @llm_function 返回复杂 Pydantic 模型
# 实际使用时需要配置 LLM interface

"""
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible

# 配置 LLM interface（需要实际的配置）
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["your_provider"]["your_model"]

@llm_function(llm_interface=llm_interface)
async def search_companies(query: str, max_results: int = 10) -> SearchResult:
    '''
    搜索符合条件的公司信息。
    
    根据查询词搜索公司，返回包含公司详细信息、地址、联系方式和产品列表的搜索结果。
    
    Args:
        query: 搜索查询词，例如 "科技公司"、"互联网企业" 等
        max_results: 最大返回结果数量，范围 1-50，默认为 10
    
    Returns:
        SearchResult 对象，包含搜索结果和公司详细信息
    '''
    pass

# 使用示例
async def main():
    result = await search_companies("科技公司", max_results=5)
    print(f"找到 {result.total_results} 个结果")
    for company in result.companies:
        print(f"- {company.name}: {company.address.city}")
"""


if __name__ == "__main__":
    asyncio.run(demonstrate_pydantic_return())

