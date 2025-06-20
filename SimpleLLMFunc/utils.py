"""这个文件中包含各种在整个项目中被广泛使用的工具函数
"""
from typing import Generator, TypeVar, AsyncGenerator

T = TypeVar("T")

def get_last_item_of_generator(generator: Generator[T, None, None]) -> T | None:
    """
    获取生成器的最后一个元素
    """
    last_item = None
    for item in generator:
        last_item = item
    return last_item

async def get_last_item_of_async_generator(generator: AsyncGenerator[T, None]) -> T | None:
    """
    获取异步生成器的最后一个元素
    """
    last_item = None
    async for item in generator:
        last_item = item
    return last_item