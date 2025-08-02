"""
@File    :   __init__.py
@Time    :   2025/08/03 02:19:19
@Author  :   Jingzhe Ni 
@Contact :   nijingzhe@zju.edu.cn
@License :   (C)Copyright 2025, Jingzhe Ni
@Desc    :   Init for SimpleLLMFunc
"""
import nest_asyncio  # type: ignore
nest_asyncio.apply()

from rich import traceback
traceback.install(show_locals=True)

from SimpleLLMFunc.config import *
from SimpleLLMFunc.llm_decorator import *
from SimpleLLMFunc.logger import *
from SimpleLLMFunc.tool import *
from SimpleLLMFunc.interface import *