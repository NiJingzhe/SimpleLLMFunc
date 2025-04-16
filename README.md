# SimpleAgent
----
## Config

   - 使用dotenv实现配置文件
   - 环境变量高于.env高于config.py
   


## LLM Provider

  - 应当尽可能的简单
  - 本身是无状态的函数调用
  - 只提供LLM Call这一个功能，不管历史记录
  - 全部使用OpenAI的SDK来实现
  - 提供普通调用和流式调用两种
  - 返回未经处理的response或者chunk


## Agent自身的状态管理

  - Agent自身应当有一个程序化的状态管理机制，而不是仅仅凭借LLM的能力进行状态转移。
  - 以下是一个Agent状态管理的流程:
    
    - Perception: 从 Public Space 中读取上下文
    - Planning: 生成一个任务计划（产生一堆子任务， 基于提供的能力）
    - Execution: 执行计划
    - Feedback: 反馈执行结果
  

