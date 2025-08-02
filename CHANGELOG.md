# Change log for SimpleLLMFunc

## 0.2.12.2 版本更新说明(Latest)

1. 给框架包添加了 `py.typed` 文件，用于支持类型检查。
-----

## 0.2.12.1 版本更新说明

1. 优化了日志记录，此前的日志自动个归档会对所有的indices进行统计，每次启动也会加载全部的indices。这是没有必要的。我们现在会将同一个trace id的log统计在同一个以trace
id命名的小文件中，这样可以实现增量存储同时分散磁盘读写压力。

2. 优化了Interface中的异步支持，使用asyncio的延时替换了同步延时。

-----

## 0.2.10 版本更新说明

1. **优化对工具返回值的处理**：现在的多模态工具可以返回说明文本的图像模态的组合。例如，你可以定义一个工具返回一个文本和一个图像路径的元组，这样在提示LLM如何理解工具返回时会更加灵活。

-----

## 0.2.8 版本更新说明

1. **添加了多模态返回支持的Tool Call**：原来的tool只能是返回str或者是可序列化的内容的对象，现在可以通过标注返回值是`ImgPath`来返回本地图像，或者标注为`ImgUrl`来返回网络图像。请依然注意tool装饰器装饰的函数只能有一个返回值，不能是元组或列表，也不能是不可序列化的类型。

2. **优化了日志记录**：减少了重复日志的保存。

-----

## 0.2.7 版本更新说明

1. **修改了llm chat装饰器返回的history内容**：原来会自动过滤掉tool call内容，现在这些内容也会被返回，用户可以自行决定是否保留或去除

2. **使用rich提供更好的traceback打印**：现在的trace back打印更加优雅和美观了，rich果然是好东西

-----

## 0.2.6 版本更新说明

1. **给Tools添加了可选参数支持**：现在你可以在定义工具时添加可选参数，这些参数可以在调用工具时被省略。这样可以使得工具更加灵活，适应不同的调用场景。

2. **修复了以llm function作为tools提供给llm chat使用时的问题**：Python本身不允许async loop中嵌套另一个async loop，但是由于不管是同步装饰器还是异步装饰器，实际上都是依靠异步调用来实现的。同步的实现是通过一个async loop来得到异步函数的结果。而在同步llm chat中调用同步llm function就会使得两个async loop嵌套。现在通过引入了`nested_asyncio`通过`monkey-patch`的形式修复了这个问题。现在你可以在`llm_chat`中使用同步的`llm_function`作为工具调用。

**WARNING**: 由于使用了`nested_asyncio`，我们不能保证在所有的环境下都能正常工作，特别是在某些特定的异步框架中可能会出现问题。如果你遇到问题，请考虑提出issue

-----

## 0.2.5 版本更新说明

1. **修正多模态参数解析问题**

2. **修正了同步装饰器依赖于asyncio loop异步方法条件下的loop不正确关闭的问题**

3. **添加了TokenBucket**： OpenAICompatible接口现在集成了令牌桶算法用于流量平滑，和APIKeyPool一样可以在json中配置，也是针对一个模型管理一个令牌桶。

4. 更新了文档， 新了了关于llm 接口的详细说明

## 0.2.3 版本更新说明

1. **提供了多模态支持，现在能够处理图像输入了**：提供了更加语意明确的多模态类型标注，`Text`， `ImgUrl`和`ImgPath`，推荐使用这三个进行所有`llm_function`和`llm_chat`装饰器的函数参数类型标注，装饰器会根据标注构建正确的多模态用户消息。但是对于`Text`，使用`str` 类型标注也可以正常工作，但从语意上来说，`Text`更为明确。

2. **使用原生async来实现异步，而不是上一个版本的基于线程池的异步**：现在所有的异步函数都使用原生的`async`和`await`来实现，这样可以更好地支持异步调用，避免了线程池带来的性能瓶颈。现在`llm_function`和`llm_chat`装饰器都支持异步函数，使用时只需在函数定义前添加`async`关键字，并使用相应的异步装饰器``@async_llm_function`或`@async_llm_chat`。

----

## 0.2.1版本更新说明

1. **llm function异步支持**：专门为异步函数调用提供了`@async_llm_function`装饰器，允许开发者创建异步的 LLM 函数。这个装饰器与`@llm_function`类似，但支持异步函数的调用。

2. **更好的log**：

- 现在将更多用于中间调试的信息改为debug级别的log，减少了info级别的log输出，使得日志更加简洁明了。专门为提取训练数据而生。
- 注册了一个自定义清理逻辑，确保在主进程结束时所有的日志都会被正确地写入文件中，避免日志丢失。

3. **文档更新**：
   - 更新了文档中对于`@async_llm_function`的使用示例，提供了更清晰的异步函数调用方式。
   - 增加了关于如何在异步环境中使用 LLM 函数的指导。

----

## 0.2.0版本更新说明

1. **Stream支持**：现在`llm_chat`装饰器支持传入一个`stream`参数，允许用户在调用时选择是否开启流式响应。开启流式响应后，LLM的响应将以流的形式返回，而不是一次性返回完整结果。

-----

## 0.1.13版本更新说明

1. 原来的tool call流程中，extract出来的tool call不会按照openai定义的message格式添加到message中，这再众多模型上不会导致问题，但是在OpenAI自家的模型上，如果没有assistant的tool call信息，API在接收到 role 为 tool 的消息时会出现错误。现在修复了这个问题，tool call会被正确的添加到message中。

-----

## 0.1.12版本更新说明

1. 更新了文档中对于`OpenAICompatible`借口创建的不正确样例
2. 将ReAct模式下的工具调用判定修改的更加具有语义性
3. 给API Key Pool加上互斥锁，确保读取和修改load的线程安全，但是高并发下这个锁可能成为性能瓶颈

-----

## 0.1.11版本更新说明

1. 修复了`llm_chat`装饰器中工具信息没有在system prompt中被正确传递的问题

-----

## 0.1.10版本更新说明

1. 修复了`llm_chat`装饰器中不正确的Callable类型标注

-----

## 0.1.9版本更新说明

1. 修复了`llm_function`装饰器中由于三元表达式没有使用括号界定范围导致在没有toolkit的时候system prompt消失的问题

-----

## 0.1.8版本更新说明

1. 修复了`llm_chat`装饰器中由于`toolkit`不传递导致的`tool_objects`局部变量未定义的问题

-----

## 0.1.7版本更新说明

1. fix a small bug
2. 发现此前的版本号更新有点太慢了，其实可以跨度大一点所以直接到0.1.7
3. 0.2.0版本计划增加一个Tool Call策略，为后续支持MCP以及不支持Tool Call的模型做准备

## 0.1.6.3版本更新说明  

1. **tools序列化手段更新**
   - 此前的版本中，如果你想手动序列化一个工具，比较朴素的方法是使用`Tool`类的静态方法`serialize_tools(tools_list: List[Tool])`

**例如**：

```python
from SimpleLLMFunc import Tool, tool

@tool(name="shell", description="a shell")
def shell(command: str) -> tuple[str, str]:

    """
    Args:
        command: the command need to be executed in shell

    Returns:
        tuple[str, str]: stdout and stderr
    """
    # implement here

serialized = Tool.serialize_tools([shell])[0]
```

这样`serialized`中会存放`shell`这个工具的序列化信息。

但在此前的版本中这会有bug，实际上应该写作：

```python
serialized = Tool.serialize_tools([shell._tool])[0]
```

因为这个方法的列表要求是`Tool`对象列表，而实际上被`@tool`装饰的函数，其会被添加一个`_tool`属性，这个属性是一个`Tool`对象。

而在`0.1.6.3+`的版本中，这一反直觉的问题被修复了，我们们可以使用第一个例子中符合直觉的写法，直接传入函数来得到序列化结果。

----

## 0.1.6.2版本更新说明

### 主要更新

1. **实现token usage统计**
   - 现在`llm_function`和`llm_chat`装饰的函数都能够对一次调用中属于自己上下文范围内的token用量进行统计，并呈现在log中。
   - 我们可以通过查看同一个`function call trace id`下的所有log中时间上最后出现的带有token信息的log来看到在本次调用中消耗了多少token
   - **!!!注意!!!**：token usage只统计函数自身上下文，对于其中的嵌套调用不会统计在外层函数的计数内。

**例如:**

```json
"get_daily_recommendation_86cdc1b5-f279-4643-b2a0-13782ab30b26": [
    {
        "timestamp": "2025-05-16T20:23:02.428657+08:00",
        "level": "INFO",
        "location": "llm_function_example.py:main:116",
        "message": "LLM 函数 'get_daily_recommendation' 被调用，参数: {\n    \"city\": \"Hangzhou\"\n}",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:02.429832+08:00",
        "level": "DEBUG",
        "location": "llm_function_example.py:main:116",
        "message": "系统提示 .... 格式或代码块包裹结果，请直接输出期望的内容或者对应的JSON表示",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:02.430167+08:00",
        "level": "DEBUG",
        "location": "llm_function_example.py:main:116",
        "message": "用户提示: 给定的参数如下:\n      - city: Hangzhou\n\n直接返回结果，不需要任何解释或格式化。",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:02.430431+08:00",
        "level": "INFO",
        "location": "llm_function_example.py:main:116",
        "message": "开始 LLM 调用...",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:02.430682+08:00",
        "level": "INFO",
        "location": "utils.py:get_last_item_of_generator:12",
        "message": "LLM 函数 'get_daily_recommendation' 发起初始请求，消息数: 2",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:02.498193+08:00",
        "level": "INFO",
        "location": "utils.py:execute_llm:74",
        "message": "OpenAICompatible::chat: deepseek-v3-250324 request with API key: ..., and message: [\n    {\n        \"role\": \"system\",\n        \"content\": \"你的任务是按照以下的**功能描述**，根据用户的要求，给出符合要求的结果。\\n\\n- 功能描述:\\n    \\n通过get_weather工具获取天气信息，并给出推荐的活动\\n\\nArgs:\\n    city: 城市名称\\n\\nReturns:\\n    WeatherInfo对象，包含温度、湿度和天气状况\\n\\n\\n- 你会接受到以下参数：\\n      - city: <class 'str'>\\n\\n- 你需要返回内容的类型: \\n    WeatherInfo (Pydantic模型) 包含以下字段:\\n  - city (string, 必填): 城市名称\\n  - temperature (string, 必填): 当前温度\\n  - humidity (string, 必填): 当前湿度\\n  - condition (string, 必填): 天气状况\\n  - recommendation (string, 必填): 推荐的活动\\n\\n执行要求:\\n1. 如果有工具可用，可以使用工具来辅助完成任务\\n2. 不要用 markdown 格式或代码块包裹结果，请直接输出期望的内容或者对应的JSON表示\"\n    },\n    {\n        \"role\": \"user\",\n        \"content\": \"给定的参数如下:\\n      - city: Hangzhou\\n\\n直接返回结果，不需要任何解释或格式化。\"\n    }\n]",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.016900+08:00",
        "level": "INFO",
        "location": "utils.py:get_last_item_of_generator:12",
        "message": "LLM 函数 'get_daily_recommendation' 收到初始响应",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.017463+08:00",
        "level": "INFO",
        "location": "utils.py:get_last_item_of_generator:12",
        "message": "LLM 函数 'get_daily_recommendation' 发现 1 个工具调用，开始执行工具",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.021981+08:00",
        "level": "INFO",
        "location": "utils.py:_process_tool_calls:441",
        "message": "执行工具 'get_weather' 参数: {\"city\":\"Hangzhou\"}",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.022385+08:00",
        "level": "INFO",
        "location": "utils.py:_process_tool_calls:454",
        "message": "工具 'get_weather' 执行完成: {\"temperature\": \"32°C\", \"humidity\": \"80%\", \"condition\": \"Raining\"}",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.022618+08:00",
        "level": "INFO",
        "location": "utils.py:get_last_item_of_generator:12",
        "message": "LLM 函数 'get_daily_recommendation' 工具调用循环: 第 2/5 次调用",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.073555+08:00",
        "level": "INFO",
        "location": "utils.py:execute_llm:121",
        "message": "OpenAICompatible::chat: deepseek-v3-250324 request with API key: ..., and message: [\n    {\n        \"role\": \"system\",\n        \"content\": \"你的任务是按照以下的**功能描述**，根据用户的要求，给出符合要求的结果。\\n\\n- 功能描述:\\n    \\n通过get_weather工具获取天气信息，并给出推荐的活动\\n\\nArgs:\\n    city: 城市名称\\n\\nReturns:\\n    WeatherInfo对象，包含温度、湿度和天气状况\\n\\n\\n- 你会接受到以下参数：\\n      - city: <class 'str'>\\n\\n- 你需要返回内容的类型: \\n    WeatherInfo (Pydantic模型) 包含以下字段:\\n  - city (string, 必填): 城市名称\\n  - temperature (string, 必填): 当前温度\\n  - humidity (string, 必填): 当前湿度\\n  - condition (string, 必填): 天气状况\\n  - recommendation (string, 必填): 推荐的活动\\n\\n执行要求:\\n1. 如果有工具可用，可以使用工具来辅助完成任务\\n2. 不要用 markdown 格式或代码块包裹结果，请直接输出期望的内容或者对应的JSON表示\"\n    },\n    {\n        \"role\": \"user\",\n        \"content\": \"给定的参数如下:\\n      - city: Hangzhou\\n\\n直接返回结果，不需要任何解释或格式化。\"\n    },\n    {\n        \"role\": \"tool\",\n        \"tool_call_id\": \"call_e9tcmq348bazj913arib32d8\",\n        \"content\": \"{\\\"temperature\\\": \\\"32°C\\\", \\\"humidity\\\": \\\"80%\\\", \\\"condition\\\": \\\"Raining\\\"}\"\n    }\n]",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:07.450156+08:00",
        "level": "DEBUG",
        "location": "utils.py:get_last_item_of_generator:12",
        "message": "LLM 函数 'get_daily_recommendation' 没有更多工具调用，返回最终响应",
        "input_tokens": 360,
        "output_tokens": 60
    },
    {
        "timestamp": "2025-05-16T20:23:07.450659+08:00",
        "level": "INFO",
        "location": "llm_function_example.py:main:116",
        "message": "LLM 函数 'get_daily_recommendation' 收到最终响应",
        "input_tokens": 360,
        "output_tokens": 60
    },
    {
        "timestamp": "2025-05-16T20:23:07.451190+08:00",
        "level": "INFO",
        "location": "llm_function_example.py:main:116",
        "message": "\"ChatCompletion(id='0217473981842466773b021bc722319529820bd350d494160c9a5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{\\\"city\\\": \\\"Hangzhou\\\", \\\"temperature\\\": \\\"32°C\\\", \\\"humidity\\\": \\\"80%\\\", \\\"condition\\\": \\\"Raining\\\", \\\"recommendation\\\": \\\"It\\\\'s raining and humid, so it\\\\'s best to stay indoors or carry an umbrella if you go out.\\\"}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1747398187, model='deepseek-v3-250324', object='chat.completion', service_tier='default', system_fingerprint=None, usage=CompletionUsage(completion_tokens=60, prompt_tokens=360, total_tokens=420, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=0, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0)))\"",
        "input_tokens": 360,
        "output_tokens": 60
    },
    // 读取这条log的token usage一定能表征这次调用的token消耗
    {
        "timestamp": "2025-05-16T20:23:07.451565+08:00",
        "level": "DEBUG",
        "location": "utils.py:_extract_content_from_response:278",
        "message": "LLM 函数 'get_daily_recommendation' 提取的内容:\n{\"city\": \"Hangzhou\", \"temperature\": \"32°C\", \"humidity\": \"80%\", \"condition\": \"Raining\", \"recommendation\": \"It's raining and humid, so it's best to stay indoors or carry an umbrella if you go out.\"}",
        "input_tokens": 360,
        "output_tokens": 60
    }
]
```

2. **装饰器函数签名问题**
   - 上一个版本中，被`llm_function`和`llm_chat`装饰的函数，其返回对象的函数签名会发生变化而无法正确的被其他依赖于函数签名信息的装饰器装饰
   - 所以这一个版本修复了这个问题，主要是为了解决以下的场景：

**例如：**

你想要使用`llm_function`来实现一个`tool`，那么你会这样做：

```python
@tool(name="code_writer", description="你可以使用这个工具来生成高质量的代码")
@llm_function(llm_interface=your_llm_interface)
def code_generation(query: str) -> str:
    """
    Args:
        query: str
    Returns:
        str: 生成的代码

    你是一位得力的资深程序员 ...
    """
```

这种非常符合直觉的写法在上一个版本中是不可行的，因为被`llm_function`装饰器装饰后，返回的函数是一个`wrapper(**args, **kargs)`，缺失了参数列表和type hint，而`tool`装饰器依赖于这些信息。

但在`0.1.6+`中，无论是使用`llm_function`装饰的函数还是使用`llm_chat`装饰的函数，其返回函数的签名信息：包括参数列表，类型提示，函数名称都和原始函数一致。所以上面的例子是可行的。

------

## 0.1.5版本更新说明

### 主要更新

1. **供应商配置优化**
   - 使用 JSON 文件替代 .env 配置供应商信息
   - 更灵活的模型参数配置
   - 多供应商统一配置管理

2. **Prompt 模板优化**
   - 优化 LLM 函数装饰器的默认 prompt 模板
   - 减少 token 使用的同时提升效果
   - 更清晰的指令描述和参数说明

### 配置示例 (provider.json)

```json
{
    "volc_engine": [
        {
            "model_name": "deepseek-v3-250324",
            "api_keys": ["your-api-key"],
            "base_url": "https://api.volc.example.com/v1",
            "max_retries": 3,
            "retry_delay": 1,
        }
    ],
    "openai": [
        {
            "model_name": "gpt-3.5-turbo",
            "api_keys": ["your-api-key"],
            "base_url": "https://api.openai.com/v1",
            "max_retries": 3,
            "retry_delay": 1,
        },
        {
            "model_name": "gpt-4",
            "api_keys": ["your-api-key"],
            "base_url": "https://api.openai.com/v1",
            "max_retries": 3,
            "retry_delay": 1,
        }
    ]
}
```

### 使用示例

```python
from SimpleLLMFunc import OpenAICompatible

# 从配置文件加载所有模型接口
provider_interfaces = OpenAICompatible.load_from_json_file("provider.json")

# 获取特定模型接口
deepseek_interface = provider_interfaces["volc_engine"]["deepseek-v3-250324"]

# 在装饰器中使用
@llm_function(llm_interface=deepseek_interface)
def my_function():
    pass
```
