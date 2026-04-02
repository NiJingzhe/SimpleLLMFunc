from __future__ import annotations

import asyncio

from SimpleLLMFunc import APIKeyPool, OpenAICompatible, llm_function


llm = OpenAICompatible(
    api_key_pool=APIKeyPool(
        api_keys=["sk-your-key"],
        provider_id="openrouter-z-ai-glm-5",
    ),
    model_name="z-ai/glm-5",
    base_url="https://openrouter.ai/api/v1",
)


@llm_function(llm_interface=llm)
async def instant_answer(request: str) -> str:
    """
    Complete the user's request in a concise, practical way.
    Keep the answer technically accurate and directly usable.
    If the request asks for multiple points, use a short bullet list.
    """
    pass


print(
    asyncio.run(
        instant_answer("Explain why prompt-as-code is useful in three bullets.")
    )
)
