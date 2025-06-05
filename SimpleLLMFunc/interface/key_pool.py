import heapq
from typing import List, Tuple, Dict
from SimpleLLMFunc.logger import push_critical, get_location
import threading # 导入 threading 模块

class APIKeyPool:
    # 类变量用于存储单例实例
    _instances: Dict[str, 'APIKeyPool'] = {}
    
    def __new__(cls, api_keys: List[str], provider_id: str) -> 'APIKeyPool':
        # 如果已经为这个 app_id 创建了实例，返回现有实例
        if provider_id in cls._instances:
            return cls._instances[provider_id]
        
        # 创建新实例
        instance = super(APIKeyPool, cls).__new__(cls)
        cls._instances[provider_id] = instance
        return instance
    
    def __init__(self, api_keys: List[str], provider_id: str) -> None:
        # 如果已经初始化，跳过初始化过程
        if hasattr(self, 'initialized') and self.initialized:
            return

        if len(api_keys) == 0 or api_keys is None:
            push_critical(
                f"API 密钥池 {provider_id} 为空。请检查您的配置。", # 更新日志为中文
                location=get_location()
            )
            
            raise ValueError(f"API 密钥池 {provider_id} 为空。请检查您的配置。") # 更新错误信息为中文
            
            
        self.api_keys = api_keys
        self.app_id = provider_id
        
        # 内存中的存储，替代 Redis
        self.heap: List[Tuple[float, str]] = [(0, key) for key in self.api_keys]
        heapq.heapify(self.heap)
        self.key_to_task_count: Dict[str, int] = {key: 0 for key in self.api_keys}
        
        self.lock = threading.Lock() # 为每个实例创建一个锁
        self.initialized = True
    
    def get_least_loaded_key(self) -> str:
        with self.lock: # 获取锁保护读操作
            # 获取任务数量最小的 API key
            if not self.heap:
                raise ValueError(f"{self.app_id} 没有可用的 API 密钥") # 更新错误信息为中文
            return self.heap[0][1]
    
    def increment_task_count(self, api_key: str) -> None:
        with self.lock: # 获取锁
            if api_key not in self.key_to_task_count:
                raise ValueError(f"API 密钥 {api_key} 不在池中") # 更新错误信息为中文
            
            # 增加任务计数
            self.key_to_task_count[api_key] += 1
            
            # 更新堆
            self._update_heap(api_key, self.key_to_task_count[api_key])
    
    def decrement_task_count(self, api_key: str) -> None:
        with self.lock: # 获取锁
            if api_key not in self.key_to_task_count:
                raise ValueError(f"API 密钥 {api_key} 不在池中") # 更新错误信息为中文
            
            # 减少任务计数
            self.key_to_task_count[api_key] -= 1
            
            # 更新堆
            self._update_heap(api_key, self.key_to_task_count[api_key])
    
    def _update_heap(self, api_key: str, new_task_count: int) -> None:
        # 找到并移除当前条目
        for i, (count, key) in enumerate(self.heap):
            if key == api_key:
                self.heap[i] = (float('inf'), key)  # 标记为移除
                break
                
        # 重新堆化以将标记的项移到末尾
        heapq.heapify(self.heap)
        
        # 移除标记的项并添加更新后的项
        self.heap.pop()
        heapq.heappush(self.heap, (new_task_count, api_key))