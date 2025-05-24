# 贡献指南

感谢你对 SimpleLLMFunc 项目的兴趣！我们欢迎并鼓励社区贡献，无论是修复错误、改进文档，还是添加新功能。

## 如何贡献

### 提交问题(Issue)

如果你发现了问题或有新功能建议，请先在 [GitHub Issues](https://github.com/NiJingzhe/SimpleLLMFunc/issues) 页面搜索相关内容，以确保你的问题或建议尚未被提出。如果没有相关内容，你可以创建一个新的 Issue，请务必：

1. 使用清晰的标题描述问题
2. 提供详细的问题描述或功能请求
3. 如果是 bug，请提供复现步骤和环境信息
4. 如果可能，包含代码示例或截图

### 提交代码更改(Pull Request)

1. Fork 项目仓库
2. 创建你的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

### 开发流程

1. 确保你已经设置好开发环境（见下文）
2. 在开始工作前，请先同步最新的代码
3. 为你的功能或修复编写测试用例
4. 确保所有测试都通过
5. 遵循项目的代码风格和约定

## 开发环境设置

### 依赖项

- Python 3.10 或更高版本
- Poetry (推荐的依赖管理工具)

### 安装开发依赖

```bash
git clone https://github.com/NiJingzhe/SimpleLLMFunc.git
cd SimpleLLMFunc
poetry install
```

<!--

### 测试

TODO

-->

## 代码规范

### 代码风格

我们使用 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 作为 Python 代码风格指南，使用 [Black](https://github.com/psf/black) 格式化器自动化格式化过程：

```bash
black SimpleLLMFunc tests
```

### 类型注解

我们鼓励使用类型注解以提高代码可读性和安全性。可以使用 Pylint 检查类型。

### 文档

- 所有公共 API 都应该有清晰的文档字符串
- 文档注释应遵循 [Google Python 文档风格](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
- 更新功能时，请同时更新相关文档


<!-- TODO: 版本控制与 semver 并不一致
## 版本控制和发布
我们使用 [Semantic Versioning](https://semver.org/) 进行版本控制：

- MAJOR 版本号：不向后兼容的 API 更改
- MINOR 版本号：向后兼容的功能添加
- PATCH 版本号：向后兼容的错误修复

每次提交应包含适当的提交消息，推荐遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
-->

## 行为准则

请尊重所有项目参与者，保持友好的交流环境。任何形式的骚扰或冒犯行为都是不可接受的。

## 获取帮助

如果你在贡献过程中需要帮助，可以：

- 在 GitHub Issues 中提问
- 联系项目维护者

再次感谢你对 SimpleLLMFunc 的贡献！
