# 双语文档配置说明

## 快速概览

SimpleLLMFunc 文档已配置支持**中文（简体）和英文**两种语言。

## 配置文件总览

| 文件 | 用途 | 状态 |
|------|------|------|
| `.readthedocs.yaml` | ReadTheDocs 构建配置 | ✅ 已配置 |
| `docs/source/conf.py` | Sphinx 配置 | ✅ 已配置 |
| `docs/locale/en/LC_MESSAGES/*.po` | 英文翻译文件 | ✅ 已生成 |
| `docs/locale/zh_CN/LC_MESSAGES/*.po` | 中文翻译文件 | ✅ 已生成 |

## 文件生成情况

已为以下文档生成翻译文件：

**核心文档:**
- index.po (索引)
- introduction.po (项目介绍)
- quickstart.po (快速开始)
- guide.po (使用指南)
- examples.po (示例代码)
- contributing.po (贡献指南)
- langfuse_integration.po (Langfuse 集成)

**详细指南:**
- detailed_guide/config.po (配置)
- detailed_guide/llm_chat.po (LLM 聊天)
- detailed_guide/llm_function.po (LLM 函数)
- detailed_guide/llm_interface.po (LLM 接口)
- detailed_guide/tool.po (工具)

## 当前翻译状态

### 英文翻译 (docs/locale/en/LC_MESSAGES/)
- ⚠️ **未完成** - 所有 .po 文件都需要翻译
- msgstr 字段都为空，需要手动填写英文翻译

### 中文翻译 (docs/locale/zh_CN/LC_MESSAGES/)
- ⚠️ **未完成** - 中文是源语言，这些文件主要用于维护翻译ID

## ReadTheDocs 部署步骤

### 1. 项目设置

1. 登录 [ReadTheDocs](https://readthedocs.org)
2. 导入项目或进入已有项目
3. 转到 **Admin** → **Internationalization**

### 2. 启用多语言

在 ReadTheDocs 项目设置中：

```yaml
# 已在 .readthedocs.yaml 中配置
languages:
  - code: en
    name: English
  - code: zh_CN
    name: 中文（简体）
```

### 3. 构建结果

ReadTheDocs 会自动为每种语言创建独立的文档网站：

- **中文**: `https://your-project.readthedocs.io/zh_CN/latest/`
- **英文**: `https://your-project.readthedocs.io/en/latest/`

## 本地开发命令

### 生成翻译文件

当源文档更新时，重新生成 .po 文件：

```bash
cd docs
make clean
make gettext
sphinx-intl update -p build/gettext -l zh_CN -l en
```

### 编译翻译

将 .po 文件编译为 .mo 文件（ReadTheDocs 或本地构建时需要）：

```bash
cd docs
sphinx-intl build
```

### 本地测试（英文）

```bash
cd docs
make -e SPHINXOPTS="-D language=en" html
# 或使用环境变量
LANGUAGE=en make html
```

### 本地测试（中文）

```bash
cd docs
make html  # 默认使用中文
```

## 文件说明

### .po 文件结构

每个 .po 文件包含若干翻译条目：

```po
# 注释行，标注源位置
#: ../../source/index.md:5
msgid "原始文本（中文）"
msgstr "翻译文本（目标语言）"
```

**翻译任务：** 将所有 `msgstr ""` （空翻译）改为对应的英文翻译。

### 示例（英文翻译）

```po
# docs/locale/en/LC_MESSAGES/index.po

#: ../../source/index.md:1
msgid "SimpleLLMFunc documentation"
msgstr "SimpleLLMFunc Documentation"

#: ../../source/index.md:3
msgid "SimpleLLMFunc 是一个轻量级、可配置的 LLM 应用开发框架。"
msgstr "SimpleLLMFunc is a lightweight, configurable LLM application development framework."

#: ../../source/index.md:5
msgid "快速开始"
msgstr "Quick Start"
```

## 翻译编辑工具

推荐使用以下工具编辑 .po 文件：

1. **Poedit** - 专业 .po 文件编辑器
   - 支持跨平台 (Windows, macOS, Linux)
   - 用户友好的图形界面
   - 提供翻译统计和质量检查

2. **VS Code** - 带 Gettext 扩展
   - 扩展: "Gettext" 或 "PO file support"
   - 免费开源

3. **文本编辑器** - 直接编辑
   - 支持 UTF-8 编码
   - 编辑 `msgstr ""` 字段

## 问题排查

### 问题: 本地测试英文翻译不显示

**解决方案:**
```bash
# 确保编译了翻译文件
cd docs
sphinx-intl build

# 清理构建缓存后重新构建
make clean
make -e SPHINXOPTS="-D language=en" html
```

### 问题: ReadTheDocs 没有显示语言切换

**检查清单:**
1. ✅ `.readthedocs.yaml` 包含 `languages` 配置
2. ✅ `docs/locale/en/LC_MESSAGES/` 有 .po 文件
3. ✅ ReadTheDocs 重新构建项目
4. ✅ 检查 ReadTheDocs 项目设置中的多语言选项

### 问题: Git 提交时忽略翻译文件

翻译文件应该提交到 Git，确保 `.gitignore` 没有忽略 `docs/locale/` 目录：

```bash
# 检查是否被忽略
git check-ignore docs/locale/

# 强制添加
git add -f docs/locale/
```

## 维护建议

1. **定期同步翻译**
   - 每次修改 `docs/source/` 中的文档后，运行生成翻译命令
   - 保持 .po 文件与源文档同步

2. **使用翻译管理平台**
   - 可选：集成 Transifex、Crowdin 等翻译管理工具
   - 适合大型项目和社区贡献翻译

3. **版本控制**
   - 将翻译文件提交到 Git
   - 跟踪翻译的变更历史

4. **文档一致性**
   - 维护术语表，确保翻译术语的一致性
   - 在 GitHub Issues 中讨论翻译标准

## 相关资源

- [Sphinx 国际化指南](https://www.sphinx-doc.org/en/master/usage/advanced/intl.html)
- [ReadTheDocs 文档](https://docs.readthedocs.io/)
- [sphinx-intl 工具](https://pypi.org/project/sphinx-intl/)
- [gettext 格式](https://www.gnu.org/software/gettext/)

---

**配置日期**: 2025-10-27
**状态**: 双语框架已建立，等待英文翻译完成
