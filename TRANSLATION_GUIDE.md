# 多语言翻译工作流程指南

本项目已配置双语支持（中文和英文）。本指南说明如何管理和更新翻译。

## 项目结构

```
docs/
├── source/              # 源文档（中文）
│   ├── conf.py
│   ├── index.md
│   ├── introduction.md
│   ├── quickstart.md
│   ├── guide.md
│   ├── examples.md
│   ├── contributing.md
│   ├── langfuse_integration.md
│   └── detailed_guide/
│       ├── config.md
│       ├── llm_chat.md
│       ├── llm_function.md
│       ├── llm_interface.md
│       └── tool.md
├── locale/              # 翻译文件目录
│   ├── en/              # 英文翻译
│   │   └── LC_MESSAGES/
│   │       ├── index.po
│   │       ├── introduction.po
│   │       └── ... (其他 .po 文件)
│   └── zh_CN/           # 中文翻译（如需）
│       └── LC_MESSAGES/
│           └── ... (中文 .po 文件)
└── build/               # 构建输出目录
    └── gettext/         # 提取的翻译模板
```

## 翻译文件格式（.po 文件）

.po（Portable Object）文件是标准的翻译格式。每个文件包含：

```po
#: ../../source/index.md:5
msgid "原始文本（中文）"
msgstr "翻译文本（目标语言）"
```

**字段说明：**
- `#:` - 标注源文件和行号
- `msgid` - 原始文本（从源文档提取）
- `msgstr` - 翻译文本（需要翻译者填写）

## 工作流程

### 1. 更新文档后生成新的翻译文件

当您修改或添加 `docs/source/` 中的文档时：

```bash
# 进入 docs 目录
cd docs

# 清理旧的构建文件
make clean

# 生成新的翻译模板
make gettext

# 更新所有语言的 .po 文件
sphinx-intl update -p build/gettext -l zh_CN -l en
```

**说明：**
- `make gettext` 会从源文档提取所有可翻译的字符串
- `sphinx-intl update` 会同步新的字符串到 .po 文件
- 已翻译的字符串保留，新字符串添加为空翻译条目

### 2. 翻译 .po 文件

编辑 `docs/locale/en/LC_MESSAGES/` 下的 .po 文件，将 `msgstr` 字段翻译为英文：

**示例：**

```po
# 翻译前（msgstr 为空）
#: ../../source/index.md:5
msgid "快速开始"
msgstr ""

# 翻译后
#: ../../source/index.md:5
msgid "快速开始"
msgstr "Quick Start"
```

**编辑工具推荐：**
- **Poedit** (图形界面，跨平台)
- **Visual Studio Code** + Gettext 扩展
- **任何文本编辑器**（直接编辑）

### 3. 编译翻译文件

翻译完成后，将 .po 文件编译成 .mo（Machine Object）文件：

```bash
cd docs
sphinx-intl build
```

这会生成 `.mo` 文件在 `docs/locale/<language>/LC_MESSAGES/` 下。

### 4. 本地测试翻译

```bash
# 进入 docs 目录
cd docs

# 使用特定语言构建文档
# 为英文构建
make -e SPHINXOPTS="-D language=en" html

# 为中文构建（默认）
make html
```

输出在 `docs/build/html/` 目录中。

### 5. 部署到 ReadTheDocs

提交更新的翻译文件后，ReadTheDocs 会自动：

1. **检测** `.readthedocs.yaml` 中的语言配置
2. **为每个语言单独构建** 文档网站
3. **提供语言切换器** 在文档页面上

**ReadTheDocs 多语言 URL 结构：**
- 默认（中文）: `https://your-project.readthedocs.io/zh_CN/latest/`
- 英文: `https://your-project.readthedocs.io/en/latest/`

## 翻译内容检查清单

- [ ] 打开 `docs/locale/en/LC_MESSAGES/` 文件夹
- [ ] 检查每个 .po 文件中是否有未翻译的 `msgstr ""`
- [ ] 翻译所有 `msgstr` 为空的条目
- [ ] 保留代码块和特殊格式不变
- [ ] 测试本地构建：`make -e SPHINXOPTS="-D language=en" html`
- [ ] 验证输出 HTML 中的翻译效果
- [ ] 提交翻译更新

## 常见问题

### Q: 如何处理翻译中的代码块或特殊语法？

A: 不要翻译代码块、变量名、函数名等。翻译时保持原样：

```po
msgid "使用 `llm.chat()` 函数"
msgstr "Use the `llm.chat()` function"
```

### Q: 如何添加新的文档文件？

A:
1. 在 `docs/source/` 下创建新的 .md 文件
2. 运行 `make gettext && sphinx-intl update -p build/gettext -l zh_CN -l en`
3. 翻译新生成的 .po 文件
4. 运行 `sphinx-intl build` 编译翻译

### Q: 翻译后本地看不到效果？

A: 确保执行了以下步骤：
1. 编译翻译文件：`sphinx-intl build`
2. 使用正确语言构建：`make -e SPHINXOPTS="-D language=en" html`
3. 清理缓存：`make clean` 后重新构建

### Q: ReadTheDocs 上没有显示语言切换器？

A: 检查以下项：
1. `.readthedocs.yaml` 是否包含 `languages` 配置
2. 至少有一个 .po 翻译文件存在（非空）
3. ReadTheDocs 项目设置中是否启用了多语言
4. 等待 ReadTheDocs 重新构建项目

## 参考资源

- [Sphinx 国际化文档](https://www.sphinx-doc.org/en/master/usage/advanced/intl.html)
- [sphinx-intl 工具](https://pypi.org/project/sphinx-intl/)
- [ReadTheDocs 多语言项目](https://docs.readthedocs.io/en/stable/guides/build-using-os-commands.html#internationalization)
- [gettext .po 文件格式](https://www.gnu.org/software/gettext/manual/gettext.html#PO-Files)

## 配置文件检查清单

已配置的文件：

- ✅ `.readthedocs.yaml` - 包含 `languages` 配置
- ✅ `docs/source/conf.py` - 国际化配置
- ✅ `docs/locale/` - 翻译文件目录
- ✅ `docs/locale/en/LC_MESSAGES/*.po` - 英文翻译文件
- ✅ `docs/locale/zh_CN/LC_MESSAGES/*.po` - 中文翻译文件

## 维护提示

1. **定期更新翻译** - 每次文档变更后更新 .po 文件
2. **使用版本控制** - 将 .po 文件提交到 Git
3. **保持一致性** - 使用术语表维持翻译的一致性
4. **自动化检查** - 可以设置 GitHub Actions 检查翻译的完整性

---

**最后更新**: 2025-10-27
**项目**: SimpleLLMFunc
**当前支持的语言**: 中文（简体）、English
