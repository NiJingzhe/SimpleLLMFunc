from __future__ import annotations

from scripts.translate_mintlify_docs import (
    build_initial_translation_cache,
    build_placeholder_template,
    fill_placeholders,
    prefix_internal_routes,
)


class TestBuildPlaceholderTemplate:
    def test_extracts_frontmatter_headings_attrs_and_text(self) -> None:
        source = """---
title: 快速开始
---

## 入门

<Card title=\"项目介绍\" href=\"/introduction\">
  查看项目概览。
</Card>

- 使用指南

```python
print(\"不要翻译代码\")
```
"""

        template, units = build_placeholder_template(source)

        assert "__SLMF_I18N_0__" in template
        assert "__SLMF_I18N_1__" in template
        assert 'title="__SLMF_I18N_' in template
        assert 'print("不要翻译代码")' in template
        assert units == [
            "快速开始",
            "入门",
            "项目介绍",
            "查看项目概览。",
            "使用指南",
        ]

    def test_preserves_tag_only_lines(self) -> None:
        source = """<Steps>
<Step title=\"环境准备\">
内容
</Step>
</Steps>
"""

        template, units = build_placeholder_template(source)

        assert "<Steps>" in template
        assert "</Steps>" in template
        assert "</Step>" in template
        assert units == ["环境准备", "内容"]

    def test_splits_table_cells_instead_of_whole_row(self) -> None:
        source = "| 功能 | 文档 | 说明 |\n|-----|------|------|\n| 基础配置 | [配置](/detailed_guide/config) | API 密钥 |\n"

        template, units = build_placeholder_template(source)

        assert template.splitlines()[0].count("__SLMF_I18N_") == 3
        assert template.splitlines()[1] == "|-----|------|------|"
        assert template.splitlines()[2].count("__SLMF_I18N_") == 3
        assert units == [
            "功能",
            "文档",
            "说明",
            "基础配置",
            "[配置](/detailed_guide/config)",
            "API 密钥",
        ]

    def test_handles_inline_jsx_with_attributes_and_text(self) -> None:
        source = '<Card title="类型推断">自动从函数签名中提取参数类型和说明。</Card>\n'

        template, units = build_placeholder_template(source)

        assert 'title="__SLMF_I18N_0__"' in template
        assert ">__SLMF_I18N_1__</Card>" in template
        assert units == ["类型推断", "自动从函数签名中提取参数类型和说明。"]


class TestTranslationMemoryFill:
    def test_prefers_page_memory_then_global_memory(self) -> None:
        texts = ["快速开始", "项目介绍", "不存在的句子"]
        cache, missing = build_initial_translation_cache(
            texts=texts,
            page_memory={"快速开始": "Quick Start"},
            global_memory={"项目介绍": "Project Introduction"},
            override_memory={},
            source_lang="zh_CN",
            target_lang="en",
        )

        assert cache["快速开始"] == "Quick Start"
        assert cache["项目介绍"] == "Project Introduction"
        assert missing == ["不存在的句子"]

    def test_override_memory_has_highest_priority(self) -> None:
        cache, missing = build_initial_translation_cache(
            texts=["快速开始"],
            page_memory={"快速开始": "Page Memory"},
            global_memory={"快速开始": "Global Memory"},
            override_memory={"快速开始": "Quick Start"},
            source_lang="zh_CN",
            target_lang="en",
        )

        assert cache["快速开始"] == "Quick Start"
        assert missing == []

    def test_fill_placeholders_uses_cache(self) -> None:
        template = "# __SLMF_I18N_0__\n\n- __SLMF_I18N_1__\n"
        units = ["快速开始", "项目介绍"]
        rendered = fill_placeholders(
            template,
            units,
            {
                "快速开始": "Quick Start",
                "项目介绍": "Project Introduction",
            },
        )

        assert rendered == "# Quick Start\n\n- Project Introduction\n"


class TestPrefixInternalRoutes:
    def test_prefixes_internal_routes_but_not_assets_or_external_links(self) -> None:
        source = """
<Card title="Quickstart" href="/quickstart" />
<Card title="Logo" href="/img/favicon.png" />
[Guide](/guide)
[External](https://example.com)
"""

        rendered = prefix_internal_routes(source, "en")

        assert 'href="/en/quickstart"' in rendered
        assert 'href="/img/favicon.png"' in rendered
        assert "[Guide](/en/guide)" in rendered
        assert "[External](https://example.com)" in rendered

    def test_does_not_double_prefix_existing_locale_routes(self) -> None:
        source = '[Guide](/en/guide)\n<Card title="Intro" href="/en/introduction" />\n'

        rendered = prefix_internal_routes(source, "en")

        assert rendered == source
