# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SimpleLLMFunc'
copyright = '2025, Nijingzhe'
author = 'Nijingzhe'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = []   # type: ignore

# 国际化配置
# locale_dirs：指定翻译文件（.po）的目录位置
locale_dirs = ['../locale/']
# gettext_compact：False 表示为每个文件创建单独的 .po 文件
gettext_compact = False
# 指定需要翻译的目标
gettext_additional_targets = [
    'index',
    'introduction',
    'quickstart',
    'guide',
    'examples',
    'contributing',
    'langfuse_integration',
    'detailed_guide/config',
    'detailed_guide/llm_chat',
    'detailed_guide/llm_function',
    'detailed_guide/llm_interface',
    'detailed_guide/tool',
]

# 支持的语言
# 默认语言设置为中文
language = 'zh_CN'

# 语言显示名称映射
languages = {
    'zh_CN': '中文（简体）',
    'en': 'English',
}

# ReadTheDocs 环境检测
import os
is_readthedocs = os.environ.get('READTHEDOCS') == 'True'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# HTML主题选项
html_theme_options = {
    'analytics_id': '',  # 可选：Google Analytics ID
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# HTML上下文配置 - ReadTheDocs 多语言支持
html_context = {
    # 支持的语言列表
    'languages': languages,
    # 当前使用的语言
    'current_language': language,
    # 当前版本
    'current_version': 'latest',
    # 版本映射
    'versions': {
        'latest': 'latest',
    },
    # 在 ReadTheDocs 上显示语言切换器
    'display_version': True,
}

# 如果在 ReadTheDocs 上构建，确保使用正确的 locale 路径
if is_readthedocs:
    # ReadTheDocs 会自动处理多语言构建
    pass
