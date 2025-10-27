# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'SimpleLLMFunc'
copyright = '2025, Nijingzhe'
author = 'Nijingzhe'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]


language = os.environ.get('SPHINX_LANGUAGE', 'en')

# 国际化配置
locale_dirs = ['locale/']
gettext_compact = False

# 语言显示名称映射
languages = {
    'zh_CN': '中文（简体）',
    'en': 'English',
}


source_suffix = [ '.md', '.rst' ]
master_doc = 'index'

# ReadTheDocs 环境检测
is_readthedocs = os.environ.get('READTHEDOCS') == 'True'

# 如果在ReadTheDocs环境中，使用环境变量设置的语言
if is_readthedocs:
    language = os.environ.get('READTHEDOCS_LANGUAGE', 'en')

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# HTML主题选项
html_theme_options = {
    'analytics_id': '',  # 可选：Google Analytics ID
    'logo_only': False,
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

# HTML上下文配置
html_context = {
    'current_version': 'latest',
    'versions': {
        'latest': 'latest',
    },
    'display_version': True,
}


