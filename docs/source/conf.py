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
locale_dirs = ['../locale/']   # 翻译文件目录
gettext_compact = False        # 支持多个域
gettext_additional_targets = [
    'index',
    'introduction',
    'quickstart',
    'guide',
    'examples',
    'langfuse_integration',
]

# 支持的语言
language = 'zh_CN'
languages = {
    'zh_CN': '中文',
    'en': 'English',
}

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

# 多语言支持
html_context = {
    'languages': languages,
    'current_language': language,
    'current_version': 'latest',
    'versions': {
        'latest': 'latest',
    }
}
