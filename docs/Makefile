# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -c .
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
# 翻译文件目录（与 conf.py 中的 locale_dirs 保持一致）
LOCALEDIR     = source/locale

# 支持的语言列表
LANGUAGES     = zh_CN en

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  clean         - Remove build directory"
	@echo "  html          - Build HTML documentation"
	@echo "  html-all      - Build HTML for all languages"
	@echo "  gettext       - Extract translatable strings"
	@echo "  update-po     - Update .po files from .pot files"
	@echo "  compile-po    - Compile .po files to .mo files"
	@echo "  serve         - Serve documentation locally"
	@echo ""
	@echo "Language-specific targets:"
	@for lang in $(LANGUAGES); do \
		echo "  html-$$lang    - Build HTML for $$lang"; \
	done

.PHONY: help Makefile clean html html-all gettext update-po compile-po serve

# 清理构建目录
clean:
	rm -rf $(BUILDDIR)/*

# 提取可翻译字符串
gettext:
	$(SPHINXBUILD) -b gettext $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/gettext

# 更新.po文件
update-po: gettext
	@for lang in $(LANGUAGES); do \
		echo "Updating $$lang.po..."; \
		sphinx-intl update -p $(BUILDDIR)/gettext -l $$lang -d $(LOCALEDIR); \
	done

# 编译.po文件为.mo文件
compile-po:
	@for lang in $(LANGUAGES); do \
		echo "Compiling $$lang..."; \
		sphinx-intl build -d $(LOCALEDIR) -l $$lang; \
	done

# 构建单个语言的HTML文档
html-%: compile-po
	@lang=$$(echo $* | sed 's/html-//'); \
	echo "Building HTML for $$lang..."; \
	$(SPHINXBUILD) -b html -D language=$$lang $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html/$$lang $(O)

# 构建所有语言的HTML文档
html-all: compile-po
	@for lang in $(LANGUAGES); do \
		echo "Building HTML for $$lang..."; \
		$(SPHINXBUILD) -b html -D language=$$lang $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html/$$lang $(O); \
	done

# 默认构建中文文档
html: html-zh_CN

# 本地服务（用于测试）
serve: html-all
	@echo "Serving documentation at http://localhost:8000"
	@echo "Available languages:"
	@for lang in $(LANGUAGES); do \
		echo "  http://localhost:8000/html/$$lang/"; \
	done
	@cd $(BUILDDIR)/html && python3 -m http.server 8000

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
