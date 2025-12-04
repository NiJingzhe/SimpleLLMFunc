#!/bin/bash
source /Users/lildino/Library/Caches/pypoetry/virtualenvs/simplellmfunc-v9Mcp8Tm-py3.13/bin/activate

# 根目录文件
for file in ../docs/source/locale/zh/LC_MESSAGES/*.po; do
    basename=$(basename "$file")
    echo "Translating $basename..."
    python translate_po.py "$file" -o "../docs/source/locale/en/LC_MESSAGES/$basename" -b 20 -c 100
done

# detailed_guide子目录文件
for file in ../docs/source/locale/zh/LC_MESSAGES/detailed_guide/*.po; do
    basename=$(basename "$file")
    echo "Translating detailed_guide/$basename..."
    python translate_po.py "$file" -o "../docs/source/locale/en/LC_MESSAGES/detailed_guide/$basename" -b 20 -c 100
done

echo "All translations completed!"
