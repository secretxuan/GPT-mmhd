#!/usr/bin/env python3
"""
下载真实的中文数据集

这个脚本帮你下载可用于训练的中文数据
"""

import os
import urllib.request

def download_file(url, output_path, desc=""):
    """下载文件"""
    print(f"正在下载: {desc or url}")
    print(f"保存到: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path)
        print("下载完成!")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("GPT 训练数据下载工具")
    print("=" * 60)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("""
你有几个选择：

1. 使用 Hugging Face 数据集（推荐）
   需要先安装: pip install datasets

   然后运行:
   from datasets import load_dataset
   ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

   # 保存为文本
   with open('wikitext_large.txt', 'w', encoding='utf-8') as f:
       for item in ds:
           f.write(item['text'] + '\\n')

   print(f'保存了 {len(ds)} 条文本')

2. 手动下载（如果网络不好）
   - 去这个网站下载中文小说:
   - 保存为 UTF-8 编码的 txt 文件
   - 放在 data/ 目录下

3. 使用本地数据
   如果你有自己的文本文件，直接放在 data/ 目录下

推荐数据量：
- 快速测试: 至少 100KB
- 看到效果: 至少 10MB
- 更好效果: 至少 100MB+
    """)

    print("\n" + "=" * 60)
    print("当前已有数据:")
    print("=" * 60)

    wikitext_path = os.path.join(data_dir, 'wikitext.txt')
    if os.path.exists(wikitext_path):
        size = os.path.getsize(wikitext_path) / 1024
        print(f"wikitext.txt: {size:.1f} KB (太小，只能测试代码)")
    else:
        print("wikitext.txt: 不存在")


if __name__ == "__main__":
    main()
