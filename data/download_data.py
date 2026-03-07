"""
=================================================================================
数据集下载脚本
=================================================================================

从 Hugging Face 或其他来源下载真正有用的中文预训练数据。

推荐数据集：
1. wiki-zh: 中文维基百科 (~1GB) - 质量高，通用性强
2. chinese-novels: 中文小说 - 适合你的小说续写任务
3. CSL: 中文科学文献 - 学术风格

使用方法：
    python download_data.py --dataset wiki
    python download_data.py --dataset novel
    python download_data.py --dataset all

参考资源：
- https://huggingface.co/datasets/yuyijiong/LongData-Corpus
- https://huggingface.co/datasets/shibing624/nli-zh-all
- https://github.com/HqWu-HITCS/Awesome-Chinese-LLM
"""

import os
import argparse
import urllib.request
import gzip
import shutil


def download_file(url, output_path, desc="下载中"):
    """下载文件并显示进度"""
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")

    def report_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        print(f"\r{desc}: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, report_hook)
    print("\n下载完成！")


def download_wiki_zh(output_dir):
    """
    下载中文维基百科数据

    数据来源：Hugging Face / 其他开源
    大小：约 1GB 压缩后
    """
    print("\n" + "="*60)
    print("下载中文维基百科数据")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # 维基百科数据通常在 Hugging Face 上
    # 这里提供一个简化的示例
    print("""
推荐方式：使用 Hugging Face datasets 库

    pip install datasets
    python -c "
from datasets import load_dataset
ds = load_dataset('wikipedia', '20220301.zh', trust_remote_code=True)
ds['train'].to_json('wiki_zh.jsonl')
    "

或者直接下载预处理好的：
    https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered

这个数据集约 1GB，包含百科知识，质量很高。
    """)


def download_chinese_novels(output_dir):
    """
    下载中文小说数据集

    适合小说续写任务！
    """
    print("\n" + "="*60)
    print("下载中文小说数据集")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    print("""
推荐方式：

1. Hugging Face 中文小说数据集：
   https://huggingface.co/datasets/mvwqrst/chinese-novels

   pip install datasets
   python -c "
from datasets import load_dataset
ds = load_dataset('mvwqrst/chinese-novels')
# 保存为文本
with open('novels.txt', 'w', encoding='utf-8') as f:
    for item in ds['train']:
        f.write(item['text'] + '\\n')
   "

2. 或者下载红楼求数据（更大更全）：
   https://huggingface.co/datasets/opencsg/chinese-novel-collection

3. 古典小说集合：
   https://github.com/HqWu-HITCS/Awesome-Chinese-LLM
   里面有很多中文数据集链接
    """)


def download_csl(output_dir):
    """
    下载中文科学文献数据集

    适合学术风格文本
    """
    print("\n" + "="*60)
    print("下载中文科学文献数据集")
    print("="*60)

    print("""
推荐：CSL (Chinese Scientific Literature)

    pip install datasets
    python -c "
from datasets import load_dataset
ds = load_dataset('shibing624/nli-zh-all', split='train')
ds.to_json('csl.jsonl')
    "

包含 39 万篇中文核心期刊论文摘要
适合学习正式、学术风格的写作
    """)


def download_quick_sample(output_dir):
    """
    快速下载一个示例数据（用于测试）

    这会下载一个小的中文文本样本，可以立即开始训练
    """
    print("\n" + "="*60)
    print("下载快速示例数据")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample.txt")

    # 创建一个示例文本（如果网络下载失败）
    sample_text = """
红楼梦

第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀

此开卷第一回也。作者自云：因曾历过一番梦幻之后，故将真事隐去，
而借通灵之说，撰此《石头记》一书也。故曰甄士隐云云。

列位看官：你道此书从何而来？说起根由虽近荒唐，细按则深有趣味。

原来女娲氏炼石补天之时，于大荒山无稽崖练成高经十二丈、
方经二十四丈顽石三万六千五百零一块。娲皇氏只用了三万六千五百块，
只单单剩了一块未用，便弃在此山青埂峰下。

谁知此石自经煅炼之后，灵性已通，因见众石俱得补天，
独自己无材不堪入选，遂自怨自叹，日夜悲号惭愧。

第二回 贾夫人仙逝扬州城 冷子兴演说荣国府

话说封肃因听见公差传唤，忙出来陪笑启问。
那些人只嚷：快请出甄爷来！

第三回 贾雨村夤缘复旧职 林黛玉抛父进京都

却说雨村忙回头看时，不是别人，乃是当日同僚一案参革的号张如圭者。
他本系此地人，革后家居，今打听得都中奏准起复旧员之信，
他便四下里寻情找门路，忽遇见雨村，故忙道喜。

林黛玉常听得母亲说过，他外祖母家与别家不同。
他近日所见的这几个三等仆妇，吃穿用度，已是不凡了，何况今至其家。
因此步步留心，时时在意，不肯轻易多说一句话，多行一步路，
唯恐被人耻笑了他去。

黛玉方进入房时，只见两个人搀着一位鬓发如银的老母迎上来，
黛玉便知是他外祖母。方欲拜见时，早被他外祖母一把搂入怀中，
心肝儿肉叫着大哭起来。

第四回 薄命女偏逢薄命郎 葫芦僧乱判葫芦案

却说黛玉同姊妹们至王夫人处，见了王夫人，说起薛家之事。
王夫人道：你舅舅今日斋戒去了，可以不必过去。
我们这里有一个薛家，也是亲戚，如今也要进京。

第五回 游幻境指迷十二钗 饮仙醪曲演红楼梦

第四日早起，贾宝玉梳洗已毕，换了冠带，来至王夫人房中。
只见王夫人与黛玉、宝钗、湘云、探春、惜春、李纨、熙凤等都在那里。

第六回 贾宝玉初试云雨情 刘姥姥一进荣国府

却说秦氏因听见宝玉从梦中唤他的乳名，心中自是纳闷，
又不好细问，彼时宝玉迷迷惑惑，若有所失。
众人忙端桂圆汤来，呷了两口，遂起身整衣。

第七回 送宫花贾琏戏熙凤 宴宁府宝玉会秦钟

话说周瑞家的送了刘姥姥去后，便上来回王夫人话。
谁知王夫人不在上房，问丫鬟们时，方知往薛姨妈那边闲话去了。

第八回 比通灵金莺微露意 探宝钗黛玉半含酸

话说宝玉因送贾母回来，又至王夫人处请了安。
因想起宝钗近日身上不好，便要去瞧他。

第九回 恋风流情友入家塾 起嫌疑顽童闹学堂

话说秦业父子专候贾家的人来送上学之信。
原来宝玉急于要和秦钟相遇，却顾不得别的，遂择了后日一定上学。

第十回 金寡妇贪利权受辱 张太医论病细穷源

话说金荣因人多势众，又兼贾瑞勒令，赔了不是，给秦钟磕了头，
宝玉方才不吵闹了。大家散了学，金荣回到家中，越想越气。

话说黛玉见宝玉去了，又听见众姊妹也不在房中，
自己闷闷的，正欲回房，刚走到梨香院墙角外，
只听见墙内笛韵悠扬，歌声婉转。

黛玉便知是那十二个女孩子演习戏文呢。
只是林黛玉素习不大喜看戏文，便不留心，只管往前走。

偶然两句吹到耳内，明明白白，一字不落，
唱道是：原来姹紫嫣红开遍，似这般都付与断井颓垣。
黛玉听了，倒也十分感慨缠绵，便止住步侧耳细听。

又听唱道是：良辰美景奈何天，赏心乐事谁家院。
听了这两句，不觉点头自叹，心下自思道：
原来戏上也有好文章。可惜世人只知看戏，未必能领略这其中的趣味。

想毕，又后悔不该胡想，耽误了听曲子。
又侧耳时，只听唱道：则为你如花美眷，似水流年。
黛玉听了这两句，不觉心动神摇。

又听道：你在幽闺自怜等句，亦发如醉如痴，
站立不住，便一蹲身坐在一块山子石上，
细嚼如花美眷，似水流年八个字的滋味。

忽又想起前日见古人诗中有水流花谢两无情之句，
再又有词中有流水落花春去也，天上人间之句，
又兼方才所见西厢记中花落水流红，闲愁万种之句，
都一时想起来，凑聚在一处。

仔细忖度，不觉心痛神痴，眼中落泪。
正没个开交，忽觉背上击了一下，及回头看时，
原来是且住，且住，且听下回分解。
""" * 10  # 重复10次增加数据量

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    print(f"示例数据已保存到: {output_path}")
    print(f"数据大小: {len(sample_text)} 字符")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='下载中文预训练数据')
    parser.add_argument('--dataset', type=str, default='sample',
                        choices=['wiki', 'novel', 'csl', 'sample', 'all'],
                        help='要下载的数据集')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='输出目录')

    args = parser.parse_args()

    print("="*60)
    print("GPT 训练数据下载工具")
    print("="*60)

    if args.dataset == 'sample':
        download_quick_sample(args.output_dir)
    elif args.dataset == 'wiki':
        download_wiki_zh(args.output_dir)
    elif args.dataset == 'novel':
        download_chinese_novels(args.output_dir)
    elif args.dataset == 'csl':
        download_csl(args.output_dir)
    elif args.dataset == 'all':
        download_quick_sample(args.output_dir)
        download_wiki_zh(args.output_dir)
        download_chinese_novels(args.output_dir)

    print("\n" + "="*60)
    print("下载说明")
    print("="*60)
    print("""
对于学习 GPT 原理，推荐：

1. 快速开始（推荐新手）：
   python download_data.py --dataset sample
   然后用 tiny 配置训练

2. 想要更好的效果：
   先安装: pip install datasets
   然后:
   from datasets import load_dataset
   ds = load_dataset('mvwqrst/chinese-novels')

3. 完整训练：
   需要至少 10GB+ 的数据
   建议用 GPU 服务器

数据集来源：
- [yuyijiong/LongData-Corpus](https://huggingface.co/datasets/yuyijiong/LongData-Corpus)
- [Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)
- [shibing624/nli-zh-all](https://huggingface.co/datasets/shibing624/nli-zh-all)
    """)


if __name__ == "__main__":
    main()
