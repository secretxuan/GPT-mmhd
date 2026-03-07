#!/usr/bin/env python3
"""
下载 wikitext 数据集用于训练

wikitext 是一个高质量的文本数据集，常用于语言模型训练。
虽然它是英文的，但对于学习 GPT 原理来说足够了。

使用方法：
    python download_wikitext.py
"""

import os
import urllib.request
import gzip


def download_wikitext(output_dir="."):
    """下载 wikitext-2 数据集"""

    os.makedirs(output_dir, exist_ok=True)

    # wikitext-2-raw 数据集 URL
    base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"

    files = {
        "train.txt": "train.txt",
        "valid.txt": "valid.txt",
        "test.txt": "test.txt"
    }

    for filename in files:
        url = f"{base_url}/{filename}"
        output_path = os.path.join(output_dir, f"wikitext_{filename}")

        print(f"下载 {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            # 读取并显示文件大小
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"  保存到: {output_path}")
            print(f"  大小: {len(content)} 字符")
            print(f"  行数: {len(content.splitlines())} 行")
        except Exception as e:
            print(f"  下载失败: {e}")

    # 合并所有训练数据
    train_path = os.path.join(output_dir, "wikitext_train.txt")
    valid_path = os.path.join(output_dir, "wikitext_valid.txt")

    # 读取并合并
    all_text = []
    for fname in ["wikitext_train.txt", "wikitext_valid.txt"]:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_text.append(f.read())

    merged_path = os.path.join(output_dir, "wikitext_all.txt")
    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    print(f"\n合并后的数据: {merged_path}")
    with open(merged_path, 'r') as f:
        print(f"总大小: {len(f.read())} 字符")

    return merged_path


def create_chinese_sample(output_dir="."):
    """创建一个中文样本数据（更大的红楼梦片段）"""

    # 使用公共领域的红楼梦文本（更长的版本）
    # 这里我们创建一个足够大的样本用于训练
    sample_text = """
红楼梦

第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀

此开卷第一回也。作者自云：因曾历过一番梦幻之后，故将真事隐去，而借"通灵"之说，撰此《石头记》一书也。故曰"甄士隐"云云。但书中所记何事何人？自又云："今风尘碌碌，一事无成，忽念及当日所有之女子，一一细考较去，觉其行止见识，皆出于我之上。何我堂堂须眉，诚不若彼裙钗哉？实愧则有余，悔又无益之大无可如何之日也！当此，则自欲将已往所赖天恩祖德，锦衣纨绔之时，饫甘餍肥之日，背父兄教育之恩，负师友规训之德，以至今日一技无成，半生潦倒之罪，编述一集，以告天下人：我之罪固不免，然闺阁中本自历历有人，万不可因我之不肖，自护己短，一并使其泯灭也。虽今日之茅椽蓬牖，瓦灶绳床，其晨夕风露，阶柳庭花，亦未有妨我之襟怀笔墨者。虽我未学，下笔无文，又何妨用假语村言，敷演出一段故事来，亦可使闺阁昭传，复可悦世之目，破人愁闷，不亦宜乎？故曰'贾雨村'云云。"

此回中凡用"梦"用"幻"等字，是提醒阅者眼目，亦是此书立意本旨。

列位看官：你道此书从何而来？说起根由虽近荒唐，细按则深有趣味。待在下将此来历注明，方使阅者了然不惑。

原来女娲氏炼石补天之时，于大荒山无稽崖练成高经十二丈、方经二十四丈顽石三万六千五百零一块。娲皇氏只用了三万六千五百块，只单单剩了一块未用，便弃在此山青埂峰下。谁知此石自经煅炼之后，灵性已通，因见众石俱得补天，独自己无材不堪入选，遂自怨自叹，日夜悲号惭愧。

一日，正当嗟悼之际，俄见一僧一道远远而来，生得骨格不凡，丰神迥异，说说笑笑来至峰下，坐于石边高谈快论。先是说些云山雾海神仙玄幻之词，后便说到红尘中荣华富贵。此石听了，不觉打动凡心，也想要到人间去享一享这荣华富贵，但自恨粗蠢，不得已，便口吐人言，向那僧道说道："大师，弟子蠢物，不能见礼了。适闻二位谈那人世间荣耀繁华，心切慕之。弟子质虽粗蠢，性却稍通，况见二仙形仙迹，高人恳请携带弟子得入红尘，在那富贵场中，温柔乡里受享几年，自当永佩洪恩，万劫不忘也。"

二仙师听毕，齐憨笑道："善哉，善哉！那道人道："果是罕闻。只是到红尘中去，有一段昌明隆盛之邦，诗礼簪缨之族，花柳繁华地，温柔富贵乡去安身立命。倒只怕善始者多，善终者少。"

那僧便说："你且同我到警幻仙子宫中，将这蠢物交割清楚，待这一干风流孽鬼下世已完，你我再去。如今虽已有一半落尘，然犹未全集。"

第二回 贾夫人仙逝扬州城 冷子兴演说荣国府

话说封肃因听见公差传唤，忙出来陪笑启问。那些人只嚷："快请出甄爷来！"封肃忙陪笑道："小人姓封，并不姓甄。只有当日小婿姓甄，今已出家一二年了，不知可是问他？"那些公人道："我们也不知什么'真''假'，因奉太爷之命来问，他既是你女婿，便带了你去亲见太爷面禀，省得乱跑。"说着，不容封肃多言，大家推拥他去了。封家人个个都惊慌，不知何兆。

那天约二更时，只见封肃方回来，欢天喜地。众人忙问端的。他乃说道："原来本府新升的太爷姓贾名化，本胡州人氏，曾与女婿旧日相交。方才在咱们门前过去，因看见娇杏那丫头买线，所以他只当女婿移住于此。我一一将原故回明，那太爷倒伤感叹息了一回，又问外孙女儿，我说看灯丢了。太爷说：'不妨，我自使番役务必探访回来。'临走倒送了我二两银子。"

甄家娘子听了，不免心中伤感。一宿无话。次日，早有雨村遣人送了两封银子、四匹锦缎，答谢甄家娘子，又寄一封密书与封肃，转托他向甄家娘子要那娇杏作二房。封肃喜的屁滚尿流，巴不得去奉承，便在女儿前一力撺掇成了，乘夜只用一乘小轿，便把娇杏送进去了。雨村欢喜，自不必说，乃封百金赠封肃，外谢甄家娘子许多物事，令其好生养赡，以待寻访女儿下落。封肃回家无话。

却说娇杏这丫鬟，乃是甄家之婢，那日回顾雨村，本出于无意，因想这官人面熟，故看了两次。谁知雨村错认他是个知己，时刻放在心头。即受了甄家娘子之聘，今见娇杏那丫头，虽无十分姿色，却亦有动人之处，遂趁此机会，正遂了平生之愿。只一年，便生了一子；又半载，雨村嫡妻忽染疾下世，雨村便将他扶侧作正室夫人了。正是：

偶因一回顾，便为人上人。

第三回 贾雨村夤缘复旧职 林黛玉抛父进京都

却说雨村忙回头看时，不是别人，乃是当日同僚一案参革的号张如圭者。他本系此地人，革后家居，今打听得都中奏准起复旧员之信，他便四下里寻情找门路，忽遇见雨村，故忙道喜。二人见了礼，张如圭便将此信告诉雨村，雨村自是欢喜，忙忙的叙了两句，遂作别各自回家。冷子兴听得此言，因忙画计，令雨村央烦林如海，转向都中去央烦贾政。雨村依计而行，作书与林如海。

那女学生黛玉，身体方愈，原不忍弃父而往；无奈他外祖母致意务去，且兼如海说："汝父年将半百，再无续室之意；且汝多病，年又极小，上无亲母教养，下无姊妹兄弟扶持，今依傍外祖母及舅氏姊妹去，正好减我顾盼之忧，何反云不往？"黛玉听了，方洒泪拜别，随了奶娘及荣府几个老妇人登舟而去。雨村另有一只船，带两个小童，依附黛玉而行。

有日到了都中，进入神京，雨村先整了衣冠，带了小童，拿着宗侄的名帖，至荣府的门前投了。彼时贾政已看了妹丈之书，即忙请入相会。见雨村相貌魁伟，言语不俗，且这贾政最喜读书人，礼贤下士，济弱扶危，大有祖风；况又系妹丈致意，因此优待雨村，更又不同。便竭力内中协助，题奏之日，轻轻谋了一个复职候缺。不上两个月，金陵应天府缺出，便谋补了此缺，拜辞了贾政，择日上任去了，不在话下。
""" * 20  # 重复20次增加数据量

    output_path = os.path.join(output_dir, "hongloumeng_large.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    print(f"创建中文样本数据: {output_path}")
    print(f"大小: {len(sample_text)} 字符")

    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("GPT 训练数据下载工具")
    print("=" * 60)

    output_dir = "/Users/even/Desktop/workspace/GPT-mmhd/data"

    print("\n选项:")
    print("1. 下载 wikitext (英文，推荐用于学习)")
    print("2. 创建中文样本数据")
    print("3. 两者都下载")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        download_wikitext(output_dir)
    elif choice == "2":
        create_chinese_sample(output_dir)
    elif choice == "3":
        download_wikitext(output_dir)
        create_chinese_sample(output_dir)
    else:
        print("无效选择，下载 wikitext...")
        download_wikitext(output_dir)

    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)
    print("""
下一步：
1. 修改 06_pretrain/config.py 中的 data_path
2. 运行训练: python 06_pretrain/train.py --config tiny
    """)
