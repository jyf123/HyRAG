import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.args import Arguments
from data.load import load_data
from tqdm import tqdm
import argparse
import json
# 1. 模型路径
model_id = "/data/jyf/code/graphtta/llama3.1-8b-Instruct"

# 2. 加载分词器和模型
# 注意：Llama 3 推荐使用 bfloat16 精度加载以节省显存并保持精度
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="left"
    )
    
    # LLaMA 系列必须显式设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    # device_map="auto"  # 自动分配到 GPU/CPU
).to(device)

def abs_prompt(text,dataset):
    if dataset in ['cora','citeseer']:
        return f"""You are an expert in machine learning and artificial intelligence.  Given the title and abstract of a research paper, generate a single-sentence **abstract description** that captures its high-level research paradigm or conceptual foundation. 
                    Output ONLY the sentence, with no prefix, explanation, or formatting.
                    Focus on:
                    - The general problem class
                    - The underlying theoretical framework
                    - Avoid mentioning specific algorithms, datasets, or implementation details.
                    The description should be general enough to group papers from the same broad research area, but still meaningful.
                    Text:
                    <<<
                    {text}
                    >>>
                    Abstract description:
                    """
    elif dataset=='history':
        return f"""You are a literary and historical analysis expert. Given the title and description of a history book, generate a single-sentence **abstract description** that captures its overarching historical theme, interpretive lens, or conceptual scope.
                    Focus on:
                    - The broad historical domain
                    - The temporal or geographical framework
                    - The narrative perspective
                    Output ONLY the sentence, with no prefix, explanation, or formatting.
                    Avoid mentioning specific names, battles, dates, or titles.
                    Text:
                    <<<
                    {text}
                    >>>
                    Abstract description:
                    """
    elif dataset=='children':
        return f"""You are a children's literature analysis expert. Given the title and description of a children's book, generate a single-sentence **abstract description** that captures its overarching theme, developmental focus, or narrative scope.
                    Focus on:
                    - The broad subject or theme of the book
                    - The general setting or everyday/literary context
                    - The emotional, educational, or narrative perspective
                    Output ONLY the sentence, with no prefix, explanation, or formatting.
                    Avoid mentioning specific character names, exact events, or book titles.
                    Text:
                    <<<
                    {text}
                    >>>
                    Abstract description:
                    """
    elif dataset in ['computer','photo']:
        return f"""You are an expert in product ontology and e-commerce semantics. Given a user review of an electronic product, generate a single-sentence abstract description that captures its core functional category, usage context, or technological purpose—without mentioning specific brands, model numbers, personal opinions, or named entities.
                    Rules:
                    - Focus on what the product *does* or *is used for*, not how the user feels.
                    - Generalize to a conceptual class (e.g., "wireless audio playback device", "smart home environmental sensor").
                    - Do NOT include: brand names, personal pronouns ("I", "my"), ratings ("great", "terrible"), or exact specifications unless essential to function.
                    - Output ONLY the sentence. No prefix, no explanation.
                    Text:
                    <<<
                    {text}
                    >>>
                    Abstract description:"""
    elif dataset=='wikics':
        return f"""You are an expert in technical semantics. Given a Wikipedia-style description of a computing-related entity, generate a single-sentence abstract description that captures **what it fundamentally enables or achieves**.
                    Rules:
                    - Describe the core capability or purpose using only functional verbs and general nouns.
                    - Never mention: brand names, people, dates, companies, locations, or high-level categories (e.g., "AI", "networking", "security").
                    - Do not use phrases like "a type of X" or "an example of Y".
                    - Start with a capital letter and form a complete sentence.
                    - Output ONLY the sentence.
                    Text:
                    <<<
                    {text}
                    >>>
                    Abstract description:"""
    elif dataset=='instagram':
        return f"""You are an expert in social media semantics. Given a user's Instagram bio, generate a single-sentence abstract description that captures the **primary communicative intent explicitly stated or strongly implied by the text**.
                    Rules:
                    - ONLY use information directly present in the bio. Do NOT infer unstated purposes, audiences, or business models.
                    - NEVER add details not mentioned (e.g., do not assume "photographer" = "offers paid services" unless stated).
                    - Remove all emojis, @mentions, email addresses, phone numbers, hashtags, and specific locations (replace with generic terms like "a city" if needed for fluency).
                    - Avoid words: "commercial", "normal", "user", "account", "influencer", "brand", "profile".
                    - If the bio only expresses personal identity or mood (e.g., "mom+wife life", "Summer forever"), describe it as self-documentation.
                    - Output ONLY one grammatically complete sentence. No prefix.
                    Text:
                    <<<
                    {text}
                    >>>
                    Abstract description:"""
    
def con_prompt(text,dataset):
    if dataset in ['cora','citeseer']:
        return f"""You are an expert in machine learning and artificial intelligence.  Given the title and abstract of a research paper, generate a single-sentence **concrete description** that summarizes its specific technical contributions or key components.
                    Focus on:
                    - Specific methods, models, or algorithms used 
                    - Tasks, benchmarks, or applications mentioned 
                    - Novel techniques or architectural choices
                    Do NOT generalize beyond what is stated in the title or abstract. Output ONLY the sentence, with no prefix, explanation, or formatting.
                    Text:
                    <<<
                    {text}
                    >>>
                    Concrete description:
                    """
    elif dataset=='history':
        return f"""You are a literary and historical analysis expert.  
                    Given the title and description of a history book, generate a single-sentence **concrete description** that summarizes its specific subject matter, key figures, events, or settings.
                    Focus on:
                    - Named individuals, empires, wars, dynasties, or locations (e.g., "Julius Caesar", "the Ming Dynasty", "the American Civil War")
                    - Specific time periods or events (e.g., "the fall of Constantinople in 1453")
                    - Book format or focus (e.g., "a biography of Eleanor Roosevelt", "a photographic history of World War I")
                    Do NOT generalize beyond what is stated in the title or description.
                    Output ONLY the sentence, with no prefix, explanation, or formatting.
                    Text:
                    <<<
                    {text}
                    >>>
                    Concrete description:
                    """
    elif dataset=='children':
        return f"""You are a children's literature analysis expert.   
                    Given the title and description of a children's book, generate a single-sentence **concrete description** that summarizes its specific subject matter, characters, settings, or story focus.
                    Focus on:
                    - Named characters, animals, creatures, families, or locations explicitly mentioned in the text
                    - Specific activities, topics, or scenarios (e.g., bedtime, school, holidays, friendship, counting, dinosaurs, fairy tales)
                    - Book format or focus (e.g., "a picture book about a bear learning to share", "an early reader set in a magical school", "a bedtime story featuring farm animals")
                    Do NOT generalize beyond what is stated in the title or description.
                    Output ONLY the sentence, with no prefix, explanation, or formatting.
                    Text:
                    <<<
                    {text}
                    >>>
                    Concrete description:
                    """
    elif dataset in ['computer','photo']:
        return f"""You are an expert in extracting factual details from user-generated content. Given a user review of an electronic product, generate a single-sentence concrete description that summarizes the key tangible attributes, features, use cases, or performance characteristics explicitly mentioned in the review.
                    Rules:
                    - Preserve specific details: connectivity (Bluetooth, Wi-Fi), form factor (compact, handheld), functionality (noise cancellation, voice assistant support), compatibility, etc.
                    - Do NOT invent information not present in the review.
                    - Do NOT generalize beyond what is stated (e.g., if it says "works with Alexa", do not say "smart speaker").
                    - Avoid subjective language; rephrase opinions as observable claims if possible (e.g., "user reports long battery life" → "advertised for extended battery operation").
                    - Output ONLY the sentence. No prefix, no explanation.
                    Text:
                    <<<
                    {text}
                    >>>
                    Concrete description:"""
    elif dataset=='wikics':
        return f"""You are an expert in computer science ontology. Given a Wikipedia-style description of a technical entity, generate a concise concrete description that summarizes its **key technical functionality or purpose**, using only factual and generalizable information.
                    Rules:
                    - Focus on WHAT it does or enables, not WHO made it or WHEN.
                    - Remove brand names, person names, and specific version numbers unless essential to function.
                    - For lists (e.g., "List of column-oriented DBMSes"), state: "A comprehensive list of [category] systems."
                    - Keep to one sentence if possible; at most two short sentences.
                    - Output ONLY the description. No prefix, no explanation.
                    Text:
                    <<<
                    {text}
                    >>>
                    Concrete description:"""
    elif dataset=='instagram':
        return f"""You are an expert in social media semantics. Generate a concise factual summary of the user's bio using ONLY explicit claims made in the text.
                    Rules:
                    - Include ONLY roles, activities, or topics the user directly states (e.g., "shares news", "is a photographer", "discusses skincare").
                    - NEVER assume monetization, audience size, or professional status unless explicitly declared (e.g., "offers SEO services" is OK; "makes money from photography" is NOT).
                    - Replace specific place names with generic descriptors (e.g., "Pamplona" → "a city in Spain"; "Los Angeles" → "a major U.S. city").
                    - Strip all emojis, @usernames, email addresses, promotional hashtags (#SEO), and subjective slogans ("welcome to my visual diary").
                    - If the bio contains no concrete claims (e.g., only emojis or moods), output: "Shares personal reflections and daily life moments."
                    - Output ONLY one sentence. No explanation.

                    Text:
                    <<<
                    {text}
                    >>>
                    Concrete description:"""

def generate_answer_only(prompt_text):
    # 3. 构建对话格式 (Chat Template)
    # Llama 3 Instruct 需要特定的 system/user 格式
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    # 4. 应用模板并将文本转为 ID
    # add_generation_prompt=True 会自动添加引导模型输出的特殊 token
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 5. 定义停止符 (Terminators)
    # Llama 3 有两个结束符：eos_token 和 <|eot_id|>
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 6. 生成 (Generate)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    # ================= 关键步骤 =================
    # 7. 只提取生成的回答部分
    # outputs[0] 是包含了 [Input IDs + Generated IDs] 的完整序列
    # 我们只取 input_ids 长度之后的部分
    l = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(l)
    response = outputs[0][input_ids.shape[-1]:]
    
    # 8. 解码
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_batch_answers(batch_prompts):
    """
    处理批量输入，返回纯回答列表
    """
    # all_responses = []
    
    # 2. 停止符设置 (Llama 3 特有)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # # 按 batch_size 分块处理
    # for i in range(0, len(prompts), batch_size):
    #     batch_prompts = prompts[i : i + batch_size]
        
        # === 关键修改点 2: 批量应用 Chat 模板 ===
        # 我们手动把每个 prompt 包装成 message 格式
    batch_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}
        ] for p in batch_prompts
    ]
    
    # apply_chat_template 如果传入 list of list，会自动处理
    # 但为了保险起见，我们先转成 text list 再 tokenize
    batch_texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]

    # === 关键修改点 3: Tokenize 并 Padding ===
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,       # 自动填充到当前 batch 中最长的长度
        truncation=True,    # 防止超长
        max_length=2048     # 根据显存情况调整
    ).to(model.device)

    # 获取输入的长度 (Padding 后的统一长度)
    input_len = inputs.input_ids.shape[1]

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id # 显式指定 pad_token_id
        )

    # === 关键修改点 4: 批量切片 ===
    # outputs 的形状是 [batch_size, seq_len]
    # 因为做了 Left Padding，所有生成的 token 都整齐地接在 input_len 之后
    generated_tokens = outputs[:, input_len:]

    # === 关键修改点 5: 批量解码 ===
    decoded_responses = tokenizer.batch_decode(
        generated_tokens, 
        skip_special_tokens=True
    )
        
        # all_responses.extend(decoded_responses)

    return decoded_responses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split graph data into train/val/test node sets")
    parser.add_argument("--dataset", help="path to the saved torch data file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for generation")
    parser.add_argument("--mode", help="random seed")
    parser.add_argument("--part", type=int, default=1, help="part id for generation, default 1")

    args = parser.parse_args()
    data, text, _, _ = load_data(args.dataset, seed=1)
    abs_texts = []
    con_texts = []
    raw_texts = data.raw_texts
    batch_size = args.batch_size
    num_node = len(raw_texts)
    print(f"Total nodes: {num_node}, Processing part {args.part} with batch size {batch_size}...")
    part_size = num_node // 3
    start_idx = (args.part - 1) * part_size
    if args.part == 3:
        end_idx = num_node
    else:
        end_idx = start_idx + part_size
    # end_idx = min(start_idx + part_size, len(raw_texts))
    # raw_texts = raw_texts[start_idx:end_idx]
    print(f"Processing nodes from index {start_idx} to {end_idx}...")
    for i in tqdm(range(start_idx,end_idx,batch_size)):
        j = i+batch_size if i+batch_size<len(raw_texts) else len(raw_texts)
        text = raw_texts[i:i+batch_size]
        if args.mode=='abs':
            abstract_prompts = [abs_prompt(t,args.dataset) for t in text]
            abstract_outputs = generate_batch_answers(abstract_prompts)
            abs_texts.extend(abstract_outputs)
        elif args.mode=='con':
            concrete_prompts = [con_prompt(t,args.dataset) for t in text]
            concrete_outputs = generate_batch_answers(concrete_prompts)
            con_texts.extend(concrete_outputs)
    if args.mode=='abs':
        results = {"abs_texts": abs_texts}
    elif args.mode=='con':
        results = {"con_texts": con_texts}
    with open(f"./processed_data/{args.dataset}/{args.mode}_{args.part}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        # print(len(abs_texts))
    # print(data.x.size(0),len(abs_texts),len(con_texts))
    # data.abs_texts = abs_texts
    # data.con_texts = con_texts
    # torch.save(data, f"./processed_data/{args.dataset}-ac.pt")


# --- 测试 ---
# test_inputs = [
#    """You are an expert in product ontology and e-commerce semantics. Given a user review of an electronic product, generate a single-sentence abstract description that captures its core functional category, usage context, or technological purpose—without mentioning specific brands, model numbers, personal opinions, or named entities.
#                      Rules:
#                      - Focus on what the product *does* or *is used for*, not how the user feels.
#                      - Generalize to a conceptual class (e.g., "wireless audio playback device", "smart home environmental sensor").
#                      - Do NOT include: brand names, personal pronouns ("I", "my"), ratings ("great", "terrible"), or exact specifications unless essential to function.
#                      - Output ONLY the sentence. No prefix, no explanation.
#                      Text:
#                     <<<
#                      All-in-wonder OK, but could be better ! Hi all, Well, I was the first to buy one of these cards inSouth Africa. And yes there were some pitfalls, but I'll get intothose later ... At the price it is a great card. DVD facilities, digital VCR, graphics accelerator and a whole host of usefull software included in the bundle. The graphics of the card are great, I was originally going for the 32M Riva TNT but when I saw the features of this card I was sold. Playing Half-life, Unreal, Speed 3, Star-Wars pod racer I have NO complaints - this card does what it's supposed to ... As far as the pitfalls go, Hmmmmmm , documentation inside the box , well, it is in all languages possible but doesn't really tell you much ! I was trying to read German to see if there was anything that was skipped out in the English version ! There is an on-line fault reporting service but they do take their time replying but they did manage to clear up one of my problems - I am still waiting for a reply for the other. I would really reccomend a large hard drive for the video capture feature, the files are BIG when saved in editable formats - but the software bundle does include a utility to compress the video into MPEG2 format which is a lot smaller but is still big. All in all, yes I would reccomend this card, I mean look at all of the features
#                      >>>
#                      Abstract description:""",
#     """You are an expert in product ontology and e-commerce semantics. Given a user review of an electronic product, generate a single-sentence abstract description that captures its core functional category, usage context, or technological purpose—without mentioning specific brands, model numbers, personal opinions, or named entities.
#                      Rules:
#                      - Focus on what the product *does* or *is used for*, not how the user feels.
#                      - Generalize to a conceptual class (e.g., "wireless audio playback device", "smart home environmental sensor").
#                      - Do NOT include: brand names, personal pronouns ("I", "my"), ratings ("great", "terrible"), or exact specifications unless essential to function.
#                      - Output ONLY the sentence. No prefix, no explanation.
#                      Text:
#                     <<<
#                      PCTV almost ready for prime-time I've just bought the PCTV and have noticed some bugs in the software. You can watch TV just fine with the card, but the included Studio software will not run on my system without crashing. I haven't been able to capture a single clip since buying this card. I'm a systems tech., so I have some idea how to troubleshoot PC problems, but after checking out Pinnacle's msg. boards, I'm no closer to figuring out what's wrong. There may be a problem with DirectX 7.0. A co-worker who has almost an identical system (except he has DirectX 6.1) has no problem at all. This rings true for most of the people on the msg. boards, also. I bought this card mainly to watch TV on my PC, so the capture problems don't affect me too greatly, but if you're buying this card to primarily capture video, be aware you might run into some problems.
#                      >>>
#                      Abstract description:""",
#     """You are an expert in product ontology and e-commerce semantics. Given a user review of an electronic product, generate a single-sentence abstract description that captures its core functional category, usage context, or technological purpose—without mentioning specific brands, model numbers, personal opinions, or named entities.
#                      Rules:
#                      - Focus on what the product *does* or *is used for*, not how the user feels.
#                      - Generalize to a conceptual class (e.g., "wireless audio playback device", "smart home environmental sensor").
#                      - Do NOT include: brand names, personal pronouns ("I", "my"), ratings ("great", "terrible"), or exact specifications unless essential to function.
#                      - Output ONLY the sentence. No prefix, no explanation.
#                      Text:
#                     <<<
#                      An absurd extension of the expensive-stereo-cable phenomenon Monster Cable is best known for its high-end stereo and video cables, with equally high-end prices. There are those who say that these expensive cables can make all the difference to a high-end audio/video system; there are others who say that they make no difference whatsoever. But when it comes to telephone cable, there is simply no room for that sort of debate. Spending $20 for a ten-foot length of telephone cable is simply ludicrous, and Monster Cable loses a great deal of credibility in my eyes for even offering the product. The telephone signal going to your modem has in all likelihood travelled down thousands of feet of cables, and a maze of telephone company switching equipment, before it reaches that jack in your wall. That journey has caused signal loss and distortion. Unless your current modem cable is physically damaged in some way, there is simply no way that those ten feet of modem cable will make a significant difference. And if you're still inclined to disagree, consider this: What about the cable running through your house's walls, connecting the phone company's junction box to your wall jack? It's almost certainly not made out of any special high-tech materials; it's probably whatever brand the contractor could find on sale when he or she wired your house. And it's almost certainly much longer than your modem cable, meaning that its capability to cause distortion and signal loss is much greater. If you're not going to rip that cable
#                      >>>
#                      Abstract description:""",
#     """You are an expert in product ontology and e-commerce semantics. Given a user review of an electronic product, generate a single-sentence abstract description that captures its core functional category, usage context, or technological purpose—without mentioning specific brands, model numbers, personal opinions, or named entities.
#                      Rules:
#                      - Focus on what the product *does* or *is used for*, not how the user feels.
#                      - Generalize to a conceptual class (e.g., "wireless audio playback device", "smart home environmental sensor").
#                      - Do NOT include: brand names, personal pronouns ("I", "my"), ratings ("great", "terrible"), or exact specifications unless essential to function.
#                      - Output ONLY the sentence. No prefix, no explanation.
#                      Text:
#                     <<<
#                      Great performance, good variety of platforms supported This is Bay Networks (Netgear is their consumer line) implementation of the venerable DEC "Tulip" PCI ethernet chipset. There are many manufacturers of 10/100 cards that use Tulip, so you can go by price since they are all pretty much the same. That\'s what makes this card good -- it\'s one of the lower priced cards from a respected industry vendor. Tulip cards are known for their low CPU utilization, high throughput and their ability to run on a variety of platforms. I have personally used this card with great success on Linux, FreeBSD, Solaris x86, Windows 95, 98, and NT. I did dock one star however, and that\'s because their have been some minor problems with the drivers included on the driver diskettes. Those seem to have been fixed with subsequent releases, and the card itself wasn\'t to blame. If you do have problems, and you can use one of the other Tulip drivers included with your OS, such as the Digital DE500. In fact, Windows may autodetect it as an "Intel 21140 Fast Ethernet" or some such.
#                      >>>
#                      Abstract description:""",
# ]
# answers = generate_batch_answers(test_inputs, batch_size=2)

# for q, a in zip(test_inputs, answers):
#     # print(f"问: {q}")
#     print(f"{a}")
#     print("-" * 30)
# --- 测试 ---
# user_input = """You are an expert in product ontology and e-commerce semantics. Given a user review of an electronic product, generate a single-sentence abstract description that captures its core functional category, usage context, or technological purpose—without mentioning specific brands, model numbers, personal opinions, or named entities.
#                     Rules:
#                     - Focus on what the product *does* or *is used for*, not how the user feels.
#                     - Generalize to a conceptual class (e.g., "wireless audio playback device", "smart home environmental sensor").
#                     - Do NOT include: brand names, personal pronouns ("I", "my"), ratings ("great", "terrible"), or exact specifications unless essential to function.
#                     - Output ONLY the sentence. No prefix, no explanation.
#                     Text:
#                     <<<
#                     All-in-wonder OK, but could be better ! Hi all, Well, I was the first to buy one of these cards inSouth Africa. And yes there were some pitfalls, but I'll get intothose later ... At the price it is a great card. DVD facilities, digital VCR, graphics accelerator and a whole host of usefull software included in the bundle. The graphics of the card are great, I was originally going for the 32M Riva TNT but when I saw the features of this card I was sold. Playing Half-life, Unreal, Speed 3, Star-Wars pod racer I have NO complaints - this card does what it's supposed to ... As far as the pitfalls go, Hmmmmmm , documentation inside the box , well, it is in all languages possible but doesn't really tell you much ! I was trying to read German to see if there was anything that was skipped out in the English version ! There is an on-line fault reporting service but they do take their time replying but they did manage to clear up one of my problems - I am still waiting for a reply for the other. I would really reccomend a large hard drive for the video capture feature, the files are BIG when saved in editable formats - but the software bundle does include a utility to compress the video into MPEG2 format which is a lot smaller but is still big. All in all, yes I would reccomend this card, I mean look at all of the features
#                     >>>
#                     Abstract description:"""
# answer = generate_answer_only(user_input)

# print("-" * 20)
# print(answer)
# print("-" * 20)