
import json
import pandas as pd
import os
from lm_modeling import load_model, load_text2embedding
import torch
def get_node2id(data_path,cskg_df):
    node1_series = cskg_df['node1'].astype(str).str.strip()
    node2_series = cskg_df['node2'].astype(str).str.strip()
    all_entities = pd.Series(pd.concat([node1_series, node2_series]).unique())
    all_entities = all_entities[all_entities != 'nan']  # 去除可能的 nan 字符串

    # 按字典序稳定排序（也可以按出现顺序）
    entities_sorted = sorted(all_entities.tolist())
    entity2id = {ent: idx for idx, ent in enumerate(entities_sorted)}
    # 将结果保存到 args.data_path 目录下，兼容后续代码读取 entities.dict / relations.dict
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, 'entities.dict'), 'w') as fout:
        for ent, eid in entity2id.items():
            fout.write(f"{eid}\t{ent}\n")
    print(f"Total entities: {len(entity2id)}")
    print(f"Saved entity2id to {os.path.join(data_path, 'entities.dict')}")
    return entity2id

def get_relation2id(data_path,cskg_df):
    # 关系
    rel_series = cskg_df['relation'].astype(str).str.strip()
    rels_unique = sorted([r for r in rel_series.unique() if r != 'nan'])
    relation2id = {rel: idx for idx, rel in enumerate(rels_unique)}
    relations = rels_unique  # 按 id 顺序的关系列表
    with open(os.path.join(data_path, 'relations.dict'), 'w') as fout:
        for rel, rid in relation2id.items():
            fout.write(f"{rid}\t{rel}\n")
    print(f"Total relations: {len(relation2id)}")
    print(f"Saved relation2id to {os.path.join(data_path, 'relations.dict')}")
    return relation2id

def get_embeddings(data_path, entity2id, relation2id):
    # 得到节点和边描述
    with open("./data/wikidata_descriptions.json", 'r', encoding='utf-8') as f:
        node_descriptions = json.load(f)
    #构建 label_map（先从 node1/node2 列收集标签，取第一个非空）
    df1 = cskg_df[["node1","node1;label"]].rename(columns={"node1":"node","node1;label":"label"})
    df2 = cskg_df[["node2","node2;label"]].rename(columns={"node2":"node","node2;label":"label"})
    labels_df = pd.concat([df1, df2], ignore_index=True)
    labels_df = labels_df[labels_df["label"].notna()].drop_duplicates(subset="node", keep="first")
    label_map = dict(zip(labels_df["node"], labels_df["label"]))
   
    # 4. 一次性生成 node_desc 列表（长度和 nodes 对应）
    node_desc = [None] * len(entity2id)
    for node,nid in entity2id.items():
        label = label_map.get(node, None)
        if label is None:
            label = node
        node_desc[nid] = get_raw_text(node, label, node_descriptions)

    # Build mapping from relation -> relation;label (if available)
    relation_label_map = {}
    if 'relation;label' in cskg_df.columns:
        df_rel = cskg_df[['relation', 'relation;label']].rename(columns={'relation;label': 'label'})
        df_rel = df_rel[df_rel['label'].notna()].drop_duplicates(subset='relation', keep='first')
        relation_label_map = dict(zip(df_rel['relation'], df_rel['label']))

    assert len(relation_label_map)==len(relation2id), "Some relations are missing labels!"
    # Create relation description list ordered by relation id (0..N-1)
    relation_desc = [''] * len(relation2id)
    for rel, rid in relation2id.items():
        label = relation_label_map.get(rel)
        relation_desc[rid] = label
    
    model_name = "sbert"
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]
  
    
    # cache embeddings to files so next run can load directly
    node_attr_path = os.path.join(data_path, 'node_attr.pt')
    edge_attr_path = os.path.join(data_path, 'edge_attr.pt')
    # test = text2embedding(model, tokenizer, device, ["test","train"])
    
    node_attr = text2embedding(model, tokenizer, device, node_desc) 
    torch.save(node_attr, node_attr_path)
    print("Saved node embeddings to %s" % node_attr_path)
   
    edge_attr = text2embedding(model, tokenizer, device, relation_desc)
    torch.save(edge_attr, edge_attr_path)
    print("Saved edge embeddings to %s" % (edge_attr_path))
        
def get_train_valid_test(data_path,cskg_df):
    # split data into train/valid/test with ratio 0.98/0.01/0.01 per source
    train_parts = []
    valid_parts = []
    test_parts = []

    for src, grp in cskg_df.groupby('source'):
        grp_shuffled = grp.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(grp_shuffled)
        n_train = int(n * 0.98)
        n_valid = int(n * 0.01)
        n_test = n - n_train - n_valid

        train_parts.append(grp_shuffled.iloc[:n_train])
        valid_parts.append(grp_shuffled.iloc[n_train:n_train + n_valid])
        test_parts.append(grp_shuffled.iloc[n_train + n_valid:])

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f"Total triples: {len(cskg_df)}")
    print(f"Train: {len(train_df)} ({len(train_df)/len(cskg_df):.4f})")
    print(f"Valid: {len(valid_df)} ({len(valid_df)/len(cskg_df):.4f})")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(cskg_df):.4f})")

    # base_name = os.path.splitext(os.path.basename(cskg_path))[0]
    train_path = os.path.join(data_path, f"train.tsv")
    valid_path = os.path.join(data_path, f"valid.tsv")
    test_path = os.path.join(data_path, f"test.tsv")

    train_df.to_csv(train_path, sep="\t", index=False)
    valid_df.to_csv(valid_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    print(f"Saved train to {train_path}")
    print(f"Saved valid to {valid_path}")
    print(f"Saved test  to {test_path}")

    return train_df, valid_df, test_df

def get_raw_text(node_name,node_label,node_descriptions):
    raw_text =''
    
    if node_name in node_descriptions:
        if isinstance(node_label,str):
            if node_descriptions[node_name]!="No description available":
                raw_text = node_label +":"+ node_descriptions[node_name]
            else:
                raw_text = node_label
        else:
            if node_descriptions[node_name]!="No description available":   
                raw_text = node_descriptions[node_name]
            else:
                raw_text = "Unknown"
    else:
        if isinstance(node_label,str):
            raw_text = node_label
        else:
            raw_text = "Unknown"
    return raw_text
if __name__ == "__main__":
    data_path = "./data/cskg-wd" 
    cskg_df = pd.read_csv("./data/cskg.tsv", sep="\t", on_bad_lines="skip")
    # 只保留来源为 WN、CN 和 WD 的行（大小写不敏感）
    cskg_df['source'] = cskg_df['source'].astype(str).str.strip().str.upper()
    cskg_df = cskg_df[cskg_df['source'].isin({
        # 'WN',  # WordNet
        # 'CN',  # ConceptNet
        'WD'   # Wikidata
    })]
    # cskg_df['source'] = cskg_df['source'].astype(str).str.strip()
    # cskg_df = cskg_df[cskg_df['source'].notna() & (cskg_df['source'] != '')]
    # cskg_df = cskg_df[~cskg_df['source'].str.lower().eq('nan')]
    # cskg_df = cskg_df[~cskg_df['source'].str.lower().eq('vg')]
    print(set(cskg_df["source"]))
    print(len(cskg_df))
    save_path = os.path.join(data_path, f"triplets.tsv")
    print(f"Saving filtered triples to {save_path}")

    cskg_df.to_csv(save_path, sep="\t", index=False)

    #0.构建 node2id / relation2id 字典
    # entity2id = get_node2id(data_path,cskg_df)
    # relation2id = get_relation2id(data_path,cskg_df)
    # #1.处理数据集划分
    # # train_df, valid_df, test_df = get_train_valid_test(data_path,cskg_df)
    # get_embeddings(data_path, entity2id, relation2id)