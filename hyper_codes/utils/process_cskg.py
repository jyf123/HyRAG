import pandas as pd

def load_graph(cskg_path: str, embedding_path: str,source:str):
    # 1. 读取知识图谱三元组
    cskg_df = pd.read_csv(cskg_path, sep="\t", on_bad_lines="skip")
    print(set(cskg_df["source"]))
    print(len(cskg_df))
    # 只保留非 VG 的三元组（删除 source 包含 "VG" 的行）
    cskg_df = cskg_df[cskg_df["source"].str.contains(source, na=False)]