import os
import pandas as pd
import numpy as np
import jieba
#! pip install -U ckip-transformers jupyter notebook 寫法
from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger


# Initialize drivers
#model 有其它的可以選，如 "bert-base"
#device=0 是使用 GPU， device=-1 是使用 CPU，不指定也可以。
print("Initializing drivers ... WS")
ws_driver = CkipWordSegmenter(model="albert-base", device=-1)
print("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="albert-base", device=-1)
print("Initializing drivers ... all done")
print()

#讀取檔案預處理
def check_data(file_path):
  """根據檔案類型讀取 Excel 或 CSV，並對 'lastname' 欄位進行分詞"""
  # 檢查檔案是否存在
  if not os.path.exists(file_path):
      raise FileNotFoundError(f"❌ 找不到檔案: {file_path}")

  # 判斷檔案副檔名
  ext = os.path.splitext(file_path)[1].lower()
  # 根據副檔名讀取檔案
  if ext == ".xlsx" or ext == ".xls":
      df = pd.read_excel(file_path)
  elif ext == ".csv":
      df = pd.read_csv(file_path, encoding="utf-8")  # 避免亂碼，可根據需求修改編碼
  else:
      raise ValueError("不支援的檔案類型！請上傳 CSV 或 Excel 檔案")

  # 必須包含的欄位
  required_columns = {"lastname"}
  # 檢查缺少的欄位
  missing_columns = required_columns - set(df.columns)
  if missing_columns:
      raise KeyError(f"檔案缺少以下欄位：{', '.join(missing_columns)}，請檢查數據！")

  return df

# 文本預處理，這裡可以使用 jieba 進行中文分詞
def jieba_text(text):
    # 使用 jieba 分詞
    words = jieba.cut(text)
    return ' '.join(words)


# 保留斷詞
def clean(sentence_ws, sentence_pos):
  short_with_pos = []
  short_sentence = []
  stop_pos = set(['Nep', 'Nh', 'Nb',]) # 這 3 種詞性不保留
  for word_ws, word_pos in zip(sentence_ws, sentence_pos):
    # 只留名詞和動詞
    is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N") or word_pos.startswith("A")
    # 去掉名詞裡的某些詞性
    is_not_stop_pos = word_pos not in stop_pos
    # 只剩一個字的詞也不留
    is_not_one_charactor = not (len(word_ws) == 1)
    # 組成串列
    if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
      short_with_pos.append(f"{word_ws}({word_pos})")
      short_sentence.append(f"{word_ws}")
  return (" ".join(short_sentence), " ".join(short_with_pos))


#產生restult以及 n_lastname，並且將jeiba 補上
def n_lastname(df, ws_driver, pos_driver, clean, jieba_text):
  
  # 初始化新欄位
  df["result"] = ""
  df["n_lastname"] = ""

  # 進行每行的文本處理
  for index, row in df.iterrows():
      description = row["lastname"]
    
      # 分詞
      ws_result = ws_driver([description])
      # 詞性標註
      pos_result = pos_driver(ws_result)

      # 清洗數據
      cleaned_sentence, cleaned_with_pos = clean(ws_result[0], pos_result[0])

      # 保存結果
      df.at[index, "result"] = cleaned_with_pos
      df.at[index, "n_lastname"] = cleaned_sentence

  # 使用 jieba 進行分詞
  df["jeiba_lastname"] = df["lastname"].apply(jieba_text)

  # 填補 'n_lastname' 的空值，並去除前導空格
  df["n_lastname"] = df["n_lastname"].replace("", np.nan)
  df["n_lastname"] = df["n_lastname"].fillna(df["jeiba_lastname"]).str.lstrip()

  return df

if __name__ == "__main__":
  data =  check_data("C:/Users/user/Downloads/l_test.xlsx")
  data = n_lastname(data, ws_driver, pos_driver, clean, jieba_text)
  data.to_csv("C:/Users/user/Downloads/data.csv", index=False)
  print("數據處理完成！")

