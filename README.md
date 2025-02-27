# -PSFD_學系分類使用
## 使用PSFD 對於台灣各大學的學系分類，共18個分類
| mainid| mainname|
|-------|-----------|
| 2	| 教育學類| 
| 3	| 藝術學類| 
| 4	| 人文學類| 
| 5	| 經濟、社會及心理學類| 
| 6	| 商業及管理學類| 
| 7	| 法律學類| 
| 8	| 自然科學類| 
| 9	| 數學及電算機科學類| 
| 10| 醫藥衛生學類| 
| 11| 工業技藝學類| 
| 12| 工程學類| 
| 13| 建築及都市規劃學類| 
| 14| 農林漁牧學類| 
| 15| 家政學類| 
| 16| 運輸通信學類| 
| 17| 觀光服務類| 
| 18| 大眾傳播學類| 
| 97| 其他學類| 

## 使用ckip_nlp 以及 jeiba 進行分詞
ckip 參照https://github.com/ckiplab/ckiptagger/wiki/Chinese-README

## 使用random forest、naive_bayes、SVM、LightGBM 進行訓練
### 各模型準確率
| Model| Accu.|
|-------|-----|
|RdF | 0.869|
|NB  | 0.890|
|SVM | 0.897|
|LightGBM|0.872|
|集成分類|0.899|


## 資料來源：華人家庭動態調查(https://psfd.sinica.edu.tw/V2/?page_id=368&lang=zh)



