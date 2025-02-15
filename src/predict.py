from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from collections import Counter
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

#載入模型
def load_models(Models_dir):
    
    model_dir=Models_dir  #修改成Moldes的所在地

    # 確保路徑存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f" 模型目錄不存在: {model_dir}")

    # 載入 Vectorizer 和 LabelEncoder
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")

    if not os.path.exists(vectorizer_path) or not os.path.exists(label_encoder_path):
        raise FileNotFoundError(" 缺少 vectorizer 或 label_encoder，請檢查模型目錄！")

    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)

    # 建立模型字典
    model_files = {
        "rdf_smote": "random_forest.joblib",
        "nb_smote": "naive_bayes.joblib",
        "svm_smote": "svm.joblib",
        "lgbm_smote": "lightgbm.joblib",
    }

    models = {}
    for model_name, file_name in model_files.items():
        model_path = os.path.join(model_dir, file_name)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"缺少模型: {file_name}")

    print("所有模型載入完成！")
    return vectorizer, label_encoder, models



# 預測與評估（加入預測機率）
def predict_and_evaluate(new_data, model, vectorizer, label_encoder):
    # 確保輸入數據包含必要的欄位
    required_columns = {"n_lastname"}  #不需要有main_id
    missing_columns = required_columns - set(new_data.columns)
    if missing_columns:
        raise KeyError(f"缺少必要欄位：{', '.join(missing_columns)}")

    # 將文本數據轉換為數值特徵
    X_new = vectorizer.transform(new_data['n_lastname'])

    # 取得預測結果
    predicted_ids = model.predict(X_new)
    predicted_labels = label_encoder.inverse_transform(predicted_ids)

    # 嘗試獲取預測機率（部分模型可能不支援）
    try:
        probs = model.predict_proba(X_new)  # 取得所有類別的機率分布
        max_probs = probs.max(axis=1)  # 取得最高的機率值
    except AttributeError:
        max_probs = [None] * len(predicted_ids)  # 若模型不支援，則填 None

    # 建立新 DataFrame，避免修改原始數據
    new_data = new_data.copy()
    new_data['predicted_label'] = predicted_labels
    new_data['prediction_confidence'] = max_probs  # 新增預測機率欄位


    return new_data

#預測所有模型，並且進行merge
def merge_model_predictions(data2, models, predict_and_evaluate, vectorizer, label_encoder):

    # 預測所有模型
    results_df = {}
    for model_name, model in models.items():
        results_df[model_name] = predict_and_evaluate(data2.copy(), model, vectorizer, label_encoder)

    # 定義欄位後綴，避免欄位名稱衝突
    suffixes = {
        "rdf_smote": "_rdf",
        "nb_smote": "_nb",
        "svm_smote": "_svm",
        "lgbm_smote": "_lgbm"
    }

    # 重新命名各模型的預測欄位
    for model_name, df in results_df.items():
        df.rename(columns={
            "predicted_label": f"predicted_label{suffixes[model_name]}",
            "code": f"code{suffixes[model_name]}",
            "prediction_confidence": f"prediction_confidence{suffixes[model_name]}"
        }, inplace=True)

    # 4️⃣ 以 rdf_smote 作為基準，合併所有模型預測結果
    merged_smote_df = results_df["rdf_smote"]
    for model_name in ["nb_smote", "svm_smote", "lgbm_smote"]:
        merged_smote_df = merged_smote_df.merge(
            results_df[model_name],
            on=["lastname", "result", "n_lastname", "jeiba_lastname"],
            how="outer"
        )
    print("所有模型預測結果已經merge")
    return merged_smote_df, results_df , suffixes

#投票表決
def weighted_voting(row):
    """當 SVM 信心機率 < 90% 時，使用加權投票"""
    model_weights = {
        "nb_smote_pred": 3,   # NB 權重最高
        "lgbm_smote_pred": 1, # LGBM 權重最低
        "rdf_smote_pred": 2    # 隨機森林權重 中間
    }

    threshold = 0.90  # SVM 信心機率閾值
    svm_pred = row["svm_smote_pred"]
    svm_confidence = row["prediction_confidence_svm_smote"]  # SVM 預測的信心機率

    # 若 SVM 信心高於 90%，則直接使用 SVM 預測結果
    if pd.notna(svm_confidence) and svm_confidence >= threshold:
        return svm_pred

    # 使用加權投票
    vote_scores = Counter()
    for model, weight in model_weights.items():
        if pd.notna(row[model]):  # 確保不處理 NaN
            vote_scores[row[model]] += weight

    # 取得權重最高的類別
    if vote_scores:
        return max(vote_scores, key=vote_scores.get)
    else:
        return svm_pred  # 若其他模型無法提供結果，回傳 SVM 預測值

#集成
def ensemble_voting(data2, models, results_df, suffixes, weighted_voting, output_file="final_predictions.csv"):

    # 建立 DataFrame 存放所有模型的預測結果
    ensemble_results = pd.DataFrame()

    # 加入各模型的預測標籤與機率
    for model_name in models.keys():
        pred_col = f"predicted_label{suffixes.get(model_name, '')}"
        conf_col = f"prediction_confidence{suffixes.get(model_name, '')}"
        
        # 檢查欄位是否存在，避免 KeyError
        if pred_col in results_df[model_name].columns:
            ensemble_results[f"{model_name}_pred"] = results_df[model_name][pred_col]
        else:
            print(f"{model_name} 缺少 {pred_col}，跳過！")

        if conf_col in results_df[model_name].columns:
            ensemble_results[f"prediction_confidence_{model_name}"] = results_df[model_name][conf_col]
        else:
            print(f"{model_name} 缺少 {conf_col}，跳過！")

    # 套用加權投票
    ensemble_results["final_pred"] = ensemble_results.apply(weighted_voting, axis=1)

    # 儲存結果
    ensemble_results.to_csv(output_file, index=False)
    print(f"✅  已儲存至 {output_file}")

    return ensemble_results  



if __name__ == "__main__":

    # 載入Models
    MODEL_DIR = "E:/PSFD_Department-Classification/Models"
    vectorizer, label_encoder, models = load_models(MODEL_DIR)

    # 上載想要預測的數據
    data2 = pd.read_csv("C:/Users/user/Downloads/data.csv", encoding="utf-8")

    # 執行預測並合併結果
    merged_smote_df, results_df , suffixes = merge_model_predictions(data2, models, predict_and_evaluate, vectorizer, label_encoder)
    
    #進行結果輸出
    ensemble_results = ensemble_voting(data2, models, results_df, suffixes, weighted_voting)

