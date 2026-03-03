import pandas as pd
from sklearn.cluster import KMeans

# 1. 加载你的标准化数据（请修改为你的实际路径）
X = pd.read_pickle("data/processed/questionnaire_standardized.pkl")

# 2. 简单运行一次K-means++，不做任何重命名
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    max_iter=100,
    tol=0.01,
    random_state=42  # 固定随机种子，保证可复现
)
labels = kmeans.fit_predict(X)
centers_raw = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns, index=[f"簇{0}", f"簇{1}", f"簇{2}"])

# 3. 打印6项核心指标的原始聚类中心
key_indicators = ["驾龄", "每周开车频率", "限速遵守度", "换道观察充分性", "情绪稳定性", "施工区安全感"]
print("="*80)
print("【原始聚类中心（未重命名）】")
print(centers_raw[key_indicators].round(2))
print("\n【各簇样本数量】")
print(pd.Series(labels).value_counts().sort_index())
print("="*80)