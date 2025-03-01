import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql.types import StructField, StructType, DoubleType
from pyspark.sql.functions import pandas_udf
import pickle

# 启用 Arrow 加速 Pandas 和 Spark 之间的数据传输
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# 将 Spark DataFrame 转换为 Pandas DataFrame
trainingDataPandas = trainingData.select("*").toPandas()

# 提取特征和标签
nfeatures = trainingDataPandas.shape[1] - 1
Xtrain = trainingDataPandas.iloc[:, 0:nfeatures]
ytrain = trainingDataPandas.iloc[:, -1]

# 训练 Scikit-learn 模型（随机森林分类器）
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(Xtrain, ytrain)

# 由于 Scikit-learn 支持 Pickle，因此 **不需要 ModelWrapperPickable**
model_bytes = pickle.dumps(rf_model)

# 从 testData 提取特征
Xtest = testData.select(spam_names[:-1])

# 定义新架构（增加预测列）
pred_field = [StructField("prediction", DoubleType(), True)]
new_schema = StructType(Xtest.schema.fields + pred_field)

# 定义 `predict` 函数
def predict(iterator):
    # 反序列化模型
    model = pickle.loads(model_bytes)
    for features in iterator:
        preds = model.predict(features)
        yield pd.concat([features, pd.Series(preds, name="prediction")], axis=1)

# 使用 `mapInPandas()` 进行分布式预测
prediction_sklearn_df = Xtest.mapInPandas(predict, new_schema)

# 提取预测值并转换为 NumPy 数组
ypred_sklearn = prediction_sklearn_df.select('prediction').toPandas().values

# 提取测试集真实标签
testDataPandas = testData.select("*").toPandas()
ytest = testDataPandas.iloc[:, -1].values

# 计算准确率
accuracy = accuracy_score(ytest, ypred_sklearn)
print(f"Scikit-learn RandomForest Accuracy: {accuracy:.4f}")
