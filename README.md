
---

# タイタニック生存予測の記録

このデータは、タイタニック号の生存予測に関するものです。以下はプロジェクトの主な部分とその内容です。

## 1. データのロードと前処理

### 1-1. データの読み込み
- `train.csv`と`test.csv`データセットをロードし、それぞれ`train_data`と`test_data`に格納します。
- トレーニングデータとテストデータを結合し、前処理を一括で行うために`all_df`として扱います。

```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
all_df = pd.concat([train_data, test_data], sort=False).reset_index(drop=True)
```
### 1-2. 汎用的なデータ前処理
#### 1-2-1. 欠損値の処理
- `Fare`列の欠損値を、`Pclass`（客室クラス）別の平均値で補完します。

```python
Fare_mean = all_df[["Pclass", "Fare"]].groupby("Pclass").mean().reset_index()
Fare_mean = Fare_mean.rename(columns={"Fare": "Fare_mean"})
all_df = pd.merge(all_df, Fare_mean, on="Pclass", how="left")
all_df.loc[all_df["Fare"].isnull(), "Fare"] = all_df["Fare_mean"]
all_df = all_df.drop("Fare_mean", axis=1)
```
#### 1-2-2. カテゴリ変数のエンコーディング
- `Sex`、`Embarked`、`Pclass`、`honorific`、`alone`のカテゴリ変数をラベルエンコーディングします。

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_df["Sex"] = le.fit_transform(all_df["Sex"])
categories = ["Embarked", "Pclass", "honorific", "alone"]
for cat in categories:
    if all_df[cat].dtypes == "object":
        le = LabelEncoder()
        all_df[cat] = le.fit_transform(all_df[cat])
```
### 1-3. タイタニックデータに特有の前処理
#### 1-3-1. 新しい特徴量の作成
- `Name`列を分割して`honorific`（敬称）を抽出し、年齢分布の分析に使用します。
- 家族の有無を示す`alone`列を作成します。

```python
name_df = all_df["Name"].str.split("[,.]", expand=True, n=2)
name_df = name_df.rename(columns={0: "family_name", 1: "honorific", 2: "name"})
all_df = pd.concat([all_df, name_df], axis=1)
all_df["alone"] = (all_df["SibSp"] + all_df["Parch"] == 0).astype(int)
```

### 1-4. データの分割
- トレーニングデータとテストデータを再度分割します。

```python
train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
train_Y = train_data["Survived"]
test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
```

## 2. 特徴量の選択とモデルのトレーニング

### 2-1. 特徴量の選択とコーディング
- 上記の前処理ステップで、モデルに適した特徴量を選択し、必要なエンコーディングを行います。

### 2-2. モデルのトレーニング
- LightGBMモデルを使用してトレーニングと評価を行います。

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2)

lgbm_params = {
    "objective": "binary",
    "random_seed": 1234
}

lgh_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature=categories, reference=lgh_train)

model_lgb = lgb.train(
    lgbm_params,
    lgh_train,
    valid_sets=lgb_eval,
    num_boost_round=100,
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(10)
    ]
)
```

## 3. 予測と提出ファイルの作成

### 3-1. 予測
- テストデータセットに対して予測を行います。

```python
y_pred = model_lgb.predict(test_X, num_iteration=model_lgb.best_iteration)
```

### 3-2. 提出ファイルの作成
- 予測結果を`submission.csv`というファイルに保存します。

```python
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': np.round(y_pred).astype(int)})
output.to_csv('submission.csv', index=False)
```

### 3-3. 提出ファイルのダウンロードリンク
- `submission.csv`ファイルをダウンロードするためのリンクを提供します。

```python
from IPython.display import FileLink
FileLink('./submission.csv')
```

---
