# Sector Lead-Lag Analysis Tool

米国業種ETFの日次リターンから日本業種ETFの翌日リターンを予測する、部分空間正則化PCAベースのリード・ラグ分析ツールです。

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 本日のシグナル生成

```bash
python run.py signal
```

`output/signal_YYYYMMDD.json` にロング・ショート候補が出力されます。

### バックテスト

```bash
python run.py simulate --start 2020-01-01 --end 2025-12-31
python run.py simulate --start 2020-01-01 --end 2025-12-31 --plot
```

`output/simulation_result.csv` と `output/simulation_plot.png` が保存されます。

## 設定

`config.yaml` で以下のパラメータを調整できます:

- `lambda`: 正則化パラメータ (0.0〜1.0)
- `n_components`: 使用する主成分数
- `rolling_window`: ローリングウィンドウ（営業日数）
- `top_n`: サジェストする業種数

## 免責事項

- 本ツールは研究・学習目的のOSSです
- 投資判断は自己責任で行ってください
- バックテスト結果は将来のパフォーマンスを保証しません

## 参考文献

- Subspace Regularized PCA を用いた業種間リード・ラグ構造の分析手法に基づく実装
