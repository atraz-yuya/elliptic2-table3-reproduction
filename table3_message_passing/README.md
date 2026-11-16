
# Elliptic2 Table 3 のメッセージパッシング4手法

GCN、GraphSAGE、GAT、GIN は観測グラフごとに1 logit を返し、サブグラフラベルで end-to-end に学習する。Suspicious を正例 (1) とし、sigmoid を適用したスコアを保存する。実装と出力はこのディレクトリにある。

論文の Table 2 に合わせて対象集合・観測グラフを復元した実験は [README_paper.md](README_paper.md) を参照。

## 実行

リポジトリルートから、WSL の既存 `.venv-gnn-v1` を使用する。Python 3.11 以上と NumPy、pandas、scikit-learn、PyTorch が必要。実験時の正確なバージョン、コマンド、コードと入力の SHA-256 は `results/experiment.json` および `results/commands.jsonl` に保存する。

```bash
.venv-gnn-v1/bin/python -m pytest -q -p no:cacheprovider table3_message_passing/test_smoke.py
.venv-gnn-v1/bin/python -u table3_message_passing/run.py --device cuda
.venv-gnn-v1/bin/python table3_message_passing/run.py --device cuda --model gin --fold 1
.venv-gnn-v1/bin/python table3_message_passing/run.py --aggregate-only
```

同じ出力先で再実行すると、完了済みのモデル・foldを再利用する。実行中だったfoldは最初から学習する。入力、コード、実行環境を変える場合は `--out-dir` で新しい出力先を指定する。1つの出力先に対して1プロセスが実行できる。失敗時はスタックトレースを保存し、同じ設定で1回再試行する。

## 論文と対象集合

[Buu & Kim (2026)](https://doi.org/10.3390/math14213924) の提供PDF（SHA-256 `bb5480425721d38ce845bc1d041326e6f5f232b2d750aef0cd5fe10d29df57c0`）の §§3.1–3.2、§4.1、Tables 2–3 を参照した。PDFの Data Availability Statement では、論文を裏付ける生データは著者への請求により提供される。提供PDF、[出版社の論文一覧](https://www.mdpi.com/journal/mathematics/most-recent)、[研究室の論文一覧](https://aisalab.gnu.ac.kr/publication)、および題名・DOI・著者名による公開検索を確認した。対象ID、除外条件、著者のfold、Table 3 の著者コード・設定・補足ファイルは確認できなかった。

対象集合には [Elliptic2 公開版](https://github.com/MITIBMxGraph/Elliptic2) のラベル付き全121,810件 (Licit 119,047 / Suspicious 2,763) を採用した。論文の対象集合は97,448件 (Licit 95,249 / Suspicious 2,199) で、対象差は24,362件。以下のグラフ構築、分割、および実装の仮定を採用した公開記述に基づく再実装である。論文と実測値の平均差には、この対象集合と設定の差が含まれる。

実測の平均観測ノード数（Licit / Suspicious）は 9.13 / 9.36、平均有向辺数は 10.36 / 10.46。論文の Table 2 ではそれぞれ 6.51 / 6.82 と 6.00 / 6.24。参加者をfit側と共有する外側評価グラフの比率は全foldで100%（論文では97.16–97.63%）。各foldは学習98,666件、内側検証10,963件、外側評価12,181件。これらの差は、集計値を比較する際に考慮する。

## 観測グラフ

- 公開版の `nodes.csv` と `edges.csv` から対象メンバーと内部辺を取得した。境界辺は、背景辺の全196,215,606行を走査して作られた既存の `boundary_unique_neighbors.sqlite` を読み取り専用で使った。各内部メンバーの方向別件数を抽出時の台帳と照合した。観測グラフごとに順序付きノード対を一意化し、相互送金は両方向を保持する。外部間の辺とエッジ属性は対象外。
- 同じ外部クラスタが送金元と送金先になる場合、1ノードに funding / receiving の両フラグを立てる。グラフ間で同じクラスタが現れる場合、各グラフのノードと辺は個別にバッチ化する。
- 背景ノードの既存属性抽出結果から43属性を読み、キャッシュ間で共通するIDの属性一致を検証する。各foldのfitグラフに出る一意なクラスタで平均と母標準偏差を計算する。標準偏差が0の列はスケール1。3つの役割フラグは0/1のまま付加する。IDとラベルはモデル特徴から外す。

## 学習の設定

| 手法 | モデルと readout | 上限ステップ |
| --- | --- | ---: |
| GCN | 32次元×2、ReLU、全観測ノードの平均、32次元ReLUヘッド | 12,000 |
| GraphSAGE | 32次元×2、自己表現と全入近傍の平均を連結し線形変換、ReLU、層ごとのL2正規化、全観測ノードの平均 | 6,000 |
| GAT | 32次元×2、各層4ヘッド×8次元を連結、ELU、全観測ノードの平均 | 12,000 |
| GIN | 32次元×4、固定 eps=0、入力と各層で全観測ノードをsum pooling、層別logitの和 | 6,000 |

- 伝搬方向は送金元から送金先。GCNとGATは全ノードに自己ループを1本追加する。GCNの辺重みは自己ループ込みの入次数を両端に適用する。GATの加法attentionは受信ノード別、ヘッド別にsoftmaxし、LeakyReLUの負勾配は0.2。GraphSAGEとGINは近傍集約の外で自己表現を加える。取引の自己辺は伝搬時に除く。
- GINの各MLPは線形→BatchNorm→ReLU→線形で、MLP後にもBatchNormとReLUを適用する。層別logitに学習時だけ dropout 0.5 を適用する。4伝搬層と層別sum readoutは論文記載に基づき、入力層のreadoutとMLP・正規化・dropoutは原GINの構造を参考にした。
- 原実装は更新式の参照に使った。[GCNの自己ループと正規化](https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L110-L122)、[GraphSAGEのmean aggregator](https://github.com/williamleif/GraphSAGE/blob/master/graphsage/aggregators.py#L39-L59)、[GATのattention](https://github.com/PetarV-/GAT/blob/master/utils/layers.py#L5-L31)、[GINの層別readout](https://github.com/weihua916/powerful-gnns/blob/master/models/graphcnn.py#L177-L206)。
- Adam (lr=0.001)、非加重 BCEWithLogitsLoss、バッチ128、400更新ごとに検証。学習グラフは各エポックでシャッフルする。GraphSAGEの学習率・損失・上限、およびGCN/GATの上限・検証間隔は論文記載の値を使い、残りの最適化設定は実験仕様の仮定を適用した。GraphSAGEのreadoutとGATのヘッド内幅・結合も実験仕様の仮定を適用した。
- 外側は seed 42 の層化10-fold。外側学習候補をさらに層化し、約10%を内側検証に充てる (seed 42 + fold番号)。全手法の同じfoldで初期化seedも共通とする。学習側と外側評価側の参加者重複率を記録する。
- 内側検証APでcheckpointを選び、同値なら最初の更新を採用する。そのcheckpointの内側検証でF1最大のスコア閾値を選び、同値なら最大閾値を採用する。外側では score ≥ threshold を正例とし、APは scikit-learn の average_precision_score を使う。

## 出力

| 出力 | 内容 |
| --- | --- |
| `experiment.json`, `commands.jsonl` | 設定、実行コマンド、入力・分割・コード識別子、依存バージョン |
| `splits.npz`, `split_audit.json` | `graph_ids` と foldごとの fit / validation / test 行番号、クラス数、参加者重複率 |
| `graph_inventory.csv`, `composition.json` | 対象ID別のラベル・役割別ノード数・有向辺数と Table 2 との比較 |
| `<model>/fold_<nn>/` | 選択checkpoint、検証履歴、選択step・AP・閾値・混同行列・標準化係数、ID付き外側予測 |
| `fold_metrics.csv`, `summary.csv` | fold別結果、実測平均と標本標準偏差、論文値、10fold完了モデルの平均差 |
