# 環境構築手順
```
# https://huggingface.co/docs/lerobot/installation
uv python install 3.12
uv venv --python 3.12

source .venv/bin/activate

git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .
uv pip install 'lerobot[smolvla,pi,peft]'

wandb login
```

# airoa-momaデータセットの加工
`pi05` を `airoa-moma` で学習する前に、`prepare_pi05_dataset.py` を一度実行して、`pi05` 用の派生データセットを作成する必要があります。

理由は、`airoa-moma` のアクション列名と、`pi05` が学習時に参照するアクションキーが一致していないためです。

- `airoa-moma` 側の主なアクション列: `action.relative` or `action.absolute`
- `pi05` 側が期待するキー: `action`

このままでは、`pi05` の学習コードが `action` を見つけられず、forward 時にエラーになります。

この問題を `lerobot` 本体の改変で吸収するのではなく、データセット側で整形して解決するのがこのスクリプトの役割です。これにより、upstream の `lerobot` を変更せずに運用でき、将来のアップデートにも追従しやすくなります。

`prepare_pi05_dataset.py` は、元の `airoa-moma` を直接変更せず、`pi05` 用の派生データセットを別ディレクトリに作成します。主な処理は以下のとおりです。

- parquet 内の `action.relative` or `action.absolute` 列を `action` として書き出す
- `meta/info.json` 内の feature 定義を `pi05` 向けに更新する
- `meta/stats.json` 内の統計情報のキーも同じように更新する
- 元データセットは保持したまま、派生データセットだけを学習に使用する

つまり、このスクリプトは「追加の前処理」ではなく、`airoa-moma` を `pi05` が読める形に合わせるための互換変換です。

```bash
python airoa-evaluation-ICRA/scripts/prepare_pi05_dataset.py
```

生成後は、`training_config_pi05_airoa_moma.json` で指定している派生データセットを使って、そのまま `pi05` の学習を開始できます。
