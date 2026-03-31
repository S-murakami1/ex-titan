# ex-titan

ConCH のパッチ特徴が入った HDF5 から、[MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) でスライド埋め込みを計算し、同じ H5 の `titan/features` に書き戻します。

- メンテナ: [S-murakami1](https://github.com/S-murakami1)

## 環境

[uv](https://docs.astral.sh/uv/) を想定。

```bash
uv sync
uv run python extract_feratures.py /path/to/h5_directory
```

ワンライナー例は `comand.txt` を参照。

## Hugging Face

`huggingface-cli login` または環境変数 `HF_TOKEN`。

## リポジトリを GitHub に載せたあと（例）

リポジトリ名を `ex-titan` とした場合の remote 例:

```bash
git remote add origin https://github.com/S-murakami1/ex-titan.git
git branch -M main
git push -u origin main
```

SSH を使う場合: `git@github.com:S-murakami1/ex-titan.git`
