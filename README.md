# 使い方
```sh
# Qwen3-0.6Bを実行します。
# 最大まで最適化された状態で実行します。コンパイルには数分から数十分かかります。
# ソースコードの変更がなければ、一度コンパイルすれば二度目以降は比較的高速になります。
CUDA=1 BEAM=2 uv run python -m tinylm chat unsloth/Qwen3-0.6B --dtype bfloat16

# llm-jp/llm-jp-3.1-1.8b-instruct4を実行します。
CUDA=1 BEAM=2 uv run python -m tinylm chat llm-jp/llm-jp-3.1-1.8b-instruct4 --dtype bfloat16

# tinygradで使用可能なバックエンドを表示します。
uv run python -m tinygrad.device
# PASSと表示されたものは、LLVM=1などの環境変数を利用することで使用できます。
# 現在のmain.pyでは、bfloat16の使用を前提としているため、CPUなどのbfloat16をサポートしないバックエンドでは動作しません。
# NV=1などのバックエンドをお試しください。
# CUDAの場合はCUDA、NVの場合はnvrtcが必要です。
# 他のバックエンドについては未検証であり、よくわからんエラーが出る可能性があります。コードは問題ないはずなので頑張ってください。
```

# 質問
[ローカルLLMに向き合う会](https://discord.gg/U2bNgWstTS) あたりにいる私に遠慮なくメンションして聞いてください。