[![DeepWiki](https://img.shields.io/badge/DeepWiki-neodyland%2Ftinylm-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/neodyland/tinylm)
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
