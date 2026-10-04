# Strata セットアップ手順（日本語）

GTX 10 シリーズ（Pascal）で実際にインストール・起動・LAN 公開まで通した手順を、はまった箇所込みでまとめたもの。
英語の正式ドキュメントは [docs/DETAILS.md](DETAILS.md)。こちらは実測メモなので、数値やパスはこのマシン固有。

## 1. 検証マシン（実測環境）

| 項目 | 値 |
|---|---|
| OS | Linux (Ubuntu 22.04 系) |
| GPU | NVIDIA GeForce GTX 1080 Ti（11264 MiB、compute capability 6.1 = `sm_61`） |
| ドライバ | 550.107.02 |
| CUDA Toolkit | 12.6（`/usr/local/cuda-12.6`、nvcc 12.6.68） |
| メモリ | 62 GB |
| ディスク | /home に 1.8 TB（IQ3_XXS 一式で約 79 GB。IQ2_XS を足すと 118 GB、Q2_0 を足すと約 180 GB） |
| Strata engine | 0.1.33、ローカルビルド（`archs: [61]`、`sm60: true`、`vision: "cpu"`） |
| モデル | Qwen3.8-Flash-Next / **Q2_0**（IQ3_XXS → IQ2_XS → Q2_0 と載せ替えた）、context 32768、vision cpu |

## 2. 前提条件

- **GPU**
  - 公式サポートは **RTX 20 系以降（compute capability 7.5 以上、VRAM 8 GB 以上）**。
  - GTX 10（Pascal `sm_60/sm_61`）と Volta（`sm_70`）は **`--experimental-sm60`** を付けたときだけ通る
    コミュニティビルド。上流はサポートしない（#236）。
- **ドライバ**
  - 通常構成（CUDA 13）: **580 以上**。
  - `--experimental-sm60`: **525 以上**でよい（CUDA 12.0 以上があれば足りる。550.107.02 + CUDA 12.6 で動作確認済み）。
- **CUDA Toolkit**
  - Pascal / Volta には **CUDA 12.x が必須**。CUDA 13 の nvcc は `compute_60/61/70` を削除しているため、
    SM60 ビルドは CUDA 13 では作れない。`setup.py` は SM60 のとき 12.6 を選ぶ（未インストールなら導入を案内する）。
- **メモリ**: 下表の「必要 RAM」より小さいと setup が警告し、小さいサイズを勧める。`arena` は GPU に常駐させる
  expert の分で、この分がシステム RAM を圧迫する（IQ3_XXS では約 47 GB の expert を RAM に読み、サーバーが確保する
  RAM は約 40 GB だった）。
- **ディスク**: 70〜120 GB。`Strata-data` は既定ではリポジトリの**隣**に作られる。

モデルサイズ（`setup.py` の `MODELS` より）:

| サイズ | 特徴 | ダウンロード | 必要 RAM | expert arena |
|---|---|---|---|---|
| `Q2_0` | 2-bit、最速 | 66.4 GB | 48 GB | 34.0 GB |
| `IQ2_XS` | 2-bit i-quant、少し高品質で速度はほぼ同じ | 68.0 GB | 48 GB | 35.5 GB |
| `IQ3_XXS` | 3-bit i-quant、高品質だが CPU 側が遅い | 75.8 GB | 60 GB | 42.9 GB |
| `IQ3_S` | 3.5-bit、最高品質・最遅。64 GB PC 推奨 | 83.6 GB | 62 GB | 50.3 GB |
| `IQ1_M` | Coder 版専用（expert の半分） | 58.4 GB | 32 GB | 23.4 GB |

## 3. セットアップ（1 回目）

```bash
git clone <このリポジトリ> && cd Strata

# 対話で 4 問（モデル / サイズ / context / 画像）に答える
./setup.sh --experimental-sm60

# 既定値で進めるなら
./setup.sh --experimental-sm60 --model IQ3_XXS --context 32768 --yes
```

> このマシンで最終的に使っているのは **Q2_0** の方（`--model Q2_0 --kv int8 --vision cpu`）。
> GTX 10 級で VRAM が 11 GB しかない PC では expert の常駐数が速度を決めるが、それを越えたところで
> **CPU 側の 1 行あたりの復号コスト**が効いてくる。i-quant でない 2-bit（Q2_0）が最速だった。
> IQ3_XXS → IQ2_XS → Q2_0 の比較実測は 9.3 節と 9.6 節。

`setup.sh` は Python 3.10+（venv と pip が使えるもの）が無ければ apt/dnf で入れるだけの薄いラッパ（sudo を聞く
ことがある）。Windows は `START-HERE.bat` が同じ `setup.py` を呼ぶ。

実行されるステップ（**すでに済んだものはスキップ**される。中断してもう一度実行すると続きから進める）:

1. PC チェック（GPU・ドライバ・RAM・CPU・ディスク）
2. 質問
3. `.venv` に Python パッケージ（numpy、jinja2、NVIDIA CUDA ライブラリ等）
4. エンジンの入手 — **RTX 20 以降はプリビルド版を使うが、`--experimental-sm60` では必ずローカルコンパイル**（10〜20 分）
5. Hugging Face からモデルの GGUF をダウンロード（再開可能）
6. Strata 用にモデルを準備（pack）＋ MTP ドラフト層を取得（約 5 GB）
7. `run-<model>.sh` を書いて起動

SM60 で実際に残るログ（抜粋）:

```
Compiling the Strata engine for your GPU (10-20 minutes, once) ...
[ok] Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf downloaded
[ok] Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf downloaded
[ok] model prepared: .../Strata-data/packs/iq3_xxs
```

## 4. 何が生成されるか（このマシンの実測）

リポジトリの隣にデータフォルダ、リポジトリ内に設定と起動スクリプト:

```
Workspace/
├── Strata/                     # リポジトリ
│   ├── .venv/                  # Python 環境（cmake・ninja もここに入る）
│   ├── engine/strata           # ビルドしたエンジン
│   ├── engine/strata-vision    # 画像エンコーダ（vision を有効にしたとき増える）
│   ├── engine/BUILD.json       # どの構成でビルドしたかの記録
│   ├── build/CMakeCache.txt    # STRATA_EXPERIMENTAL_SM60:BOOL=ON が入る
│   ├── strata-q2_0-fast.json   # 起動設定（いま動いているもの。gitignore 済み）
│   ├── run-q2_0.sh             # 起動スクリプト（gitignore 済み）
│   ├── strata-q2_0-vision.json # 上の設定に `--vision` を足したもの（画像入力あり、9.9 節）
│   ├── run-q2_0-vision.sh      # 同上の起動スクリプト
│   ├── strata-q2_0.log         # エンジンのログ（毎リクエストの tok/s、cache hit rate が出る）
│   ├── strata-iq2_xs.json / run-iq2_xs.sh / strata-iq2_xs.log   # 途中まで使っていた IQ2_XS
│   └── strata-iq3_xxs.json / run-iq3_xxs.sh / strata-iq3_xxs.log   # 先に入れておいた IQ3_XXS
└── Strata-data/                # モデル本体（IQ3_XXS + IQ2_XS + Q2_0 で約 180 GB）
    ├── models/IQ3_XXS/         # 71 GB  GGUF の 2 分割シャード（44 GB + 27 GB）
    ├── models/IQ2_XS/          # 39 GB + 29 GB（shard 2 は IQ3_XXS とハードリンク共有）
    ├── models/Q2_0/            # 37 GB + 29 GB（shard 2 も同じハードリンク共有）
    ├── models/mmproj-*.gguf    # 0.91 GB  画像エンコーダ（vision）
    ├── packs/iq3_xxs/          # 1.5 GB pack（dense 重み・expert の索引・トークナイザ）
    ├── packs/iq2_xs/           # 同上（IQ2_XS 用）
    ├── packs/q2_0/             # 同上（Q2_0 用）
    └── mtp/                    # 6.5 GB MTP ドラフト層（サイズ間で共有、再ダウンロードなし）
```

`engine/BUILD.json`（SM60 と vision の記録がここに残る）:

```json
{
  "source": "local",
  "version": "0.1.33",
  "archs": [61],
  "vision": "cpu",
  "cuda_dirs": ["/usr/local/cuda-12.6/bin", "/usr/local/cuda-12.6/lib64"],
  "src": "bd5219cbd261b225",
  "vision_src": "4d89409b228a1daa",
  "sm60": true
}
```

**`"sm60": true` が重要**：これがあるおかげで、2 回目以降は `--experimental-sm60` を付けなくても
（`./setup.sh` だけでも、`run-q2_0.sh` だけでも）同じビルドとして認識され、GPU チェックを通過する。

## 5. 起動・停止

```bash
./run-q2_0.sh             # 起動（約 34 GB の expert を RAM に読む）
./run-q2_0-vision.sh      # 同上 + 画像入力（`--vision`。速度は 9.9 節で実測）
./run-iq2_xs.sh           # 前に使っていた IQ2_XS 版（約 36 GB）
./run-iq3_xxs.sh          # 先に入れておいた IQ3_XXS 版（約 47 GB）
./setup.sh                # どれを起動するかの一覧が出る（既定は最後に作った方）
```

- 初回ロードは **1〜3 分**固まる。ログに
  `YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.`
  と出るので待つのみ。
- 準備できると:

  ```
  ready: http://127.0.0.1:8080/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages,
        context 32768 tokens, API key required)
         open http://127.0.0.1:8080/ in a browser to chat
  ```

- `http://127.0.0.1:8080/` にブラウザでチャット UI がある。API は OpenAI 互換（`/v1/chat/completions`）と
  Anthropic 互換（`/v1/messages`）の両方。
- 停止は端末を閉じるか `Ctrl+C`。裏で動かすなら:

  ```bash
  nohup ./run-q2_0.sh > /tmp/strata-server.log 2>&1 &
  pgrep -af 'serve/server.py|engine/strata'      # 確認
  kill <PID>                                     # 停止（エンジン子プロセスも一緒に終わる）
  ```

  停止後は RAM が数秒で戻る（`free -g` で確認）。

## 6. LAN 越しにアクセスする（他の PC から使う）

コードの変更は不要。設定ファイル `strata-q2_0.json` に 2 つ足すだけ
（`setup.py --host 0.0.0.0 --api-key <鍵>` でも同じ）:

```json
  "port": 8080,
  "gpu": 0,
  "gpus_asked": true,
  "host": "0.0.0.0",
  "api_key": "<自分で決めた鍵>"
```

再起動すると、LAN のアドレスが起動ログに出る:

```
ready: http://127.0.0.1:8080/v1  (… API key required)
       from other devices: http://192.168.75.24:8080/   (API: http://192.168.75.24:8080/v1)
```

クライアント側:

```python
from openai import OpenAI
c = OpenAI(base_url="http://192.168.75.24:8080/v1", api_key="<鍵>")
```

- 鍵は `Authorization: Bearer <鍵>` でも `x-api-key: <鍵>` でも送れる（サーバー側は定数時間比較）。
- ブラウザのチャット UI は `http://<IP>:8080/` で開き、鍵を聞かれる。
- Claude Code など Anthropic 系は `ANTHROPIC_BASE_URL=http://<IP>:8080` + `ANTHROPIC_AUTH_TOKEN=<鍵>`。
- 鍵なしで叩くと `401 authentication_error`。

このマシンでの確認結果:

| 確認 | 結果 |
|---|---|
| リッスン | `0.0.0.0:8080` |
| 鍵なし `/v1/chat/completions` | 401 |
| 鍵あり `/v1/models` | 200（`status: loaded`） |
| LAN IP 経由の推論 | 正常に応答 |
| チャット UI を LAN 経由 | 200 |

注意:

- **HTTP 平文**。鍵も会話内容も LAN に平文で流れるので、信頼できる LAN 内に限定すること。
  インターネット公開はトンネル（cloudflared 等）＋鍵を使う（[DETAILS.md](DETAILS.md) 参照）。
- 鍵は `strata-*.json` に平文で保存される（このファイルは gitignore 済み）。
- ローカル専用に戻すには `"host": "127.0.0.1"` に変えて再起動。
- 繋がらないときはファイアウォール（`sudo ufw status` / `sudo nft list ruleset`）とルータのクライアント分離を確認。
  `0.0.0.0` にすると Docker の `docker0`（172.17.0.1）にも listen する。
- `--api-key` に空文字を渡すと起動時にエラーになる（空の鍵は拒否される仕様）。

## 7. 実測性能（GTX 1080 Ti + 62 GB RAM）

ベンチマークではなく、このマシンで体感した範囲のメモ。同じ 6562 token の文脈を 2 回投げて、
1 回目（プロンプトを一から読む）と 2 回目（前回の文脈を再利用）を分けてある。

| 場面 | IQ3_XXS | IQ2_XS | IQ2_XS + 調整後 | **Q2_0 + 調整後** |
|---|---|---|---|---|
| VRAM に常駐する expert | 2794 / 24576 | 3815 / 24576 | 3815 / 24576 | **3933** / 24576 |
| expert cache ヒット率（コールド → 暖まった後） | 57.6% → 69.4% | 62.5% → 71.7% | 67〜72% → 77〜80% | 71% → **78〜83%** |
| プロンプト処理（6562 token、一から） | 199〜212 tok/s | 199 tok/s | 200 / 313 tok/s | 205 / **320** tok/s |
| 出力（6562 token の文脈で 128 token、コールド） | 6.0〜6.5 tok/s | 5.9 tok/s | 6.6 / 12.5 tok/s | 7.9 / **17.1** tok/s |
| 出力（同じ 6562 token の文脈を再利用） | 7.5 / 19.7 tok/s | 8.4 tok/s | 7.4 → 10.1 / 10.4 | 13.8 → **30.2** tok/s |
| 出力（短い回答） | 10 tok/s 前後 | 7 tok/s 前後 | 7.6 tok/s | 8.2 tok/s |
| 出力（直前の会話を再利用した短いプロンプト） | 17〜22 tok/s | 21.5〜22.5 tok/s | 15.1 / 21.1 / 27.2 | 16.5 / **29.4** tok/s |

「調整後」は `./setup.sh --calibrate` が選んだエンジン設定を効かせたもの。Q2_0 列は IQ2_XS からの流用値
（`--pcie-frac 0.35 --spec-min-p 0.70`）で測ったもので、Q2_0 向けに測り直した値（`0.55 / 0.70 / 2 workers`）との
比較が 9.7 節。いま config に入っているのは後者。
いま動いているのは **Q2_0** で、載せ替えた理由と品質の落ち方は 9.6 節、IQ2_XS までの経緯は 9.3〜9.5 節。

**ばらつきが大きい**点に注意。同じ条件の再実行で 7.5 と 19.7 tok/s が出ている。効いているのは
（a）その文脈が実際にどの expert を引くか（ヒット率）、（b）MTP ドラフトの採用率
（`drafts accepted 62 of 88` など。採用率が 1 割違うだけで出力速度が 1 割変わる）で、
どちらも文脈の中身で動く。上の表は 2〜3 回程度の平均ではなく個別の実行値。

- 出力速度を律速しているのは GPU 単体の計算ではなく、**RAM／CPU からの expert ストリーミング**。
  Pascal は Turing 以降のカーネルパスを使えないため、RTX 20 系以降より遅くなる前提。
- 数値は他のプロセス・コンテキスト長・expert のヒット率で大きく動く。自分の PC を測る場合は
  [docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) の手順で。
- 上の数値はエンジンログ `strata-iq2_xs.log` / `strata-iq3_xxs.log` から取ったもの。毎リクエストの速度と、
  MTP ドラフトの採用率・expert キャッシュのヒット率が出る:

  ```
  strata serve: prompt 6562 tokens = 0 reused + 6562 read in 32935 ms (199.2 tok/s), 128 generated in 21813 ms (5.9 tok/s), drafts accepted 62 of 88, 1 checkpoints
  strata serve: decode expert cache hit rate: 62.5% (43524 hits / 69664 lookups)
  ```

- 起動ログに `this CPU has no AVX-512: the expert kernels run on AVX-2` と出る。このマシンは AVX-512 非対応で、
  expert の gate/up 行のカーネルが AVX-2 で動く（出力速度に影響する）。
- 同じく起動ログに `expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable ... using 4 KB pages`。
  expert を置く RAM は pinned memory として登録されるが、この環境では hugetlb プールが無いため 4 KB ページで動いている。
  PCIe は `12.9 GB/s host->device` と実測されている（expert を GPU に流す帯域）。
- `./setup.sh --calibrate` でこの PC に合わせたエンジン設定のチューニング（5〜10 分）ができる。

## 8. 画像を入力できるようにする（vision）

画像入力は `--vision` で有効にする。エンコーダ（llama.cpp の mtmd + mmproj）はエンジン本体とは別の
常駐プロセス `engine/strata-vision` で動き、画像はハッシュでキャッシュされる（同じ画像を毎ターン
送ってきてもエンコードは 1 回）。

```bash
./setup.sh --setup --model Q2_0 --context 32768 --kv int8 --vision cpu \
  --host 0.0.0.0 --api-key <key> --port 8080 --gpu 0 --yes
```

- `--vision gpu`（`yes` と同義）: エンコーダを GPU で。1 画像あたり最大 **1024** image tokens、
  VRAM に約 1.2 GB の空きが必要で、エンジン側は `--vram-reserve-mib 700` を予約する。
  GTX 1080 Ti の 11 GB は expert と重みで埋まっていて空きが 0.5 GB しかないため、このマシンでは選ばない。
  **expert cache のスロットが減るので速度も下がる**（9.9 節）。
- `--vision cpu`: CPU でエンコード。既定は 1 画像あたり最大 **300** image tokens、スレッド 2（4 スレッドの PC ため）。
  Pascal 向けの CUDA エンコーダは上流で未検証なので、こちらが安全。
  エンコーダは VRAM を 1 バイトも使わないので、**expert cache のスロットは減らない**（9.9 節で実測）。
- 増えるもの: `engine/strata-vision`（7.6 MB。CPU ビルドなら数分でコンパイル）、
  `../Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf`（0.91 GB）。
  **本体エンジンの再ビルドは不要**（`engine/BUILD.json` の `vision` が `cpu` になるだけ）。
  0.1.33 ではこのエンコーダが GPU ツールチェーン無しでビルドできるものになった（#411 #412）。
- config に増えるキー: エンジン args の `--vision --vram-reserve-mib 700` と、`"vision"` ブロック
  （`exe` / `mmproj` / `model` / `gpu: false` / `max_tokens: 300` / `threads: 2`）。
  `--model` には**テキストモデルの 1 分割目**を入れる（エンコーダは語彙だけ必要で、重みは読まない）。
  `--vram-reserve-mib 700` は `setup.py` が **gpu/cpu 問わず**付ける値で、これは GPU エンコーダの分の予約。
  `gpu: false` なら予約は不要なので、自分で config を書く場合は 9.8 節で調整した **681** を残したほうが
  expert スロット 19 個ぶん得をする（9.9 節）。
- 使い方:
  - ブラウザのチャット画面に画像を添付
  - `chat.py` で `/image <path>`
  - API: OpenAI 形式の `image_url` part、Anthropic 形式の `image` block（base64）
- 有効かの確認: `/v1/models` の `architecture` に `input_modalities` が入る（vision 無しのときは `["text"]`）:

  ```json
  {"id": "qwen3.8-flash-next-q2_0", "status": {"value": "loaded"},
   "meta": {"n_ctx": 32768},
   "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}}
  ```

### 8.1 実際に画像を送った結果

640x360 の PNG（赤い円・青い四角・黒文字）を OpenAI 形式の `image_url`（base64 data URL）で送った
（この測定は IQ2_XS のときのもの。Q2_0 では 9.6 節の速度になる）:

| 回 | 内容 | 実測 |
|---|---|---|
| 1 回目 | 画像エンコード + プロンプト 261 token を一から | wall 28.6 s、prompt 67 tok/s、出力 7.2 tok/s |
| 2 回目（同じ画像） | 画像はハッシュキャッシュ、prompt 256 token 再利用 | wall 5.6 s、prompt 24.6 tok/s、出力 22.5 tok/s |
| 3 回目（同じ画像） | 同上 | wall 7.3 s、153 token を 21.5 tok/s |

返答は正しかった（`The picture contains a red circle, a blue square, and black text on a light background.`）。
reasoning 側にも `It has red circle on left, blue square on right, background light lavender` と出ていて、
実際に画像を読んでいるのが分かる。

- 画像 1 枚で増える prompt token は **256 前後**（`--vision cpu` の上限 300 に収まっている）。
- 1 枚目の 28.6 s のうち数秒が CPU エンコード。同じ画像を毎ターン送る分は 2 回目以降消える。
- `max_tokens` を小さくしすぎると思考（`reasoning_content`）だけで使い切られ、`content` が `null` になる。
  画像の要約でも 300〜500 は欲しい。

### 8.2 画像入力の疎通確認（スクリプト）

サーバーを起動したまま、1 枚の PNG を投げて返答を見る:

```bash
.venv/bin/python tools/vision_smoke.py            # 64x64 の赤い四角を生成して送る
```

```
health: images=True
answer: 'Red'
usage: prompt=77 completion=41
PASS
```

`/health` の `images` が `true`、`/v1/models` の `input_modalities` が `["text", "image"]` に変わっていれば
有効になっている。`max_tokens` を 32 にすると `content: null`（`reasoning_content` には
`Image shows red square` と出ている）になるので、疎通確認では 128 以上を渡す。

## 9. 出力を速くする

### 9.1 まず診断する（このマシンでは VRAM の空きがボトルネックだった）

エンジンログに構成が全部出る:

```
strata generate: expert cache auto: 5.45 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 2111 slots
strata generate: expert cache 2794 slots, 4.58 GiB of VRAM
strata generate: profile data/expert-profile.bin: 24576 ranked pairs, built for 24576 slots
strata serve: the prompt path borrows 2128 CUDA0 cache slots (3.48 GiB)
strata serve: 519 MiB of VRAM free with everything loaded
strata serve: decode expert cache hit rate: 57.6% (36374 hits / 63177 lookups)
```

- 11 GB の VRAM には dense 重み・グラフ・prefill バッファが先に入り、expert 用に残るのは約 5.5 GiB
  → **24576 個中 2794 個（約 11%）しか常駐できない**。
- 外れた expert は CPU で計算する。この PC は **Core i7-6700（4 スレッド、AVX2）** なので、そこが律速になる。
- 同じモデルでも文脈長で速度が大きく違う（IQ3_XXS の実測）:

| 場面 | 出力速度 |
|---|---|
| 直前の会話を再利用（新規 token 5 程度） | 17〜22 tok/s |
| 6562 token を一から読む | 6.5 tok/s（プロンプト 211.6 tok/s、expert ヒット率 57.6%） |

つまり「遅い」の主因は GPU 計算ではなく **VRAM に載らない expert を 4 スレッドの CPU で計算している**こと。

### 9.2 試して効かなかった手

- `--kv k8v4`（INT8 K + 4bit V、816 B/cell）: expert slots が 2794 → **2854（+2%）** だけで、
  出力 6.5 → 6.0 tok/s と変化なし。32768 context では KV は VRAM の大半を占めない
  （この設定が効くのは 128K 以上の context）。`--kv q4_0` も同じ理由で期待薄。
- `--max-context` を下げて KV を削る手もあるが、32768 を保ったまま進めた。

### 9.3 効くのは「expert そのものを小さくする」

| サイズ | download | RAM | 特徴 |
|---|---|---|---|
| Q2_0 | 66 GB | 約 34 GB | 最速 |
| IQ2_XS | 68 GB | 約 36 GB | Q2_0 とほぼ同じ速さ、品質は少し上 |
| IQ3_XXS | 76 GB | 約 43 GB | 品質重視、遅い（1 token あたりの CPU 仕事が多い） |
| IQ3_S | 84 GB | 約 50 GB | 最高品質、最遅（64 GB PC で他を止める前提） |
| IQ1_M（Coder 系） | 58 GB | 約 32 GB | expert が半分（コード・ツール・画像は維持）。コード用途なら最速 |

2-bit にすると (1) expert の blob が小さいので同じ VRAM に多く載りヒット率が上がる、
(2) CPU で計算するときの 1 行あたりの仕事も減る、(3) RAM の余裕も増える。
shard 2（約 27 GB）はサイズ間で同一ファイルなのでハードリンク共有され、
追加ダウンロードは shard 1 だけ（IQ2_XS で 39.23 GB、実測 10〜15 MB/s で 40〜60 分）。

```bash
./setup.sh --setup --model IQ2_XS --context 32768 --kv int8 --vision cpu --yes
```

**実際に IQ2_XS へ入れ直した結果**（shard 2 共有で追加ダウンロードは 39.23 GB のみ、10〜17 MB/s で約 40 分）:

| 項目 | IQ3_XXS | IQ2_XS |
|---|---|---|
| expert arena が確保した VRAM | 4.8 GiB → 2794 slots | 5.13 GiB → **3815 slots** |
| expert blob 1 個 | 約 1.72 MB | 約 1.41 MB |
| ヒット率（コールド） | 57.6% | 62.5% |
| 出力（長い文脈を再利用） | 7.5 tok/s | 8.4 tok/s |
| 出力（長い文脈を一から） | 6.0 tok/s | 5.9 tok/s |

期待したほど速くならなかった。理由は 3 つ:

1. expert の常駐数は +37% 増えたが、ヒット率は +5 ポイントしか上がっていない。この文脈で引かれる
   expert の分布が偏っていて、VRAM 側を増やしても CPU 側の仕事が残る。
2. IQ2_XS は 2-bit の i-quant で、AVX-512 の無いこの CPU では i-quant の gate/up 行を AVX-2 の
   マルチトークンパスで計算する。blob が小さくても 1 行あたりのデコードコストが下がりきらない。
3. おまけに MTP ドラフトの採用率が落ちた（IQ3_XXS で 68/88、IQ2_XS で 62/88）。ドラフトが外れるほど
   1 窓あたりの生成 token が減るので、これだけで 1 割近く遅くなる。

つまり **このマシンでは「サイズを落とす」だけでは速くならない**。効くとすれば 9.4 の PC 依存設定
（特に `--pcie-frac`。PCIe が 13 GB/s あり、CPU で計算するより GPU に送る方が安くなる可能性がある）
と、`--pool-workers`（既定 3 スレッド → 4 スレッド全部使う）で、`--calibrate` がそれを測る。

### 9.4 PC 依存の設定を実測で合わせる

```bash
./setup.sh --calibrate                                      # 設定を測って config に保存
python tools/calibrate.py strata-iq2_xs.json                # 測るだけ
```

測るのは 3 つ（`tools/calibrate.py`）:

| 設定 | 意味 |
|---|---|
| `--pcie-frac` | VRAM に無い expert のうち、PCIe で GPU に送って計算する割合。この PC は host->device 12.9 GB/s 実測で auto が 0.27（既定 0.55） |
| `--spec-min-p` | ドラフト層が verify 窓をもう 1 行伸ばす確率の下限（既定 0.5）。遅い CPU は高い方が得なことがある |
| `--pool-workers` | CPU expert プールのスレッド数。既定は物理コアからホストスレッドを引いた数（4 スレッドなら 3 前後）。**測るのは既定値より少ない方だけ**（2/3 と 1/2）なので、増やしたいときは自分で config に書く。値ごとにエンジン再起動 |

`--pcie-frac` と `--spec-min-p` はエンジン再起動なしで計測でき、既定より 3% 以上速い場合だけ採用される。
結果は PC とモデルごとに保存され、`--setup` で入れ直しても引き継がれる。

このマシン（IQ2_XS）での実測（約 30 分）:

| 試した値 | 出力 tok/s |
|---|---|
| `--pcie-frac` 0.00 / 0.20 / 0.27（auto） / **0.35** / 0.55 / 0.75 | 16.6 / 15.5 / 21.5 / **23.0** / 19.1 / 22.0 |
| `--spec-min-p` 0.30 / 0.50（既定） / **0.70** | 19.7 / 19.5 / **25.0** |
| `--pool-workers` **3（既定）** / 2 | **22.5** / 21.6 |

採用されて config に書き込まれたのは `--pcie-frac 0.35 --spec-min-p 0.70`。`--pool-workers` は既定の 3 が
最速だったので変えていない（増やす方向は測らないので、4 にしたいなら自分で config に書く）。

**この表は鵜呑みにしない方がいい**。同じ sweep の中で 15.5 と 23.0 が出ているが、`--pcie-frac` を
0.20 にしても 0.75 にしても計算量の総量はあまり変わらないはずで、差が大きすぎる。原因は測る順序だと
思われる: sweep は小さい値から順に、同じ 3 プロンプトを繰り返すので、後ろに測った値ほど
expert キャッシュと prompt cache が温まった状態で測られる。`--pcie-frac 0.00`（PCIe を一切使わない）が
16.6 と最低に出ているのも、順序のせいで最初に測られたからかもしれない。
採用された 0.35 / 0.70 が auto の 0.27 / 0.50 より速いかは、下の 9.5 でサーバーを再起動して測り直した。

エンジンに直接渡せる他の速度設定（`strata generate --help` より）:

| 設定 | 意味 |
|---|---|
| `--expert-cache N` | VRAM に置く expert 数（既定 auto、`--expert-cache-per-layer` で層ごとに枠を配分） |
| `--suffix-draft N` | 直前の繰り返しからドラフトする prompt lookup（既定 3、0 で MTP のみ） |
| `--spec N` | MTP ドラフトの深さ（既定 4） |
| `--prompt-cache N` | リクエスト間に保持する会話チェックポイント数（既定 6、1 個約 118 MB） |
| `--conversation-cache-mib N` | 退避した会話の RAM バジェット（既定 0 = off） |
| `--short-read N` | 新規 N token まで batched prompt ではなくデコード窓で読む（既定 64、0 で off） |
| `--ple-io direct\|mmap\|ram` | n-gram テーブルの読み方（既定 direct。`ram` はテーブルごと RAM にロック） |

### 9.5 調整後（`--pcie-frac 0.35 --spec-min-p 0.70`）を測り直した

サーバーを再起動して、同じベンチマーク（6562 token の文脈で 128 token 生成 / 74 token の短いプロンプト）を
投げ直した:

| 場面 | 再起直後 | キャッシュが温まった後 |
|---|---|---|
| プロンプト 6562 token を一から | 200.5 tok/s | **313.6 tok/s** |
| 出力（6562 token の文脈、プロンプトは一から） | 6.6 tok/s | **12.5 tok/s** |
| 出力（6557 token を再利用） | 7.4 tok/s | 10.1 / 10.4 tok/s |
| 出力（74 token の短いプロンプト、文脈は再利用） | 7.6 tok/s | 15.1 / 21.1 / 27.2 tok/s |
| expert cache ヒット率 | 67〜72% | 77〜80% |

IQ3_XXS で同じ測り方をしたとき（6.5 tok/s、ヒット率 57.6%）と比べると、長い文脈の出力は **約 2 倍**になった。
効いた順に:

1. IQ2_XS で expert の常駐が 2794 → 3815 増え、ヒット率が 10〜20 ポイント上がった（9.3）。
2. `--spec-min-p 0.70` でドラフトの窓を短く打ち切るようになり、外れが減った（`drafts accepted 53 of 74` など。
   窓 1 個あたりの採用 token 数が上がり、遅い CPU での verify 回数が減る）。
3. `--pcie-frac 0.35`（auto は 0.27）。

注意すべきは **直前に何を動かしていたかで 2 倍変わる**こと。サーバーを起動して最初のリクエストは
expert キャッシュが空なので常に遅い（6.6 tok/s）。同じ条件の再実行で 7.4 と 12.5 が出ている。
体感として「起動して最初の数リクエストは遅い」は仕様だと考えた方がいい。
`--prompt-cache`（既定 6 会話）と `--conversation-cache-mib`（既定 0）は、この「温まり方」を
リクエスト間に保つための設定。


### 9.6 50 tok/s を目指して Q2_0 に載せ替えた

「もっと速く（50 tok/s ）」という話で、まずこのマシンで 50 が出るのかを詰めた:

- 生成 1 token ごとに、MoE 48 層 × 10 = **480 個の expert** を評価する。
- 11 GB のカードで expert に割り当てられる VRAM は 5.1 GiB。IQ2_XS の blob は 1.41 MB なので
  **3815 個（全 24576 の 15.5%）**しか常駐できない。ヒット率 62〜80% は、つまり
  **1 token につき 96〜180 個を CPU + RAM 側で計算している**ということ。
- プロジェクト自身の推算（[docs/DETAILS.md](DETAILS.md)）では RTX 3090 24 GB で IQ2_XS 約 103〜131 tok/s、
  RTX 5060 Ti 16 GB で約 51〜77。「VRAM の追加は GPU 自体より効く。1 GB ごとに expert が約 700 個増え、
  GPU 上の expert 1 個は CPU が計算しなくていい 1 個」。
- **増設はできない**: multi-GPU（layer split）は compute capability 7.5 以上が必要で、GTX 10 系は setup に弾かれる
  （[docs/MULTI_GPU.md](MULTI_GPU.md)）。1080 Ti をもう 1 枚積んでも意味がない。

そこで無料の範囲で一番効くはずの **Q2_0**（i-quant ではない 2-bit。legacy quant なので CPU 側の復号が軽い）に入れた:

```
./setup.sh --setup --family qwen --model Q2_0 --context 32768 --kv int8 --vision cpu \
           --host 0.0.0.0 --api-key <鍵> --no-start --yes
```

- shard1 の 37.6 GB だけダウンロード（shard2 の 28.8 GB は IQ2_XS と同一ファイルなのでハードリンクで共有）。
  実測 約 13 分。`run-q2_0.sh` / `strata-q2_0.json` ができる。9.4 で決めた
  `--pcie-frac 0.35 --spec-min-p 0.70` は setup が知らない値なので手で足した。
- blob が 1.31 MB に減り、常駐 expert は **3815 → 3933**。ヒット率は 71% → 78〜83%。

同じベンチマーク、同じエンジン設定での比較:

| 場面 | IQ2_XS | Q2_0 |
|---|---|---|
| 出力（長い文脈、コールド） | 6.6 / 12.5 tok/s | **7.9 / 17.1** tok/s |
| 出力（同じ文脈を再利用） | 10.1 / 10.4 tok/s | **13.8 / 30.2** tok/s |
| 出力（短いプロンプト、文脈は再利用） | 15.1 / 21.1 / 27.2 | **16.5 / 29.4** |
| プロンプト 6562 token を一から | 200 / 313 tok/s | 205 / **320** tok/s |

**常駐 expert は 3% 増えただけなのに出力が 1.5〜3 倍**。差はほぼ全部「CPU 側で 1 行を復号するコスト」で、
i-quant（IQ2_XS）のコードブック復号がこの CPU（4 スレッド・AVX2 のみ、AVX-512 なし）で重かったということ。

品質は落ちる。このマシンで確認した範囲:

- `17 x 23 = ?` → 391（正しい）
- `fib(n)` をループで、というコード課題 → 正しい実装
- 日本語の要約 → 内容は正しいが「常驻」（簡体字混じりの表記）が出た。IQ3_XXS では出なかった

**50 tok/s には届かない**（現状 17〜30）。壁は VRAM のままなので、この PC でさらに狙うなら:

1. **自分のプロンプトで expert profile を作る**。エンジンを `--dump-routing trace.bin` で一度走らせ、
   `tools/make_profile.py` で ranking を作り、config の `--expert-profile` を差し替える。
   3933 枠に「自分の作業が実際に引く expert」を入れられ、CPU 側の miss そのものが減る。
2. `--conversation-cache-mib`（既定 0 = off）と `--prompt-cache-every`（既定 16384）で、
   長い文脈の読み直しを減らす（checkpoint 1 個あたり約 118 MB の RAM）。
3. GPU を RTX 20 系以降（16〜24 GB）に載せ替える。50 を超える唯一の経路。

なお `--experimental-speed-projection` は名前に反して**速度機能ではない**（拒否方向の制御ベクトルで、
むしろ 0.2〜0.4%/token 遅く、安全性の挙動を変える）。速度目的では入れないこと。

### 9.7 Q2_0 で `--calibrate` をやり直した

9.4 の `0.35 / 0.70` は IQ2_XS のときに測った値で、較正は PC + モデルごとに保存される（`hardware_key` は
GPU・CPU・RAM・モデル名・context・画像の有無）。Q2_0 では expert の blob が 1.41 → 1.31 MB に減るので、
PCIe で GPU に送る方が相対的に得になるはず、ということで Q2_0 向けに測り直した（**約 24 分**）:

```bash
python tools/calibrate.py strata-q2_0.json     # 測るだけ。専用エンジンを起動するのでサーバーは止めること
```

| 設定 | 試した値 → 出力 tok/s | 採用 |
|---|---|---|
| `--pcie-frac` | 0.00: 19.2 / 0.20: 24.0 / 0.27（auto）: 24.1 / 0.35: 12.9 / **0.55: 28.6** / 0.75: 23.3 | **0.55** |
| `--spec-min-p` | 0.30: 19.4 / 0.50（既定）: 21.8 / **0.70: 26.7** | **0.70** |
| `--pool-workers` | 3（既定）: 19.3 / **2: 20.5** | **2** |

- `--pcie-frac` の最適値が **0.35 → 0.55** に動いた。blob が小さいほど 1 個を PCIe で送るコストが下がるので、
  CPU で計算するより GPU に送った方が得になる範囲が広がったということ。IQ2_XS の値をそのまま使うのは間違いだった。
- `--pool-workers 2` が 6% 速い。4 スレッドのマシンで、メインスレッドと CPU expert プールを分けると
  3 スレッドでは競合しているということ。calibrate は既定より**少ない**方向しか測らないので、
  増やす方向（4）はそもそも試されていない。

サーバーを再起動して、IQ2_XS からの流用値（0.35 / 0.70）と同じベンチを投げ直した:

| 場面 | 0.35 / 0.70 | **0.55 / 0.70 / 2 workers** |
|---|---|---|
| 出力（長い文脈、コールド） | 7.9 tok/s | 7.7 |
| 出力（同じ文脈を再利用） | 13.8 | 13.1 |
| 出力（短いプロンプト、3 回） | 8.2 / 16.5 / 29.4 | 7.3 / 31.6 / 13.0 |
| 出力（2 ラウンド目の長い文脈、コールド → 再利用） | 17.1 → 30.2 | **20.3** → 29.1 |
| expert cache ヒット率 | 71〜83% | **78〜86%** |

**正直に書くと、このベンチでは較正の利きは確認できない**。同じ条件の 3 回で 7.3 / 31.6 / 13.0 と出るマシンなので、
数 % の改善はノイズに埋もれる。ヒット率は上がっている（`--pcie-frac` を上げると VRAM に送られる expert が増える）
ので較正値は採用した、という判断。`tools/calibrate.py` 自身も単発の測定が数 % 揺れることを知っていて、
採用するかどうかは既定値と候補を**交互に 3 回ずつ**測り直して決めている。

### 9.8 Q2_0 の設定スイープ（5.3 → 22.8 tok/s）

ブラウザで会話すると **5.3 tok/s**、同じバイナリをベンチで測ると 18.7 tok/s、という差が出た。
原因を潰すために Q2_0 のエンジン設定を総当たりした。全データは
[bench/results/2026-10-03-q2_0-config-sweep/](../bench/results/2026-10-03-q2_0-config-sweep/README.md)。

#### 症状（RAM の枯渇）

expert の arena が 31.64 GiB、PLE の n-gram 表（28.8 GiB）を mmap すると合計 60.4 / 62 GiB。
スワップは満杯（1.0 / 1.0 GiB）で、表のページは追い出され続ける。トークンごとに 16 行を読む必要が
あり、そのすべてが major fault + スワップの押し合いになる。**約 190 ms/token**、つまり 5.3 tok/s。
ベンチの 3 ラウンドは短いので表が RAM に残ったままなので、ベンチはこの問題も改善幅も**過小評価**していた。

#### 効いた手

| 設定 | 効果 |
|---|---|
| **`--ple-row-cache 16777216`** | **本命。** 読んだ n-gram 行をエンジン自身のキャッシュ（16M 行 × 90 B = 1.4 GiB）に保持。既定の 100 万行は 3.2 億行の 0.3% で、長い会話ではほぼ毎回 SSD 読み直しになっていた。コールド prompt で **18.7 → 28.0 tok/s（+50%）** |
| `--ple-io mmap` | 表を mmap 経由で読む。`direct` より速い（25.0 vs 25.2 = ノイズ内だが、RAM 圧があるときは mmap が無難） |
| `--ple-inflight 256` | 単体では +32%。ただし row cache を大きくすると redundant（C3 ≒ C2） |

#### 効かなかった手

| 設定 | 結果 |
|---|---|
| `--spec 6` | 28.5（`--spec 4` の 29.5 より遅い）。`--spec 4` で既に 56〜58/66 採用されており、窓を広くすると不採用ドラフトのコストを払うだけ |
| `--pcie-frac 0.75` | 22.3。9.7 で決めた 0.55 が正しい |
| `--kv q4_0` | 21.1。精度を落としつつ（[2026-09-27-kv-q4](../bench/results/2026-09-27-kv-q4/README.md)）、得られたスロット分の利得を失う |
| `--kv-resident 20480` | **15.4。** 20K 位置分の KV をトークンごとに PCIe で流すのは、32K の int8 KV を VRAM に置くより遥かに遅い。1080 Ti では使うな |
| `--expert-cache-per-layer` | エンジンが起動しない：`ExpertCache::verify_slot: slot 0 differs from the arena at byte 0`。MTP の draft head と併用不可 |
| `--pool-workers 3` | 25.8（ノイズ内）。4 スレッドで token スレッドからコアを奪う |

#### 実機確認（サーバーを止めずに）

`./run-q2_0.sh` で起動し、無関係な ~1,400 token の prompt を 4 本投げた:

| ラウンド | prompt tok/s | 出力 tok/s | expert ヒット |
|---|---:|---:|---:|
| 1（prefix 再利用） | 30.5 | 28.1 | 81.2% |
| 2 | 333.5 | 19.5 | 65.0% |
| 3 | 331.8 | 23.7 | 71.6% |
| 4 | 155.8 | 19.9 | 80.6% |

**平均 22.8 tok/s**（変更前 5.3）。コールドの 19〜24 はスイープの 28.0 より低いが、スイープの
prompt は合成的で繰り返しが多く、実文書は触る n-gram 行の種類が多い。

#### 50 tok/s が出ない理由（設定では埋まらない）

デコードは **expert cache の 13.4% ミス率**に律速される。ミス 1 回 = CPU の AVX-2 で 1 行 + PCIe 転送。
スロット数は 3,947（5.08 GiB）で、これは重み・draft head・681 MiB リザーブを引いた残り、VRAM 空きは
513 MiB しかない。ルーティング全集 24,576 個を収めるには約 31 GiB 必要で、VRAM の話であり設定の話ではない。
`docs/COMMUNITY_BENCHMARKS.md` の 50 tok/s は RTX 5090（32 GiB VRAM、PCIe Gen5、23 pool workers）の値。

#### 測り直し方

```bash
python tools/speed_bench.py strata-q2_0-fast.json   # コールドとウォームを 1 回のエンジン起動で測る
```

#### 採用設定（`strata-q2_0-fast.json`）

`--ple-row-cache 16777216`、`--ple-io mmap`、`--kv int8`、`--spec 4`、`--pcie-frac 0.55`、
`--spec-min-p 0.70`、`--pool-workers 2`、`--vram-reserve-mib 681`、`--expert-cache auto`。

**PLE 表は SSD 側に置く。** このマシンでは `/var/tmp` が Intel SSD（`rotational=0`）、`/home` が
回転 HDD（`rotational=1`）なので、既定の `Strata-data/models/.../00002-of-00002.gguf`（HDD 上）ではなく
`/var/tmp/strata-ple/` にコピーして参照している。トークンごとに 16 行を HDD から読むと致命的。

### 9.9 0.1.33 に追従し、画像入力を戻した（`--vision cpu` は無料だった）

engine を 0.1.31 → 0.1.33 に更新し（`a1eb951`、 portable image encoder #411 #412 など）、
上の採用設定のまま `--vision` を足して測り直した。実測は
[bench/results/2026-10-04-vision-catchup/README.md](../bench/results/2026-10-04-vision-catchup/README.md)。

**結論: `--vision cpu` は VRAM を使わないので、画像入力を戻しても速度はほぼ変わらない。**
エンジン側で増えるのは mrope の位置テーブル（context 32768 で `cells * 3` int32 ≒ **0.4 MiB**）だけ。
エンコーダは別プロセス（`engine/strata-vision`）で、`gpu: false` では VRAM を確保しない。

`tools/sweep_bench.py` 3 回 ×（128, 48）token、順序を入れ替えた 2 組で A/B:

| 構成 | expert cache slots | vram free | DECODE mean |
|---|---:|---:|---:|
| 画像なし（`strata-q2_0-fast.json`） | 3,947 | 513 MiB | 32.6 / 31.9 → **32.25** |
| `--vision`（`strata-q2_0-vision.json`） | 3,947 | 513 MiB | 31.3 / 30.9 → **31.1** |

スロット数も VRAM 空きも**完全に同一**。差は **-3.6%**（このマシンのノイズ幅 1% よりは大きいが、
5% には収まる）。原因として有力なのは、エンコーダの CPU スレッド 2 本がトークンスレッドと
pool workers 2 本と 4 コアで取り合うこと。気になるなら `vision.threads` を 1 にする（未測）。

9.6 節で「vision を消すと 3.1x 速い」と書いたのは **GPU エンコーダ**の話で、`--vision cpu` には当てはまらない。
`--vision gpu` は今も expert cache を削るので選ばない。

```bash
./run-q2_0-vision.sh      # 画像入力あり（推奨。上の -3.6%）
./run-q2_0.sh             # 画像入力なし（9.8 節の最速設定のまま）
```

## 10. 設定を変えたいとき

設定を変えたり別のモデルを入れるときは `--setup`（インストール済みでも再実行は安全、済んだステップはスキップ）:

```bash
./setup.sh --setup                                  # 質問に答えて設定変更・別モデル追加
./setup.sh --setup --model IQ2_XS --yes             # サイズだけ変える
./setup.sh --check                                  # PC のチェックだけ
./setup.sh --no-start --model IQ3_S --yes           # インストールだけして起動しない
```

主なオプション（`setup.py` のヘルプより）:

| オプション | 意味 |
|---|---|
| `--family qwen\|swift\|coder` | モデルの系統（既定 qwen） |
| `--model Q2_0\|IQ2_XS\|IQ3_XXS\|IQ3_S\|IQ1_M` | サイズ |
| `--context 32768` | 8192 / 32768 / 65536 / 131072 / 262144 / 393216 / 524288 |
| `--rope-scaling none\|linear\|yarn` `--rope-scale F` | 学習済み 262144 を超える context の延長 |
| `--vision yes\|no\|gpu\|cpu` | 画像入力（`yes` = `gpu`。このマシンでは VRAM を削るので `cpu`、8 節 / 9.9 節） |
| `--port 8080` | サーバーのポート |
| `--host 0.0.0.0 --api-key KEY` | LAN 越しアクセス（6 節） |
| `--data-dir DIR` | モデルファイルの置き場所（既定はリポジトリ隣の `Strata-data`、約 70〜120 GB） |
| `--models-dir DIR` / `--gguf-dir DIR` | GGUF の保存先 / 手元にある GGUF を使う |
| `--gpu 0` / `--gpus 0,1` | 使う GPU（複数カードの層分割は [MULTI_GPU.md](MULTI_GPU.md)） |
| `--build` | プリビルド版ではなく自分でコンパイル（SM60 では自動的にこれになる） |
| `--calibrate` | この PC に合わせてエンジン設定を調整（5〜10 分） |
| `--low-ram auto\|on\|off\|resident\|mmap` | expert を RAM にコピーせずファイルから読む（GPU 大きく RAM 小さい PC 向け） |
| `--draft-vocab cjk\|en` | ドラフト層の語彙（cjk が既定。en は VRAM を約 110 MiB 節約） |
| `--experimental-sm60` | GTX 10 / Volta をコミュニティビルドで動かす（#236、上流は未サポート） |
| `--experimental-speed-projection on\|off\|<GGUF>` | 実験的な速度投影（既定 off、挙動が変わる。先に DETAILS.md を読むこと） |
| `--yes` | 推奨値で質問をスキップ |

環境変数でも指定できる: `STRATA_EXPERIMENTAL_SM60=1`、`STRATA_API_KEY`、`STRATA_PREBUILT_URL`。

## 11. はまったポイントと対処

| 症状 | 原因 / 対処 |
|---|---|
| GPU が古いと弾かれて setup が止まる | 既定の下限は compute capability 7.5（RTX 20）。GTX 10 / Volta は `--experimental-sm60`（または `STRATA_EXPERIMENTAL_SM60=1`）で下限が 6.0 になる |
| SM60 のビルドが CUDA 13 で失敗する | CUDA 13 の nvcc は `compute_60/61/70` を削除済み。`setup.py` は SM60 のとき CUDA **12.6** を選ぶので、未インストールなら導入してから再実行（Windows は CUDA 12.6 を手元に入れてから再実行） |
| CMake の configure で失敗する | システムの `cmake 3.22` が `.venv` 内の `cmake 4.4.3` より優先されていた（**#236 で修正済み**：venv 内のツールを優先する）。修正前の版を使う場合は `PATH` を調整するか venv の cmake を先頭に |
| プリビルドエンジンがダウンロードされない／合わない | SM60 ではプリビルド版を使わず**必ずローカルビルド**（10〜20 分）。`--build` を付ける必要はない |
| 2 回目以降に `--experimental-sm60` を忘れた | `engine/BUILD.json` の `"sm60": true` を読んで判定するので、起動（`run-iq2_xs.sh` / `./setup.sh`）ではフラグ不要 |
| 起動中に 1〜3 分応答がなくなる | expert を RAM に読む処理。正常なので待つ |
| `WARNING: RAM is tight` | expert の arena が RAM を圧迫。小さいサイズ（`Q2_0` / `IQ2_XS`）を選ぶ、他プロセスを止める、`--low-ram` を検討する |
| ダウンロードが途中で切れた | そのまま再実行すれば続きから（HF のリビジョンは固定ピン済み） |
| LAN の他 PC から繋がらない | `host` が `0.0.0.0` になっているか、ファイアウォール、ルータのクライアント分離の順に確認 |
| `--setup` を実行したら LAN 公開が解除されていた | `--setup` は `strata-*.json` を書き直す。`--host` / `--api-key` を毎回一緒に渡す（渡さない場合は書き直し後に `host` / `api_key` を戻して再起動） |

## 12. 削除・入れ直し

```bash
kill <サーバーの PID>
rm -rf ../Strata-data          # モデル本体（IQ3_XXS + IQ2_XS で 118 GB）を消す
rm -rf engine build .venv      # ビルド成果物と Python 環境
rm -f strata-*.json run-*.sh strata-*.log
```

`git clean -xfd` でも消えるが、`Strata-data` はリポジトリの外にあるので個別に消す。

## 13. 参考

- [docs/DETAILS.md](DETAILS.md) — 正式の解説（速度、トラブルシューティング表、他デバイスからの利用）
- [docs/MULTI_GPU.md](MULTI_GPU.md) — 複数 GPU で層を分割する
- [docs/SECOND_GPU.md](SECOND_GPU.md) / [docs/AMD_HIP.md](AMD_HIP.md) — セカンド GPU / AMD
- [docs/ORCA.md](ORCA.md) — OrcaRouter 版 Flash-Next Uncensored の手動セットアップ
- [docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) — 自分の PC の測り方


