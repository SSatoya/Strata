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
| ディスク | /home に 1.8 TB（モデル一式で約 79 GB） |
| Strata engine | 0.1.31、ローカルビルド（`archs: [61]`、`sm60: true`） |
| モデル | Qwen3.8-Flash-Next / IQ3_XXS、context 32768、画像なし |

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
│   ├── engine/BUILD.json       # どの構成でビルドしたかの記録
│   ├── build/CMakeCache.txt    # STRATA_EXPERIMENTAL_SM60:BOOL=ON が入る
│   ├── strata-iq3_xxs.json     # 起動設定（gitignore 済み）
│   ├── run-iq3_xxs.sh          # 起動スクリプト（gitignore 済み）
│   └── strata-iq3_xxs.log      # エンジンのログ（毎リクエストの tok/s、cache hit rate が出る）
└── Strata-data/                # モデル本体（約 79 GB）
    ├── models/                 # 71 GB  GGUF の 2 分割シャード（44 GB + 27 GB、expert はここから読む）
    ├── packs/iq3_xxs/          # 1.5 GB pack（dense 重み・expert の索引・トークナイザ）
    └── mtp/                    # 6.5 GB MTP ドラフト層
```

`engine/BUILD.json`（SM60 の記録がここに残る）:

```json
{
  "source": "local",
  "version": "0.1.31",
  "archs": [61],
  "vision": "none",
  "cuda_dirs": ["/usr/local/cuda-12.6/bin", "/usr/local/cuda-12.6/lib64"],
  "src": "260e57c13c78d103",
  "vision_src": null,
  "sm60": true
}
```

**`"sm60": true` が重要**：これがあるおかげで、2 回目以降は `--experimental-sm60` を付けなくても
（`./setup.sh` だけでも、`run-iq3_xxs.sh` だけでも）同じビルドとして認識され、GPU チェックを通過する。

## 5. 起動・停止

```bash
./run-iq3_xxs.sh          # 起動（約 47 GB の expert を RAM に読む）
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
  nohup ./run-iq3_xxs.sh > /tmp/strata-server.log 2>&1 &
  pgrep -af 'serve/server.py|engine/strata'      # 確認
  kill <PID>                                     # 停止（エンジン子プロセスも一緒に終わる）
  ```

  停止後は RAM が数秒で戻る（`free -g` で確認）。

## 6. LAN 越しにアクセスする（他の PC から使う）

コードの変更は不要。設定ファイル `strata-iq3_xxs.json` に 2 つ足すだけ
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
- 鍵は `strata-iq3_xxs.json` に平文で保存される（このファイルは gitignore 済み）。
- ローカル専用に戻すには `"host": "127.0.0.1"` に変えて再起動。
- 繋がらないときはファイアウォール（`sudo ufw status` / `sudo nft list ruleset`）とルータのクライアント分離を確認。
  `0.0.0.0` にすると Docker の `docker0`（172.17.0.1）にも listen する。
- `--api-key` に空文字を渡すと起動時にエラーになる（空の鍵は拒否される仕様）。

## 7. 実測性能（GTX 1080 Ti + IQ3_XXS + 62 GB RAM）

ベンチマークではなく、このマシンで体感した範囲のメモ。

| 場面 | 実測 |
|---|---|
| プロンプト処理（652 token、暖まった状態） | 約 125 tok/s |
| 出力（長い回答） | 約 5.7〜6.8 tok/s |
| 出力（短い回答） | 10 tok/s 前後 |
| 起動後 1 回目リクエスト（コールド） | プロンプト 7〜21 tok/s |

- 出力速度を律速しているのは GPU 単体の計算ではなく、**RAM／CPU からの expert ストリーミング**。
  Pascal は Turing 以降のカーネルパスを使えないため、RTX 20 系以降より遅くなる前提。
- 数値は他のプロセス・コンテキスト長・expert のヒット率で大きく動く。自分の PC を測る場合は
  [docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) の手順で。
- 上の数値はエンジンログ `strata-iq3_xxs.log` から取ったもの。毎リクエストの速度と、
  MTP ドラフトの採用率・expert キャッシュのヒット率が出る:

  ```
  strata serve: prompt 59 tokens = 0 reused + 59 read in 2518 ms (23.4 tok/s), 64 generated in 11069 ms (5.8 tok/s), drafts accepted 29 of 54, 1 checkpoints
  strata serve: decode expert cache hit rate: 64.2% (25274 hits / 39379 lookups)
  ```

- 起動ログに `this CPU has no AVX-512: the expert kernels run on AVX-2` と出る。このマシンは AVX-512 非対応で、
  expert の gate/up 行のカーネルが AVX-2 で動く（出力速度に影響する）。
- 同じく起動ログに `expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable ... using 4 KB pages`。
  expert を置く RAM は pinned memory として登録されるが、この環境では hugetlb プールが無いため 4 KB ページで動いている。
  PCIe は `12.9 GB/s host->device` と実測されている（expert を GPU に流す帯域）。
- `./setup.sh --calibrate` でこの PC に合わせたエンジン設定のチューニング（5〜10 分）ができる。

## 8. 設定を変えたいとき

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
| `--vision yes\|no\|gpu\|cpu` | 画像入力（今回の SM60 構成では未検証） |
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

## 9. はまったポイントと対処

| 症状 | 原因 / 対処 |
|---|---|
| GPU が古いと弾かれて setup が止まる | 既定の下限は compute capability 7.5（RTX 20）。GTX 10 / Volta は `--experimental-sm60`（または `STRATA_EXPERIMENTAL_SM60=1`）で下限が 6.0 になる |
| SM60 のビルドが CUDA 13 で失敗する | CUDA 13 の nvcc は `compute_60/61/70` を削除済み。`setup.py` は SM60 のとき CUDA **12.6** を選ぶので、未インストールなら導入してから再実行（Windows は CUDA 12.6 を手元に入れてから再実行） |
| CMake の configure で失敗する | システムの `cmake 3.22` が `.venv` 内の `cmake 4.4.3` より優先されていた（**#236 で修正済み**：venv 内のツールを優先する）。修正前の版を使う場合は `PATH` を調整するか venv の cmake を先頭に |
| プリビルドエンジンがダウンロードされない／合わない | SM60 ではプリビルド版を使わず**必ずローカルビルド**（10〜20 分）。`--build` を付ける必要はない |
| 2 回目以降に `--experimental-sm60` を忘れた | `engine/BUILD.json` の `"sm60": true` を読んで判定するので、起動（`run-iq3_xxs.sh` / `./setup.sh`）ではフラグ不要 |
| 起動中に 1〜3 分応答がなくなる | expert を RAM に読む処理。正常なので待つ |
| `WARNING: RAM is tight` | expert の arena が RAM を圧迫。小さいサイズ（`Q2_0` / `IQ2_XS`）を選ぶ、他プロセスを止める、`--low-ram` を検討する |
| ダウンロードが途中で切れた | そのまま再実行すれば続きから（HF のリビジョンは固定ピン済み） |
| LAN の他 PC から繋がらない | `host` が `0.0.0.0` になっているか、ファイアウォール、ルータのクライアント分離の順に確認 |

## 10. 削除・入れ直し

```bash
kill <サーバーの PID>
rm -rf ../Strata-data          # モデル本体（約 79 GB）を消す
rm -rf engine build .venv      # ビルド成果物と Python 環境
rm -f strata-*.json run-*.sh strata-*.log
```

`git clean -xfd` でも消えるが、`Strata-data` はリポジトリの外にあるので個別に消す。

## 11. 参考

- [docs/DETAILS.md](DETAILS.md) — 正式の解説（速度、トラブルシューティング表、他デバイスからの利用）
- [docs/MULTI_GPU.md](MULTI_GPU.md) — 複数 GPU で層を分割する
- [docs/SECOND_GPU.md](SECOND_GPU.md) / [docs/AMD_HIP.md](AMD_HIP.md) — セカンド GPU / AMD
- [docs/ORCA.md](ORCA.md) — OrcaRouter 版 Flash-Next Uncensored の手動セットアップ
- [docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) — 自分の PC の測り方


