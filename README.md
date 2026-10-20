# Qwen3.8 Flash Next on one DGX Spark (hibrid48)

How this [Spark Arena result](https://spark-arena.com/benchmark/14ac40a1-3274-4286-be44-f0e4c452ef20?card=2)
was produced: **53.3 tok/s single-stream decode, with no drop out to 100k tokens of context**, on a single DGX Spark
(GB10, 128 GB).

Almost all of the engineering here is other people's work. See [Credits](#credits). This repo adds the
spark-vllm-docker recipe that reproduces the submission, three changes to the upstream config, and the steps that
made a ~4-hour benchmark run finish.

## What runs

| | |
|---|---|
| Model | [`myllmbox/Qwen3.8-Flash-Next-hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) @ `e316057d`, ~99 GB. NVFP4 routed experts, all 10 per token; bf16 attention and shared experts; 4-bit output head |
| Engine | myllmbox's vLLM 0.30 build, image `myllmbox/qwen38-flash-next-vllm@sha256:76eda2f6…` (v5.2) |
| Speculation | MTP, dynamic draft depth 3-7, sampled drafts with block verification |
| Recurrent layers | RecoverSSM (`--use-replayssm`): verify from one saved state, replay only the accepted tokens |
| MoE kernel | Marlin NVFP4 |
| n-gram table | NVFP4, demand-paged from NVMe instead of held in memory |
| KV | 20 GB pool (~650k tokens), 262,144-token window, 16 seats, prefix caching, async scheduling |

[`recipe.yaml`](recipe.yaml) has every flag and environment variable.

## Results

llama-benchy as run by `spark-arena-cli` v0.1.0, 2026-10-09, 936 of 936 requests, about 3 h 40 min.
Raw output is in [`results/`](results/): the `.csv` is the file submitted to the site, and the `.md` is the CLI's own table.

**Decode, tg128, total tok/s**

| context depth | c1 | c2 | c5 | c10 |
|---:|---:|---:|---:|---:|
| 0 | **53.3** | 71.6 | **115.6** | 105.3 |
| 4,096 | 50.5 | 66.8 | 47.4 | 43.7 |
| 8,192 | 49.8 | 72.9 | 47.6 | 41.8 |
| 16,384 | 55.2 | 71.5 | 51.2 | 47.4 |
| 32,768 | 52.7 | 73.4 | 50.9 | 46.9 |
| 65,535 | 52.2 | 66.5 | 49.1 | 5.0 |
| 100,000 | 55.1 | 76.9 | 43.1 | 3.2 |

**Prefill, pp2048, c1**: 1,970 tok/s from an empty context (TTFT 1.0 s), then 532-834 tok/s for 2,048 new tokens
on top of a 4k-100k context.

Reading the table:

- **c1 and c2 hold flat out to 100k.** MTP plus RecoverSSM keeps per-step cost nearly independent of depth on this
  hybrid-attention model.
- **c5 and c10 drop below c2 once the context is long.** Each 2,048-token prefill of a long request stalls decode for
  everyone else in the batch. At depth 0 the prefills are short, so c5 reaches 115.6.
- **c10 collapses at 65k and beyond.** Ten requests at 65-100k tokens need 650k-1M tokens of KV against a pool of
  649,534, so some of them queue. That is the cost of the 20 GB pool (see below). Peak decode at c10 is 239-245 tok/s
  while the pool still holds everything.

Single-stream tg128 varies ±6.7 tok/s between runs (MTP acceptance depends on content), so read c1 as 47-60.

## What was changed from upstream

The engine settings are myllmbox's [v5.2 `recipe.yaml`](https://github.com/myllmbox/qwen38-flash-next-recipe) (commit
`43b45e1`), unchanged except for these:

1. **The checkpoint: hibrid48, not v5.2's default.** v5.2 defaults to azampatti's INT4-AutoRound export, which routes
   5 of the model's 512 experts per token where the model was trained on 10. hibrid48 is the alternative listed in the same
   `recipe.yaml`: myllmbox's own export, which keeps all 10. myllmbox's notes put it about 20% slower. On a 75-exercise
   Aider polyglot subset it scored 62, against 59 for INT4-AutoRound. At that sample size the difference is
   suggestive, not significant.
2. **`--kv-cache-memory 20000000000` (upstream: 27 GB).** At 27 GB the box has about 1 GiB of host memory left
   (GB10 memory is shared between CPU and GPU). 20 GB measured the same decode speed at 1-8 streams and leaves 9-10 GiB.
   The cost is the c10 behaviour past 65k described above.
3. **`reasoning_effort: medium` as the server default.** hibrid48 ships Qwen3.8's stock chat template, which defaults
   to `xhigh`. With greedy decoding, `xhigh` loops inside the think block until it hits `max_tokens`. Clients that send
   their own effort still get it.
4. **`--enable-prompt-tokens-details`**, so `usage` reports cached prompt tokens. This is cosmetic for the benchmark.
5. **`--distributed-executor-backend mp` was left out**, by oversight while converting the recipe. It is the only
   upstream flag missing from the submission. At TP 1 vLLM picks its own executor; the effect is unmeasured.

`spark-arena-cli` forces a few more: the served name is the HF id, the port is 8000, and caches live under
`/root/.cache/vllm/mbx` (the runner mounts `~/.cache/vllm`). The runner also starts the container `--privileged`,
which covers the `seccomp=unconfined` that myllmbox's `run.sh` passes, and the recipe clears the runner's
`PYTORCH_CUDA_ALLOC_CONF` (see [Traps](#traps-that-cost-a-failed-run)). Everything else matches upstream: the
image (by digest rather than the `v5.2` tag), every other vLLM flag, the environment, the draft-depth settings
(dynamic, minimum 3, window 48), and the CPU pinning.

## Reproducing it

You need a DGX Spark with Docker and the NVIDIA container runtime, about 155 GB of free disk (99 GB of checkpoint,
a ~31 GB table map and the 22 GB image), and nothing else loaded on the GPU.

**You do not patch or build vLLM.** All of myllmbox's changes are pre-built into their v5.2 image, which the recipe pins
by digest and the runner pulls as-is (`mods: []`). The only host setting the submission box had beyond stock DGX OS:

```bash
sudo sysctl -w vm.compaction_proactiveness=0     # persist it in /etc/sysctl.d/ if you keep it
```

myllmbox recommends it (their `tune-host.sh`): on a Spark, the GPU's memory is ordinary system pages, and the kernel's
background compactor unmaps them from the GPU to move them. They measured 4-5 s stalls about every 37 s. We could not
measure a difference above run-to-run noise, so treat it as cheap insurance, not a speedup.

**1. Install the CLI** (it clones spark-vllm-docker into `~/.spark-arena/` during `setup`):

```bash
gh release download v0.1.0 -R spark-arena/spark-arena-cli -p '*arm64.deb'
sudo dpkg -i spark-arena-cli_0.1.0_arm64.deb
spark-arena-cli          # then: login, setup
cp recipe.yaml ~/.spark-arena/spark-vllm-docker/recipes/qwen38-flash-next-myllmbox-hibrid48.yaml
```

**2. Fetch the checkpoint at the pinned revision.** The recipe runs offline (`HF_HUB_OFFLINE=1`), and vLLM cannot take
`--revision` offline, so `refs/main` has to point at this commit:

```bash
hf download myllmbox/Qwen3.8-Flash-Next-hibrid48 --revision e316057ddc794cbf76c3ef28bfd1c7b62b20128c
D=~/.cache/huggingface/hub/models--myllmbox--Qwen3.8-Flash-Next-hibrid48
printf %s e316057ddc794cbf76c3ef28bfd1c7b62b20128c > $D/refs/main
```

An unthrottled download saturates a shared uplink (~27 MB/s measured). Throttle it if anyone else uses the link.

**3. Boot it once outside the CLI.** The CLI waits only about 5 minutes for the server. A warm boot takes 196 s; the
first one builds the compile cache and the n-gram table map and does not finish in time.

```bash
cd ~/.spark-arena/spark-vllm-docker
./run-recipe.sh qwen38-flash-next-myllmbox-hibrid48 --solo
# wait for /v1/models on :8000, then stop it
```

**4. Run the benchmark** in tmux, from a directory you can find again (llama-benchy writes `tests.csv` there):

```bash
tmux new -s arena
VLLM_SPARK_EXTRA_DOCKER_ARGS="--cpuset-cpus 5-9,15-19" spark-arena-cli
#   spark-arena> benchmark qwen38-flash-next-myllmbox-hibrid48
```

The cpuset pins the container to the GB10's ten performance cores, which is how the submission ran. It is optional.

Do not stop the run early, because results are written only at the end. Requests do not show in `docker logs`; track
progress with vLLM's counter, which reaches 936:

```bash
curl -s localhost:8000/metrics | awk '/^vllm:request_success_total/{split($0,x," "); s+=x[2]} END{print s}'
```

**5. Publish.** The CLI's "Upload successful" does **not** publish anything. Submit through the site's web form with
the recipe and llama-benchy's CSV. The CLI writes `tests.csv` as a markdown table, so convert it to CSV first;
[`results/llama-benchy-20261009.csv`](results/llama-benchy-20261009.csv) shows the expected format. The form values
used were NVFP4, 4 bits, 105.3 GB, 125B parameters (NVIDIA's model-card figure), MoE, 1 node, TP 1.

### Traps that cost a failed run

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**, which spark-vllm-docker's launcher exports, took
  MemAvailable to zero during weight load. The kernel then killed the tmux server and the CLI with it. The recipe sets
  the variable to empty. Keep that line.
- **The CLI holds its login in memory only.** If it dies, you log in again. Type the challenge code the browser page
  shows, not the one in the URL. Over SSH the CLI cannot open a browser and does not print the URL. Put an `xdg-open`
  on `PATH` that writes its argument to a file, then open that URL through `ssh -L 9005:localhost:9005 <spark>`.
- **The container has no out-of-memory guard.** For an unattended run, consider MiaAI-Lab's
  [`memwatch.sh`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) (AGPL-3.0), which stops the
  container on sustained low MemAvailable before the host locks up. With the settings above, MemAvailable stayed at
  10 GiB or more for the whole run.

## Credits

| Who | What it contributed | License |
|---|---|---|
| [Qwen team, Alibaba](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Qwen3.8-Flash-Next, the model | Qwen Community License 1.0 |
| [myllmbox](https://github.com/myllmbox/qwen38-flash-next-recipe) | The recipe, the v5.2 vLLM image, the hibrid48 export, the NVFP4 table reader, dynamic draft depth, the fused multi-step MTP drafts, and the RecoverSSM port to 0.30. This is the engine | Kit and patches MIT; table and depth libraries binary-only; weights Qwen Community License 1.0 |
| [Inferact](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4) | The NVFP4 routed-expert tensors that hibrid48 is built from | see model card |
| [@jschmied](https://github.com/jschmied) | RecoverSSM, [vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863) | Apache-2.0 (vLLM) |
| [vLLM](https://github.com/vllm-project/vllm) | The serving engine underneath | Apache-2.0 |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | The recipe runner Spark Arena uses | MIT |
| [eugr/llama-benchy](https://github.com/eugr/llama-benchy) | The benchmark | MIT |
| [Spark Arena](https://spark-arena.com) / [spark-arena-cli](https://github.com/spark-arena/spark-arena-cli) | The leaderboard and the CLI that ran the matrix | GPL-2.0 |
| [MiaAI-Lab](https://github.com/MiaAI-Lab) | `memwatch.sh`, and the single-Spark Flash-Next recipes ([vLLM](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark), [TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)) that this setup was measured against before switching. Their README is the model for this one | AGPL-3.0 |

This repo contains only a recipe, documentation and benchmark output. It distributes no weights, images or code from
the projects above. Using them means accepting their licenses, in particular the Qwen Community License for the weights.

## License

MIT for the contents of this repo; see [LICENSE](LICENSE).
