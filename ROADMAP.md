# Roadmap

## 0.1 (now)
- `localllm`: one command - detect GPU/RAM, pick the most accurate measured model for your language, download, start a tuned llama-server, open the chat page
- `localllm doctor`: which model sizes this GPU suits, speed estimates, how much text the recommended model holds
- `localllm eval`: global (Global-MMLU-Lite, 23 languages) + regional (INCLUDE, 44 countries; ThaiExam) multiple-choice scores
- Tuned launch: small-BAR fix (`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`), MTP drafting for Qwen3.8, single-slot unified KV, `-fit off`

## 0.2 - use less system RAM
llama-server's defaults can add 10+ GB of system RAM on top of a model that sits entirely on the GPU. On a 16 GB PC that
squeezes everything else.

- [ ] Measure first: RAM over a long multi-turn chat, default vs tuned (prompt cache, context checkpoints, mmap vs `--load-mode none`)
- [ ] `--cache-ram` (prompt cache, default 8 GiB) and `--ctx-checkpoints` (default 32 per slot; each holds a DeltaNet/SWA state of ~110-150 MB) sized to the PC's RAM, e.g. 16 GB -> 1 GiB / 4, 32 GB -> 2 GiB / 8
- [ ] Check the speed cost of smaller caches (more prompt re-processing in long chats)
- [ ] `localllm doctor`: show how much RAM the chosen model will use and how much stays free for other apps
- [ ] MoE with experts in RAM: pick `--n-cpu-moe` from free RAM instead of failing or swapping

## 0.3 - local AI that works with the cloud providers' APIs
Most apps and SDKs speak one provider's API. Make the local model a drop-in for them, and let users mix local and cloud.

- [ ] One local endpoint that speaks the OpenAI, Anthropic Messages, Gemini and Ollama APIs, so existing apps, SDKs and agents point at `localhost` with no code change
- [ ] Hybrid routing: answer locally by default, send a request to the user's own cloud key (OpenAI / Anthropic / Google / OpenRouter) when it's too long, needs a tool or model the PC can't run, or the local model is busy; per-app rules, cost and privacy shown up front
- [ ] Keys stay on the machine (OS keychain); nothing leaves the PC unless a rule sends it
- [ ] `localllm route --test`: same prompt to local and cloud, compare answer, latency and cost

## 0.4 - squeeze the GPU
A release whose changelog is all speed, so people see the upgrade on the same hardware.

- [ ] Auto-tune per GPU: short sweep of batch/ubatch, flash attention, KV cache type, MTP draft length, thread count; keep the fastest and remember it
- [ ] Fix the remaining per-token overhead found on RDNA4: recurrent-state copies in DeltaNet models (CPY/GET_ROWS of 3 MB states), upstream the fixes to llama.cpp
- [ ] Faster model load: larger upload staging buffers (16 MiB measured 1.2 s faster on a 12 GB model), skip the fit dry-run when the plan is known
- [ ] Speculative decoding wherever it pays: MTP heads, matching small draft models, prompt lookup for code/edit tasks
- [ ] Before/after table per GPU in the release notes, measured with `localllm bench`

## Later
- Language-calibrated quantizations (imatrix from each language's text), published per language with before/after scores
- More measured GPUs: `localllm eval` results from contributors feed the catalog
- Writing-quality evaluation (not just multiple choice)
