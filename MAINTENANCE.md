# Maintenance notes

Known risks and what to check after updating ComfyUI, comfy-kitchen or torch.

Tested with ComfyUI 0.43.0, comfy-kitchen 0.2.40, torch 2.14.1+cu130 (2026-10-29; Spectrum and SPEED 2026-11-01).
Spectrum on MOSS-Image and on MiniMax H3: ComfyUI 0.45.0, comfy-kitchen 0.2.41 (2026-11-18).
Guider Track Model: ComfyUI 0.45.0, comfy-aimdo 0.6.0 (2026-11-18).
SnapRef: ComfyUI 0.45.0, comfy-kitchen 0.2.41, torch 2.14.1+cu130, SnapRef node 87bdacd, weights v2a (2026-11-20).
Recheck after updating any of these. The quickest check for each node is listed with it.

## Block Compile — the stopgap one, steps off ComfyUI's direction

- Turns off dynamic VRAM, ComfyUI's current memory system. The core TorchCompileModel node does
  the same today, but ComfyUI's own speed path under dynamic VRAM is the block CUDA graphs in
  `comfy/model_prefetch.py`, so this node lives outside that.
- The CK fix bypasses the `AttentionTensorContainer` path ComfyUI is moving to. That path frees
  q/k/v right after quantizing them; the plain path used here keeps them a bit longer
  (same output, a little more temporary memory).
- Relies on internals: `comfy.ldm.modules.attention.attention_comfy_kitchen_int8`, its
  `__wrapped__` and `container_function`, and the module-level `optimized_attention`.
- Retire or patch when:
  - the original AnimaBlockCompile works with `--use-ck-attention` (container path became
    compile-safe) → delete the CK routing block;
  - ComfyUI offers compile with dynamic VRAM on → follow that instead of `disable_dynamic=True`.
- Check: CK + this node must compile without the "Cannot access data pointer of Tensor" error,
  and its uncompiled output must stay bit-identical to plain CK.

## 2.9B LoRA Remap — fits the standard LoRA flow, but can fail silently

- Relies on internals: `ModelPatcher.patches` entries being
  `(strength_patch, adapter, strength_model, offset, function)`, adapters of one LoRA file sharing
  one `loaded_keys` set (`comfy/lora.py`), keys named `diffusion_model.blocks.N....`, and
  `unet_config` `image_model` / `num_blocks`.
- If the `loaded_keys` sharing changes, LoRAs can't be told apart and a native 2.9B LoRA could
  be moved by mistake without any error.
- The block table comes from the 2.9B model layout, not from ComfyUI; it only changes if a new
  2.9B base with a different insertion pattern appears.
- Check: on a 2.9B model, 2B LoRA + this node must be bit-identical to the same LoRA converted
  to `*_29b`, and a `*_29b` LoRA + this node must be unchanged.

## INT8 Quantize — standard format, small own code

- Output is the standard `int8_tensorwise` (+ `convrot`) format loaded by ComfyUI's normal
  loader. If ComfyUI adds its own load-time int8 quantization, this node can simply be retired.
- Uses private comfy-kitchen helpers `_build_hadamard` / `_rotate_weight`
  (`comfy_kitchen.tensor.int8_utils`; they lived in `comfy_kitchen.tensor.int8` before). If they
  move, the import fails. The rotation must match what ComfyUI applies at runtime, so keep using
  comfy-kitchen's helpers rather than a local Hadamard.
- Quantizes in fp32 on purpose: comfy-kitchen's public `TensorWiseINT8Layout.quantize` gave
  slightly higher weight error in testing and isn't bit-identical to comfy-model-tools files.
- Relies on internals: `model.cached_patcher_init == (comfy.sd.load_diffusion_model,
  (unet_path, model_options))` from Load Diffusion Model, and ComfyUI calling the factory again
  with `disable_dynamic=True` (block compile does this).
- Layer selection mirrors comfy-model-tools defaults for Anima (2-D block linears, min side
  ≥ 256, llm_adapter excluded). If that tool changes its defaults, fresh files will differ from
  this node; that's expected, not a bug.
- Check: node output must be bit-identical to a file made by comfy-model-tools
  (`quant_int8_convrot.py`, default options) for the same model.

## Spectrum — re-implements the end of the Anima (and MOSS-Image) forward; swaps `_run_blocks` on MiniMax H3

Tested 2026-11-01 against ComfyUI-Spectrum-KSampler 2.13.0.

- On predicted steps the node runs its own copy of the tail of `MiniTrainDIT._forward`
  (`comfy/ldm/cosmos/predict2.py`): `t_embedder[0]` → cast → `t_embedder[1]` → `t_embedding_norm`
  → `final_layer(x, emb, adaln_lora_B_T_3D=...)` → `unpatchify` → crop. If ComfyUI changes that
  tail (new modulation, another final layer argument), predicted steps go wrong silently.
- The feature is caught with a pre-hook on `final_layer`, inside a `DIFFUSION_MODEL` wrapper. Works
  with per-block compile; the core TorchCompileModel (whole model compiled) traces the wrapper and is
  not supported. If Anima moves to the memory compiler (`comfy/model_prefetch.py`), the copy is
  already made under `pause_malloc_graph`.
- Relies on `transformer_options` keys `sample_sigmas`, `sigmas`, `cond_or_uncond`, `uuids`, and on
  the flow model getting sigma 1.0 as 0.99997 (matched with a 1e-4 tolerance). If steps stop being
  found, every call runs the full model: the log line then shows few or no predicted steps.
- ControlNet-style `control` input → full model calls only.
- Samplers (2026-11-01, 3 scenes each, 100% crops): euler, er_sde, euler_ancestral, dpmpp_2m, uni_pc,
  heun, dpmpp_sde clean. dpmpp_2m_sde(_gpu) gets iridescent blotches and dpmpp_3m_sde breaks up; the
  original DiT Spectrum Patch does the same, so it is the method, not this code: a predicted step
  can't react to the noise injected the step before, and the multistep correction amplifies that.
  The node checks `sampler_function.__name__` against `UNSUPPORTED_SAMPLERS` and stays off for them;
  a renamed or new SDE multistep sampler would slip through. Multi-call samplers: off-grid
  (midpoint) calls always run the full model, so dpmpp_sde gains little (18 vs 20 s); heun predicts
  26 of 59 calls.
- Extra VRAM (memtrace peak, 2B, batch 1): +512 MB at 1024², +1152 MB at 1536² with
  `history_size` 100; +256 / +576 MB with 8, which looked the same at 100% (25-28 dB vs 100).
- Also checked: 2.9B with a 2B LoRA + Remap + compile (LoRA applied), batch 2, CFG 1, 20 and 50
  steps (`20 steps, 14 full, 6 predicted`; `50 steps, 22 full, 28 predicted`), hires 2-pass on the
  same patched model (each pass gets its own forecast), five 1536-area buckets.
- Check: with defaults and 30 steps the log says `30 steps, 17 full model calls, 13 predicted`. Next
  to the original DiT Spectrum Patch (same settings, er_sde) the images look the same and both sit
  in the same PSNR range against no Spectrum (19-26 dB, 3 prompts x 2 seeds); kludge vs original is
  25-33 dB. Not bit-identical: the original rounds the forecast to bf16.

### Spectrum on MOSS-Image

The node picks the architecture from the model's attributes (`ARCHS` in `spectrum.py`); forecaster,
window rule and sampler wrapper are shared with Anima.

- On predicted steps it runs its own copy of the tail of `MOSSImage._forward`
  (`comfy/ldm/moss_image/model.py`): `timestep_embedding(t, 256)` → cast → `time_embedding` →
  `video_head_modulation` split into (shift, scale) → `modulated_rms_norm(x, final_norm_video,
  scale, shift)` → `final_linear_video` → transpose/reshape to `latent_channels`. If ComfyUI changes
  that tail, predicted steps go wrong silently. `modulated_rms_norm` and `timestep_embedding` are
  imported from ComfyUI inside the head, so the pack still loads on a ComfyUI without MOSS-Image.
- There is no module call between the blocks and the head, so the feature is copied from the output
  of the last block (`layers[-1]` forward hook), image tokens only (`x[:, :H*W]`; text and reference
  tokens follow). Relies on the image tokens coming first in the sequence and on the blocks returning
  the sequence. When the reference keys are not precomputed, `preprocess_refs` runs the blocks first;
  the last capture is the main pass.
- `MOSSImage.forward` hands `transformer_options` to the wrappers positionally (Anima: keyword). If
  that changes the node finds no state and every call runs the full model (log: 0 predicted).
- Forecasters are per cond, like Anima. MOSS-Image runs cond and uncond as separate calls (different
  text lengths), so the log counts calls, not steps: `25 steps, 30 full model calls, 20 predicted`.
  A cond that changes mid-run starts a new forecaster and costs a few full calls: SLG ending at 50%
  gives `33 full, 17 predicted`, the depth control ending at 60% `36 full, 14 predicted`.
- Text: at defaults one small text block came out garbled on 1 of 5 seeds of the layout template
  (with and without SLG; t2i posters, Base + SLG, edit and depth control were clean). The block is
  still forming during steps 6-10; `warmup_steps` 10 kept it intact (`flex_window` 0 and
  `tail_actual_steps` 6 did not).
- Measured (SFT int8, 1024², CK, no patch → Spectrum defaults): t2i 25 steps er_sde 10.40 → 6.5 s,
  Base + SLG 10.5 → 7.2 s, layout + SLG 11.4 → 7.8 s (8.9 s with `warmup_steps` 10), edit LoRA
  30 steps euler 16.1 → 9.7 s, depth control LoRA 10.4 → 7.7 s. Extra VRAM not measured; one
  feature is about 25 MB at 1024² (4096 tokens × 3072, bf16), 30 of them at defaults.
- Check: SFT t2i, 25 steps, er_sde, defaults → log `25 steps, 30 full model calls, 20 predicted`,
  and the image keeps the composition and text of the unpatched run (20-25 dB, 4 prompts x 2 seeds).
  Anima output must not change when this part is edited (bit-identical before/after).
- SPEED and Block Compile are not for MOSS-Image. SPEED runs but garbles text at the default
  handoff (18 of 25 steps at low resolution with the shift 5 schedule), and the settings that keep
  text are only 9-12% faster. Per-block compile gained 2% (the blocks are already fused
  comfy-kitchen kernels) and needs a comfy-kitchen patch to trace.

### Spectrum on MiniMax H3

Same forecaster, window rule and sampler wrapper; `_H3` in `spectrum.py` is the adapter.

- No copy of the model tail here. On a predicted step the node sets an instance attribute
  `dit._run_blocks` (shadowing the method) to a function that writes the forecast into the last rows
  of the packed sequence and returns it, calls the model's own `_forward`, and deletes the attribute
  again. Embed, timestep table, `final_layer` (PDD heads included) and unpatchify stay ComfyUI's
  code. Relies on `MiniMaxH3Model._forward` calling `self._run_blocks(h, t_emb, mod_segments, layout,
  transformer_options)` and using its return value as the input of `final_layer`. If the blocks loop
  is inlined or renamed, the attribute is never called: every step then runs the full model while
  the log still counts predicted steps, and the time does not drop.
- The feature is caught with a pre-hook on `final_layer(x, t_emb, video_seg, audio_seg, ...)`:
  rows from the start of the first target segment to the end. Relies on the target streams being the
  last segments of the pack ([text | references | audio | video]) and on those argument positions.
  `final_layer` only reads the target rows, so the other rows of the returned sequence don't matter.
- `x` is a list `[video, audio]` and `transformer_options` comes positionally (like MOSS-Image).
- One model call per step (BasicGuider, no CFG): `20 steps, 14 full model calls, 6 predicted`.
  Turbo models at 8 steps or fewer predict nothing (`warmup_steps` 6 + `tail_actual_steps` 3).
- Feature size is rows x 5376 in bf16: 76 MB for 864x480 2 s, about 165 MB for 5 s, about 400 MB for
  1344x768 5 s; up to 14 are kept at 20 steps. Measured peak: +1.0 GB (2 s), +2.2 GB (5 s).
  The model is larger than a 16 GB card and streamed by dynamic VRAM, so that memory is taken from
  the weights; lower `history_size` if steps slow down.
- Measured (fl2va int8, t2v, 864x480, res_multistep, simple, 20 steps): 2 s 28.7 → 18.9 s, 5 s
  69.5 → 50.1 s. Video keeps composition and motion (23-27 dB), audio waveform correlation with the
  unpatched run 0.98.
- Not run: reference (ref2va) and keyframe (i2v / first-last frame) conditioning, AddGuide, the Fun
  ControlNet patch (its block hooks are skipped on predicted steps), masks, the H3 image model,
  samplers other than res_multistep.
- Check: t2v template settings with defaults → log `20 steps, 14 full model calls, 6 predicted`,
  the sampling time drops by about a third, and the video keeps the composition of the unpatched
  run. Anima and MOSS-Image output must not change when this part is edited.
- Block Compile and SPEED are not for H3. Per-block compile runs but is 35% slower (8 GB of the
  20 GB model is offloaded and the compiled blocks wait on the transfers; the blocks are fused
  comfy-kitchen kernels already). SPEED needs a version that shrinks only the video stream of the
  packed latent; a rehearsal worked but changes composition and makes the audio louder.

## SPEED — swaps the sampler function for one call

- Wraps `SAMPLER_SAMPLE`; when the sampler is a `KSAMPLER` it temporarily replaces
  `sampler_function` with a two-stage function (low-res sigmas, expand, re-spaced full-res sigmas)
  and restores it afterwards. Relies on the k-diffusion call signature
  `(model, x, sigmas, extra_args, callback, disable, **options)`, on `extra_args["seed"]` and
  `extra_args["model_options"]["transformer_options"]["sample_sigmas"]`, and on the callback dict's
  `"i"`.
- Turns itself off (with a log line) for inpainting masks, non-k-diffusion samplers and partial
  denoise (first sigma below `model_sampling.sigma_max`). Without that check a hires pass at denoise
  0.5 (starts at sigma 0.75) shrank and re-expanded the upscaled image and came out blurred.
- The two-stage function keeps the sampler's `__name__` (functools.wraps) so Spectrum's sampler
  check still works when SPEED is outside it.
- Conditioning that depends on the latent size (area conds, concat latents) is not handled at the
  low-res stage.
- Block Compile sees two latent sizes, so torch.compile switches to dynamic shapes on the first run
  (longer first compile).
- Measured with CK + int8 + Block Compile, er_sde 30 steps (no patch → Spectrum / SPEED /
  SPEED + Spectrum): 2B 1024² 6.58 → 3.98 / 5.28 / 4.95 s, 2B 1536² 17.53 → 10.43 / 10.23 / 9.10 s,
  2.9B 1024² 9.24 → 5.52 / 7.73 / 6.62 s. At 1024² the low-res stage is too small to use the GPU well
  and only 2 of the 13 full-res steps get predicted, so Spectrum alone wins there. 1536-area
  buckets (1536², 1344x1728, 1728x1344, 1152x2048, 2048x1152): 16.3-16.6 → 9.7-10.0 / - /
  9.0-9.5 s, all coherent.
- Works with every k-diffusion sampler tried (the 8 above for Spectrum, plus dpmpp_2m_sde and
  dpmpp_3m_sde_gpu with SPEED alone); multistep samplers restart their history at the handoff.
  With 20 steps the full-res stage has 8 steps and Spectrum predicts none of them.
- Check: SPEED + Spectrum with `euler` must stay close to the original `KSampler (Spectrum + SPD /
  SPEED)` with `adaptive_smc_alpha` 0 (33-35 dB, 3 prompts x 2 seeds; same composition), and the log
  shows `64x64 for 17 steps, then 128x128 from sigma 0.6970 (aligned 0.8215)` at 1024², 30 steps,
  simple scheduler.

## Guider Track Model — works around a gap in ComfyUI's prompt model tracking

- What it fixes (ComfyUI 0.45.0, dynamic VRAM): `PromptModelTracker.add` (`comfy/model_patcher.py`)
  marks the models of the running prompt from the values that flow into nodes about to execute. It
  knows a `ModelPatcher`, an object with `.patcher` and an object with `get_models()`; a guider
  (`comfy.samplers.CFGGuider`) has only `.model_patcher`. On a cached re-run (new seed, loader and
  guider node cached) the sampled model is left unmarked, and `pin_eviction_tiers`
  (`comfy/model_management.py`) frees the `weights` RAM pins of unmarked models first, even while
  that model is sampling.
- It only shows when three things meet: the model is larger than VRAM (part of it is streamed from
  RAM pins every step), RAM is nearly full (an idle text encoder keeps its pins), and something asks
  for new pins (a video VAE decode pushes model pages out of VRAM). Then the model's own pins are
  dropped and re-read from the file each step.
- The node sets `get_models` on the guider instance (the cached object itself, not a copy) and
  returns it. Relies on internals: the tracker calling `get_models()` on link values, and
  `ModelPatcher.model_patches_models()` / `get_nested_additional_models()`.
- Retire when: `CFGGuider` gets its own `get_models()` or `.patcher`, or the tracker starts reading
  `.model_patcher` (the node already skips guiders that have `get_models`), or the eviction tiers
  stop freeing the pins of a model that is active.
- Check (MiniMax H3 t2v 864x480, 56 frames, 20 steps, video + audio decode, seeds 1, 2, 3 on one
  server): steps 3-20 must stay at the first run's time in runs 2 and 3 (1.30 s here; without the
  node 2.9 and 3.7 s), and the latents must be bit-identical to the same workflow without the node.
  Whether the model is marked can be read from
  `model.model.dynamic_pins[model.load_device]["current_prompt"]` during sampling.

## SnapRef — class-level patch of the Anima self-attention; another project's weight format

- What it is: a second runtime for SnapRef weights (`anima_ref_kv/1`). It must stay equivalent to the
  SnapRef node (https://github.com/ArcAnima/SnapRef, `comfyui_nodes/snapref`), which is the reference
  for every check below. Tested against node commit 87bdacd with `snapref_v2a.safetensors`.
- Replaces `comfy.ldm.cosmos.predict2.Attention.forward` for the whole process the first time the node
  runs. Modules without `_ksr_state` (everything except a model being sampled with this node) fall
  through to the original function. If another pack replaces the same method, the one installed last
  wraps the other; the SnapRef node patches per instance and is not affected.
- Relies on internals:
  - `predict2.Attention`: `forward(x, context, rope_emb, transformer_options)`, `compute_qkv` returning
    q/k/v as `[batch, tokens, heads, dim]`, `compute_attention`, `attn_op`, `comfy_attention`,
    `output_proj`, `output_dropout`; module-level `optimized_attention(q, k, v, heads, mask=,
    skip_reshape=True)` taking a float additive mask (Comfy Kitchen int8 attention does).
  - `predict2.Block.forward(x, emb, context, rope_emb_L_1_1_D=, adaln_lora_B_T_3D=,
    extra_per_block_pos_emb=, transformer_options=)` and `MiniTrainDIT.prepare_embedded_sequence`,
    `t_embedder`, `t_embedding_norm`, `pos_embedder`, `preprocess_text_embeds` (the reference pass
    re-implements the start of the model forward).
  - `torch.compile` modules exposing `_orig_mod` (the reference pass must run uncompiled).
  - `WrappersMP.DIFFUSION_MODEL` called as `(x, timesteps, context, fps, padding_mask, **kwargs)`,
    `transformer_options["cond_or_uncond"]` with 0 = positive, `WrappersMP.APPLY_MODEL` with the
    `BaseModel` as `class_obj`, `BaseModel.current_patcher.patches_uuid`.
  - `CallbacksMP.ON_DETACH` called as `(patcher, unpatch_all)`: `unpatch_all=False` on every sampling
    run (the model is handed to another clone), `True` on a real unload. The reference cache is dropped
    only on `True`.
  - `comfy.weight_adapter.lora.LoRAAdapter(loaded_keys, (up, down, alpha, None, None, None))` and
    `ModelPatcher.add_patches` returning the keys it took.
- What breaks loudly: a changed signature or tensor layout (TypeError / shape error on the first step),
  a weight file with another format id, a model with another block count than 28 or Anima 2.9B's 40.
- What can break silently:
  - `cond_or_uncond` missing or renamed: every row is treated as positive, so the negative branch reads
    the reference too (weaker reference, like `ref_in_uncond`).
  - `ON_DETACH` no longer firing with `True` on unload: the keys/values stay in VRAM (0.9 GB per 1 MP
    reference). If it starts firing with `True` on every run: the reference is encoded again for every
    image (about 0.4 s and a higher step-1 memory peak); the log line
    `KludgeAnimaSnapRef: encoded N reference(s)` then appears once per image instead of once.
  - torch.compile no longer sharing one compiled block between blocks: correct but slow, with
    `torch._dynamo hit config.recompile_limit` in the log. The patch avoids per-block Python values on
    purpose (the SnapRef node's `_i=index` default argument is what trips this).
  - Placed in front of Anima Block Compile it still works, but every change of the reference rebuilds
    the model (`model.clone(disable_dynamic=True)`) and recompiles; after 8 rebuilds in one server the
    recompile limit is reached and blocks run uncompiled. Keep it after Block Compile.
- The target LoRA patches share one `loaded_keys` set so that 2.9B LoRA Remap sees them as one LoRA
  reaching block 39 on a 2.9B model and does not move them.
- Follow the SnapRef node when it changes: new weight versions may add streams, modes or targets
  (`_Adapter`, `_lora_sets`), and the reference pass (`_encode_reference`) mirrors its
  `_encode_reference`. Not ported: `ref_guidance`, `use_batch`, the simple picture-picker node.
- Retire when the SnapRef node itself keeps the cond+uncond batch, merges the target LoRA and compiles
  with Block Compile.
- Check (scripts used on 2026-11-20 are in the portable's `ComfyUI/output/snapref/`):
  - Same computation: bf16 weights, pytorch attention, `--fp32-unet`, same seed, this node against the
    SnapRef node. 1 step / 30 steps PSNR was 46.0 / 41.0 dB at defaults, the same as the SnapRef node
    against itself with another batch layout (45.8 / 41.2 dB), and 40–54 dB for char, strength ±1, two
    references, weights, stitch, ref_megapixels + caption and ref_in_uncond. A wrong formula gives
    15–25 dB. `fidelity` has to be compared under bf16 compute (the SnapRef node only fades the negative
    rows under fp16 / bf16; in fp32 it behaves as ref_in_uncond): 27 dB there, with equal distances to
    the default (9.5 dB) and to ref_in_uncond (16.1 / 16.2 dB) for both nodes.
  - Speed, Anima 2B int8 convrot, CK, 1024², 30 steps, one 1 MP reference, after a warm-up image:
    7.0 s with Block Compile (6.2 s without the node), 8.0 s without compile, one `encoded` log line
    for the whole run and no `recompile_limit` line.
  - Unload: after `POST /free {"unload_models": true}` GPU memory must return to the idle level
    (1.3 GB here after a run with two references that had 7.0 GB in use).
  - 2.9B: SnapRef followed by 2.9B LoRA Remap must be bit-identical to SnapRef alone.

## Releasing to the Comfy Registry

- Registry node id `kludge`, publisher `joykst96` (`pyproject.toml`). Neither can be changed after the
  first publish. The GitHub repository keeps the name `comfyui-kludge`; no code depends on the folder
  name the pack is installed under.
- A push to main that changes `pyproject.toml` runs `.github/workflows/publish_action.yml`, which
  publishes the `version` written in that file. It needs the repository secret `REGISTRY_ACCESS_TOKEN`
  (a publishing API key of the publisher, made on registry.comfy.org). Raise `version` for each release.
  Commits pushed without touching `pyproject.toml` reach only git clone installs.
- `requires-comfyui` is `>=0.45.0`: Guider Track Model and SnapRef were only checked there, and one
  failing import takes the whole pack down. Raise it when a node starts needing something newer.
- Check before pushing a release (comfy-cli, run in the repository):
  - `uvx --from comfy-cli comfy --skip-prompt node validate` checks the metadata and the registry's
    security rules (no `eval` / `exec`, no pip install at run time).
  - `uvx --from comfy-cli comfy --skip-prompt node pack` writes `node.zip` with what would be uploaded
    (git-tracked files). Look inside, then delete it: code, README, MAINTENANCE and the four license
    files, never weights.
  - 2026-11-21, version 0.1.0, comfy-cli through uvx: validate passed, 17 files in the archive.
