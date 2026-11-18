# comfyui-kludge

Makeshift ComfyUI helper nodes, added as needs come up. Category `kludge`.
Most are for Anima (2B / 2.9B): the nodes named Anima are Anima only. Spectrum also works on
MOSS-Image and MiniMax H3, and Guider Track Model works with any model.

This repository was named `comfyui-anima-kludge` before; node IDs are unchanged.

```
Load Diffusion Model → Anima INT8 Quantize → LoRA loader(s) → Anima 2.9B LoRA Remap
  → Anima SPEED → Spectrum → Anima Block Compile → KSampler
```

Every node is optional; keep the order.

## Nodes

**Anima INT8 Quantize** (`KludgeAnimaInt8Quantize`) — quantizes a bf16 Anima model to int8 while
loading, nothing written to disk. `convrot` (default, lower error) or `rowwise` (no rotation).
Same result as a pre-quantized `int8_tensorwise` file. Put it right after Load Diffusion Model.

**Anima 2.9B LoRA Remap** (`KludgeAnima29BLoraRemap`) — lets 2B Anima LoRAs work on Anima 2.9B by
moving their patches onto the 2.9B blocks they came from. Put it after the LoRA loaders (LoRA
Manager, core LoRA loader, ...). LoRAs already made for 2.9B and 2B models are left alone.

**Spectrum** (`KludgeAnimaSpectrum`) — Spectrum feature forecasting: on most steps the
transformer blocks are skipped and their output is predicted from earlier steps. A model patch, so the
normal KSampler stays; the step count is read from the sampler, so there is no steps input to keep in
sync. Defaults match ComfyUI-Spectrum-KSampler (30 steps: 17 full model calls, 13 predicted).
Use er_sde, euler, euler_ancestral, dpmpp_2m, uni_pc or heun; it switches itself off for the
dpmpp_2m/3m SDE samplers, which get color blotches with Spectrum. Keeps the full-call features in
VRAM: about +0.5 GB at 1024² and +1.2 GB at 1536² (batch 1; `history_size` 8 halves it).
Plain Spectrum only: the original's mod guidance, SEA scheduling and CFG corrections are not included.
Also takes MOSS-Image models (SFT, Base, with the edit and control LoRAs): 25 steps run 15 full
steps and 10 predicted, about 35% faster with the same composition and text. A small text block can
still break on some seeds; `warmup_steps` 10 fixed the case found (about 22% faster instead).
Also takes MiniMax H3 video+audio models (tested on text-to-video): the template's 20 steps run 14
full steps and 6 predicted, about 30% faster with the same video and nearly the same audio. The
features are large there (about +1 GB for 2 s and +2.2 GB for 5 s at 864x480), so lower
`history_size` for big or long videos. Few-step turbo models gain nothing (no step is predicted).

**Anima SPEED** (`KludgeAnimaSPEED`) — SPD multi-resolution sampling: the early steps run on a
half-size latent, which is then spectrally expanded to full size at sigma 0.7. Unlike the original
SPEED sampler (Euler only) it wraps whichever sampler the KSampler uses. Changes the composition:
content tags hold, pose and camera tags (lying, from above, drawing bow) can get lost. With
Spectrum on the same model it is the full SPEED; it pays off at large sizes (1536²), at 1024²
Spectrum alone is faster. Only acts on full-denoise sampling: a hires pass or img2img on the same
model runs normally. Anima only: on MOSS-Image it runs but garbles rendered text, and it does not
handle MiniMax H3's packed video+audio latent.

**Anima Block Compile** (`KludgeAnimaBlockCompile`) — `torch.compile` per transformer block, also
with Comfy Kitchen attention (`--use-ck-attention`). Turns off dynamic VRAM for the model and uses a
little more attention memory. The first run after a server start or resolution change compiles.

**Guider Track Model** (`KludgeGuiderTrackModel`) — for workflows that sample through a guider
(`BasicGuider` or `CFGGuider` → `SamplerCustomAdvanced`) with a model larger than VRAM. Without it,
re-running the workflow with a new seed can get 2-3x slower per step from the second run, typically
after a video VAE decode: ComfyUI no longer counts the cached guider's model as part of the running
prompt, drops its pinned RAM first, and re-reads the offloaded part of the model file every step.
Put it between the guider node and SamplerCustomAdvanced's `guider` input. Output is unchanged
(bit-identical latents). Measured on MiniMax H3 (20 GB model, 16 GB VRAM, 40 GB RAM), three runs in
a row with decode: 1.30 / 2.94 / 3.73 s per step without it, 1.30 / 1.30 / 1.30 with it. Not needed
with KSampler, whose model input is tracked directly.

Update risks and checks for each node: [MAINTENANCE.md](MAINTENANCE.md).

## License

MIT, Copyright (c) 2026 joykst96 (see LICENSE). Code only; no model weights are included.
Parts follow other MIT-licensed code, whose notices are kept:

- `block_compile.py`: AnimaBlockCompile from anima_lora by Seunghyun Ji (LICENSE-anima-lora).
- `spectrum.py`, `speed.py`: ComfyUI-Spectrum-KSampler by sorryhyun (LICENSE-spectrum-ksampler);
  the spectral expansion in `speed.py` follows anima_lora's `spd_core` (LICENSE-anima-lora).

## Thanks

These projects were referenced while making this pack. Thank you to their authors.

- [comfyui-anima-blockcompile](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-blockcompile)
  by Seunghyun Ji: the per-block compile node this pack's Block Compile is built on.
- [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) by sorryhyun:
  the Spectrum and SPEED samplers that the Spectrum and SPEED nodes follow and were checked against.
  Papers: Spectrum (Han et al., CVPR 2026, arXiv:2603.01623), SPD (Xiao et al., arXiv:2605.18736).
- [anima_lora](https://github.com/sorryhyun/anima_lora) by Seunghyun Ji: the SPD spectral expansion
  (`networks/spd_core.py`) and the method notes behind Spectrum and SPEED on Anima.
- [ComfyUI-EasyUseAnima](https://github.com/n0va39/ComfyUI-EasyUseAnima) by n0va39: its 2.9B LoRA
  support uses the same 2B → 2.9B block table, which helped confirm the mapping. Its AiO sampler
  showed how Spectrum and SPEED are used in practice.
- [comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools) by Comfy-Org: the int8 ConvRot
  quantizer whose output the INT8 Quantize node was checked against.
- [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) by BobJohnson24: referenced
  for how LoRAs can be handled on int8 models.
- [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) by willmiao: the LoRA loader
  this pack was tested with.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and Comfy Kitchen: the compile wrapper,
  LoRA patching and int8 attention this pack relies on.
