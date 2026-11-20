"""SnapRef reference adapter for Anima, with a runtime that keeps int8 models and Block Compile fast.

Reads SnapRef weight files (format anima_ref_kv/1) and computes what the SnapRef node computes: the
reference picture runs once through the model with its own LoRA and every block's self-attention keys
and values are cached; target forwards then read those keys/values in self-attention, with the target
LoRA on.

Parts of this file follow SnapRef's ComfyUI node (comfyui_nodes/snapref/__init__.py in
https://github.com/ArcAnima/SnapRef, Apache License 2.0, see LICENSE-snapref): the weight file
loading and FP8 restore (_Adapter), the 2.9B block table, the RoPE positions and the forward of the
reference stream (_rope, _encode_reference), the reference image sizing and stitching (_encode_image,
_stitch), the cache key (_ref_digest) and the meaning of mode / strength / weights / fidelity.
Changed here: how the model is patched and how the target forward runs.

  1. The target LoRA is merged into the weights as an ordinary ComfyUI LoRA patch. SnapRef adds it with
     forward hooks on 112 Linears in every model call, which is slow on int8 models.
  2. A cond+uncond batch stays one model call. SnapRef splits it into two batch-1 calls so that only
     cond reads the reference; int8 models lose much of their speed that way. Here only self-attention
     is split by row.
  3. The reference is encoded with the uncompiled blocks (block._orig_mod) under hooks that exist only
     for that pass: reference LoRA minus target LoRA, because the weights already hold the target LoRA.
  4. The self-attention patch is one class-level function; per-block data are module attributes with
     the same types and constants on every block, so torch.compile shares one compiled block between
     all blocks instead of hitting its recompile limit.
  5. strength, reference weights and fidelity are an additive attention mask on the reference keys,
     passed to the model's own attention function (Comfy Kitchen takes float masks).
Not ported: ref_guidance, use_batch, the picture picker node.
"""

import hashlib
import json
import logging
import math
import uuid

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.ldm.cosmos.predict2 as predict2
import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy.weight_adapter.lora import LoRAAdapter

logger = logging.getLogger(__name__)

KEY = "kludge_snapref"
# Anima-2.9B inserts these blocks into the 28-block base
_INSERTED_29B = frozenset((2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36))
_adapters = {}
_cache = {}   # one reference set at a time: key -> (per-block (k, v), tokens per reference)
_state = {"patched": False, "patches_uuid": None}


# ---------------------------------------------------------------------------------------------------
# self-attention patch

def _attend(attn, q, k, v, bias, transformer_options):
    """q, k, v [B, S, heads, dim] -> [B, S, heads * dim]. bias: None, or [B or 1, 1, 1, reference tokens],
    added to the attention logits of the reference keys (the first keys of k)."""
    if bias is None:
        return attn.attn_op(q, k, v, transformer_options=transformer_options, preferred_attention=attn.comfy_attention)
    mask = F.pad(bias.to(q.dtype), (0, k.shape[1] - bias.shape[-1]))
    return predict2.optimized_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), q.shape[-2], mask=mask,
                                        skip_reshape=True, preferred_attention=attn.comfy_attention,
                                        transformer_options=transformer_options)


def _attention_forward(self, x, context=None, rope_emb=None, transformer_options={}):
    state = getattr(self, "_ksr_state", None)
    if state is None:
        return _orig_attention_forward(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
    q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
    if state == "record":   # reference pass (uncompiled blocks)
        self._ksr_rec = (k, v)
        return self.compute_attention(q, k, v, transformer_options=transformer_options)
    k_r, v_r = self._ksr_k, self._ksr_v   # [1, reference tokens, heads, dim]
    bias = self._ksr_bias
    n = self._ksr_split   # rows [0, n) and [n, B) are the two groups; 0 = every row reads the reference
    b = q.shape[0]
    if n == 0:
        out = _attend(self, q, torch.cat([k_r.expand(b, -1, -1, -1).to(k.dtype), k], dim=1),
                      torch.cat([v_r.expand(b, -1, -1, -1).to(v.dtype), v], dim=1), bias, transformer_options)
        return self.output_dropout(self.output_proj(out))
    if self._ksr_ref_last:
        plain, ref, m = slice(0, n), slice(n, b), b - n
    else:
        ref, plain, m = slice(0, n), slice(n, b), n
    out_ref = _attend(self, q[ref], torch.cat([k_r.expand(m, -1, -1, -1).to(k.dtype), k[ref]], dim=1),
                      torch.cat([v_r.expand(m, -1, -1, -1).to(v.dtype), v[ref]], dim=1), bias, transformer_options)
    out_plain = _attend(self, q[plain], k[plain], v[plain], None, transformer_options)
    out = torch.cat([out_plain, out_ref] if self._ksr_ref_last else [out_ref, out_plain], dim=0)
    return self.output_dropout(self.output_proj(out))


_orig_attention_forward = predict2.Attention.forward


def _patch_attention():
    """Class-level patch, installed on first use. Without _ksr_state on a module it is the original forward."""
    if not _state["patched"]:
        predict2.Attention.forward = _attention_forward
        _state["patched"] = True


# ---------------------------------------------------------------------------------------------------
# weight file

class _Adapter:
    def __init__(self, path):
        tensors = load_file(path)
        with safe_open(path, "pt") as handle:
            md = handle.metadata() or {}
        self.meta = json.loads(md.get("snapref") or md.get("anima_ref_kv") or "{}")
        if self.meta.get("format") != "anima_ref_kv/1":
            raise ValueError(f"{path} is not a SnapRef (anima_ref_kv/1) weight file")
        self.ref_t = int(self.meta.get("ref_t", 10))
        self.modes = list(self.meta.get("modes", ["cs"]))
        self.mode_embed = tensors.get("anima_ref_kv.mode_embed")
        self.lora = {}   # module name -> stream ("ref" / "tgt") -> (down, up, scale)
        for key in tensors:
            if not key.endswith(".lora_down"):
                continue
            root = key[: -len("lora_down")]
            _, stream, name = root[:-1].split(".", 2)
            down, up = (self._restore(tensors, root + n) for n in ("lora_down", "lora_up"))
            self.lora.setdefault(name, {})[stream] = (down, up, float(tensors[root + "alpha"]) / down.shape[0])
        self.block_indices = {int(n.split(".")[1]) for n in self.lora if n.startswith("blocks.")}

    @staticmethod
    def _restore(tensors, key):
        """float32 matrix; FP8 files carry a per-row float32 <key>_scale."""
        w = tensors[key].float()
        if key + "_scale" in tensors:
            w = w * tensors[key + "_scale"]
        return w


def _block_map(indices, dm):
    if not indices:
        raise ValueError("SnapRef weight file has no diffusion block weights")
    blocks = len(dm.blocks)
    if max(indices) + 1 == blocks:
        return tuple(range(blocks))
    if blocks == 40 and set(indices) == set(range(28)):
        return tuple(i for i in range(40) if i not in _INSERTED_29B)
    raise ValueError(f"SnapRef weights cover {max(indices) + 1} blocks but the model has {blocks}. "
                     "Only the native layout and Anima-2.9B's 28-to-40 expansion are supported.")


def _lora_sets(ad, block_map):
    """-> (weight patches for the target LoRA, {module name: (down, up)} for the reference pass).
    The reference pass runs on weights that already hold the target LoRA, so its delta is ref - tgt."""
    patches, ref_lora = {}, {}
    # one shared set, as for the patches of one LoRA file: 2.9B LoRA Remap then sees a single LoRA that reaches
    # block 39 on a 2.9B model and leaves it alone, wherever it is placed
    loaded_keys = set()
    for name, streams in ad.lora.items():
        if name.startswith("blocks."):
            _, index, suffix = name.split(".", 2)
            name = f"blocks.{block_map[int(index)]}.{suffix}"
        downs, ups = [], []
        if "ref" in streams:
            down, up, scale = streams["ref"]
            downs.append(down)
            ups.append(up * scale)
        if "tgt" in streams:
            down, up, scale = streams["tgt"]
            patches[f"diffusion_model.{name}.weight"] = LoRAAdapter(loaded_keys, (up, down, scale * down.shape[0], None, None, None))
            downs.append(down)
            ups.append(-up * scale)
        ref_lora[name] = (torch.cat(downs, dim=0), torch.cat(ups, dim=1))
    return patches, ref_lora


# ---------------------------------------------------------------------------------------------------
# reference encoding

def _rope(pe, t, h, w):
    """VideoRopePosition3DEmb for explicit per-token positions."""
    def rot(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)
    dev = t.device
    emb = torch.cat([rot(t, 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(dev))),
                     rot(h, 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(dev))),
                     rot(w, 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(dev)))], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


@torch.no_grad()
def _encode_reference(ad, dm, ref_lora, ref, context, target_hw, mode):
    """Per-block (k, v) of one reference latent: sigma 0, RoPE frame ref_t centred on the target."""
    blocks = [getattr(b, "_orig_mod", b) for b in dm.blocks]   # never the compiled ones
    x_emb, _, _ = dm.prepare_embedded_sequence(ref, fps=None, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape
    h0 = round((target_hw[0] // 2) / 2 - hp / 2)
    w0 = round((target_hw[1] // 2) / 2 - wp / 2)
    hh, ww = torch.meshgrid(torch.arange(hp, device=ref.device) + h0, torch.arange(wp, device=ref.device) + w0, indexing="ij")
    rope = _rope(dm.pos_embedder, torch.full_like(hh, ad.ref_t).flatten(), hh.flatten(), ww.flatten())
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](torch.zeros(ref.shape[0], 1, device=ref.device)).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    if ad.mode_embed is not None and mode in ad.modes:
        t_emb = t_emb + ad.mode_embed[ad.modes.index(mode)].to(t_emb)[None, None, :]
    handles = []
    for name, (down, up) in ref_lora.items():
        if name.startswith("blocks."):
            _, index, suffix = name.split(".", 2)
            module = blocks[int(index)].get_submodule(suffix)
        else:
            module = dm.get_submodule(name)
        down, up = down.to(ref.device, x_emb.dtype), up.to(ref.device, x_emb.dtype)

        def hook(mod, args, out, _down=down, _up=up):
            return out + F.linear(F.linear(args[0], _down), _up)
        handles.append(module.register_forward_hook(hook))
    attns = [b.self_attn for b in blocks]
    for attn in attns:
        attn._ksr_state = "record"
    try:
        # fp32 residual stream under fp16 compute, as the model's own forward does
        x = x_emb.float() if x_emb.dtype == torch.float16 else x_emb
        for block in blocks:
            x = block(x, t_emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0), adaln_lora_B_T_3D=adaln,
                      extra_per_block_pos_emb=None, transformer_options={})
        return [attn._ksr_rec for attn in attns]
    finally:
        for attn in attns:
            attn._ksr_state = None
            attn._ksr_rec = None
        for handle in handles:
            handle.remove()


def _encode_image(model, vae, image, megapixels):
    """One image [1, H, W, C] -> model-space latent [1, 16, 1, h, w], shrunk (never enlarged) to megapixels."""
    pixels = image[:1].movedim(-1, 1)
    h, w = pixels.shape[-2:]
    s = min(1.0, (megapixels * 1024 * 1024 / (w * h)) ** 0.5)
    width, height = max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)
    pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
    latent = vae.encode(pixels[..., :3])
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return model.get_model_object("process_latent_in")(latent), (width, height)


def _stitch(refs):
    """Side by side at the first image's height."""
    h = refs[0].shape[1]
    parts = []
    for px in refs:
        c = px[..., :3].movedim(-1, 1)
        w = max(1, round(c.shape[-1] * h / c.shape[-2]))
        parts.append(comfy.utils.common_upscale(c, w, h, "lanczos", "disabled").movedim(1, -1))
    return torch.cat(parts, dim=2)


def _digest(ref_latents, ref_cond, mode):
    h = hashlib.sha1(mode.encode())
    for t in [*ref_latents, ref_cond[0]]:
        t = t.detach().float().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _model_token(dm):
    token = getattr(dm, "_ksr_token", None)
    if token is None:
        token = dm._ksr_token = uuid.uuid4().hex
    return token


# ---------------------------------------------------------------------------------------------------
# model wrappers

def _apply_model_wrapper(executor, *args, **kwargs):
    # the weights the reference will be encoded with: another LoRA set must not reuse the cached keys/values
    patcher = getattr(executor.class_obj, "current_patcher", None)
    _state["patches_uuid"] = getattr(patcher, "patches_uuid", None)
    return executor(*args, **kwargs)


def _make_wrapper(ad, ref_lora, ref_latents, ref_cond, mode, biases, in_uncond, uncond_bias):
    """biases: logit bias per reference (strength + log weight). in_uncond: every row reads the reference.
    uncond_bias: None, or the extra bias with which the negative rows read it (fidelity < 1)."""
    digest = _digest(ref_latents, ref_cond, mode)

    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        dm = executor.class_obj
        options = kwargs.get("transformer_options", {})
        key = (digest, _model_token(dm), _state["patches_uuid"], tuple(x.shape[-2:]), x.dtype, x.device)
        if key not in _cache:
            _cache.clear()
            cross, meta = ref_cond
            ids, weights = meta.get("t5xxl_ids"), meta.get("t5xxl_weights")
            ctx = dm.preprocess_text_embeds(cross.to(x.device, x.dtype),
                                            None if ids is None else ids.unsqueeze(0).to(x.device),
                                            None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x.device, x.dtype))
            records = [_encode_reference(ad, dm, ref_lora, lat.to(x.device, x.dtype), ctx, tuple(x.shape[-2:]), mode)
                       for lat in ref_latents]
            sizes = [r[0][0].shape[1] for r in records]
            if len(records) > 1:
                records = [[(torch.cat([r[i][0] for r in records], dim=1), torch.cat([r[i][1] for r in records], dim=1))
                            for i in range(len(records[0]))]]
            _cache[key] = (records[0], sizes)
            logger.info("KludgeAnimaSnapRef: encoded %d reference(s), %d tokens", len(sizes), sum(sizes))
        kv, sizes = _cache[key]

        flags = options.get("cond_or_uncond") or [0]
        positive = [f == 0 for f in flags]
        rows = x.shape[0] // len(flags)
        fade = uncond_bias is not None and not all(positive)   # negative rows read the reference, weaker
        if in_uncond or fade or all(positive):
            split, ref_last = 0, True
        elif not any(positive):
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        elif positive == sorted(positive):                 # [negative ..., positive ...]
            split, ref_last = rows * positive.index(True), True
        elif positive == sorted(positive, reverse=True):   # [positive ..., negative ...]
            split, ref_last = rows * positive.index(False), False
        else:
            raise RuntimeError(f"SnapRef (kludge): unsupported cond/uncond layout {flags}")
        bias = None
        if fade or any(b != 0.0 for b in biases):
            bias = torch.cat([torch.full((n,), b, device=x.device, dtype=torch.float32) for n, b in zip(sizes, biases)])
            bias = bias.view(1, 1, 1, -1)
            if fade:
                extra = torch.tensor([0.0 if p else uncond_bias for p in positive for _ in range(rows)],
                                     device=x.device, dtype=torch.float32)
                bias = bias + extra.view(-1, 1, 1, 1)
        attns = [getattr(b, "_orig_mod", b).self_attn for b in dm.blocks]
        for attn, (k_r, v_r) in zip(attns, kv):
            attn._ksr_k, attn._ksr_v, attn._ksr_bias = k_r, v_r, bias
            attn._ksr_split, attn._ksr_ref_last, attn._ksr_state = split, ref_last, "read"
        try:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        finally:
            for attn in attns:
                attn._ksr_state = None
                attn._ksr_k = attn._ksr_v = attn._ksr_bias = None
    return wrapper


def _on_detach(patcher, unpatch_all=True):
    # unpatch_all False = the loaded model is only handed to another clone (every sampling run); True = unloaded
    if unpatch_all:
        _cache.clear()


def _adapter_files():
    names = folder_paths.get_filename_list("loras")
    return [n for n in names if "snapref" in n.lower() or "anima_ref_kv" in n] or names


class KludgeAnimaSnapRef:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP", {"tooltip": "Encodes ref_caption for the reference."}),
            "vae": ("VAE",),
            "image": ("IMAGE", {"tooltip": "The reference picture (first image of a batch)."}),
            "adapter": (_adapter_files(), {"tooltip": "SnapRef weights in models/loras."}),
            "mode": (["cs", "char"], {"tooltip": "cs: character and art style from the reference. "
                                                 "char: character only; style from the model and the prompt."}),
            "strength": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                                   "tooltip": "Attention to the reference, log scale. 0 = as trained, +1 = about 2.7x "
                                              "more, -1 = about 2.7x less (the outfit follows the prompt more)."}),
            "fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "1 = keep every detail of the reference. Lower: the negative branch reads "
                                              "the reference too, so the prompt is followed more freely. "
                                              "Ignored when ref_in_uncond is on."}),
        }, "optional": {
            "image_2": ("IMAGE", {"tooltip": "Another picture of the same character (for example a full-body view)."}),
            "image_3": ("IMAGE", {"tooltip": "A third picture of the same character."}),
            "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                   "tooltip": "How much image counts next to the other references. 0 = left out. "
                                              "Only with multi_layout separate."}),
            "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "How much image_2 counts."}),
            "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "How much image_3 counts."}),
            "multi_layout": (["separate", "stitch"], {"tooltip": "separate: every reference encoded on its own. stitch: "
                                                      "side by side in one reference (also for two characters)."}),
            "ref_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05,
                                         "tooltip": "Each reference is shrunk (never enlarged) to this area. "
                                                    "Trained at 1.0 or less; smaller is a little faster."}),
            "ref_caption": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "Optional appearance tags of the reference. Empty works."}),
            "ref_in_uncond": ("BOOLEAN", {"default": False, "tooltip": "The negative branch reads the reference fully "
                                          "(same as fidelity 0): weaker reference, closer to the prompt."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "kludge"
    DESCRIPTION = ("SnapRef character reference for Anima, with a runtime that stays fast on int8 models and "
                   "works with Anima Block Compile. Needs SnapRef weights in models/loras. Write only the scene "
                   "in the prompt.")

    def apply(self, model, clip, vae, image, adapter, mode, strength, fidelity, image_2=None, image_3=None,
              weight_1=1.0, weight_2=1.0, weight_3=1.0, multi_layout="separate", ref_megapixels=1.0, ref_caption="",
              ref_in_uncond=False):
        path = folder_paths.get_full_path_or_raise("loras", adapter)
        if path not in _adapters:
            _adapters[path] = _Adapter(path)
        ad = _adapters[path]
        block_map = _block_map(ad.block_indices, model.get_model_object("diffusion_model"))
        refs = [(px[:1], float(w)) for px, w in ((image, weight_1), (image_2, weight_2), (image_3, weight_3))
                if px is not None and w > 0]
        if not refs:
            raise ValueError("SnapRef (kludge): every reference has weight 0")
        if len(refs) > 1 and multi_layout == "stitch":
            refs = [(_stitch([px for px, _ in refs]), 1.0)]
        latents, sizes = zip(*(_encode_image(model, vae, px, ref_megapixels) for px, _ in refs))
        biases = [float(strength) + math.log(w) for _, w in refs]
        uncond_bias = None if fidelity >= 1.0 or ref_in_uncond else math.log(max(1.0 - float(fidelity), 1e-6))
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(ref_caption.strip()))[0]
        patches, ref_lora = _lora_sets(ad, block_map)

        _patch_attention()
        m = model.clone()
        loaded = m.add_patches(patches, 1.0)
        if len(loaded) != len(patches):
            raise RuntimeError(f"SnapRef (kludge): only {len(loaded)} of {len(patches)} target LoRA weights fit this model")
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, KEY, _apply_model_wrapper)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY,
                               _make_wrapper(ad, ref_lora, list(latents), cond, mode, biases, ref_in_uncond, uncond_bias))
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_DETACH, KEY, _on_detach)
        logger.info("KludgeAnimaSnapRef: %s, reference %s, mode %s, strength %g, fidelity %g%s", adapter,
                    ", ".join(f"{w}x{h} w{wt:g}" for (w, h), (_, wt) in zip(sizes, refs)), mode, strength, fidelity,
                    ", in uncond" if ref_in_uncond else "")
        return (m,)
