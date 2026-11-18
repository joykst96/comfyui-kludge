"""Spectrum feature forecasting for the Anima DiT (and MOSS-Image, MiniMax H3) as a MODEL patch.

Spectrum (Han et al., CVPR 2026, arXiv:2603.01623): the feature entering final_layer changes smoothly
with the step, so on most steps all transformer blocks are skipped and that feature is predicted from
earlier steps (Chebyshev ridge fit blended with a first-order Taylor step). Only t_embedder, final_layer
and unpatchify run on a predicted step.

Schedule rule, forecaster math and defaults follow ComfyUI-Spectrum-KSampler by sorryhyun, MIT License
(LICENSE-spectrum-ksampler). Different here: the step count and step index come from the sampler's own
sigmas (no steps input), and every sampler call gets fresh forecasters (hires pass, SPEED stages).

MOSS-Image and MiniMax H3 use the same forecaster; only where the feature is caught and how a predicted
step turns it into the output differ. See _Anima / _MossImage / _H3.
"""

import contextlib
import logging
import math

import torch
from comfy.patcher_extension import WrappersMP

try:
    from comfy.model_prefetch import pause_malloc_graph
except ImportError:
    pause_malloc_graph = contextlib.nullcontext

logger = logging.getLogger(__name__)

STATE_KEY = "kludge_spectrum"
# SDE multistep samplers: a predicted step can't react to the noise injected the step before, and their
# multistep correction amplifies the mismatch (iridescent blotches; 3m_sde breaks up). Same with the
# original Spectrum patch. Spectrum stays off for these; er_sde, euler_ancestral, dpmpp_sde are fine.
UNSUPPORTED_SAMPLERS = {f"sample_dpmpp_{s}" for s in ("2m_sde", "2m_sde_gpu", "2m_sde_heun", "2m_sde_heun_gpu",
                                                      "3m_sde", "3m_sde_gpu")}
# set by the SPEED node on its low-resolution stage: those steps always run the full model
LOWRES_KEY = "kludge_spd_lowres"


class _Forecaster:
    """Ridge regression on Chebyshev T_0..T_m of the step index, blended with a Taylor step by `blend`.
    The prediction is a weighted sum of the stored features, so no fp32 copy of the history is made."""

    def __init__(self, degree, lam, blend, total, history):
        self.degree, self.lam, self.blend, self.total, self.history = degree, lam, blend, total, history
        self.t, self.h = [], []

    def update(self, t, h):
        self.t.append(float(t))
        self.h.append(h)
        if len(self.t) > self.history:
            del self.t[0], self.h[0]

    def ready(self):
        return len(self.t) >= max(2, self.degree + 2)

    def _design(self, t):
        tau = (2.0 * t / self.total - 1.0).reshape(-1, 1)
        cols = [torch.ones_like(tau), tau]
        for _ in range(2, self.degree + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return torch.cat(cols[:self.degree + 1], 1)

    @torch.no_grad()
    def predict(self, t_star):
        t = torch.tensor(self.t, dtype=torch.float64)
        x = self._design(t)
        a = x.T @ x + self.lam * torch.eye(x.shape[1], dtype=torch.float64)
        x_star = self._design(torch.tensor([float(t_star)], dtype=torch.float64))[0]
        c = x @ torch.linalg.solve(a, x_star)
        if self.blend < 1.0:
            k = (t_star - self.t[-1]) / max(self.t[-1] - self.t[-2], 1e-8)
            taylor = torch.zeros_like(c)
            taylor[-1], taylor[-2] = 1 + k, -k
            c = (1 - self.blend) * taylor + self.blend * c
        out = torch.zeros(self.h[0].shape, dtype=torch.float32, device=self.h[0].device)
        for ck, hk in zip(c.tolist(), self.h):
            out.add_(hk, alpha=ck)
        return out.to(self.h[0].dtype)


class _Run:
    """Forecast state for one sigma schedule (one sampler call, or one SPEED stage)."""

    def __init__(self, cfg, sigmas):
        self.cfg, self.sigmas = cfg, sigmas
        self.steps = len(sigmas) - 1
        self.window = cfg["window_size"]
        self.consec_cached = 0
        self.step, self.mode = -1, "actual"
        self.forecasters = {}
        self.actual = self.forecast = 0

    def step_of(self, sigma):
        """Index of `sigma` in the schedule, None for an off-grid evaluation (midpoint solvers).
        The model gets sigma 1.0 as 0.99997 (flow model_sampling clamp), hence the tolerance."""
        for i, s in enumerate(self.sigmas[:-1]):
            if abs(s - sigma) <= 1e-4:
                return i
        return None

    def begin_step(self, step, key):
        """Advance the window rule to `step` and decide its mode (same rule as ComfyUI-Spectrum-KSampler)."""
        if step == self.step:
            return
        if self.mode == "actual":
            if self.step >= self.cfg["warmup_steps"]:
                self.window = round(self.window + self.cfg["flex_window"], 3)
            self.consec_cached = 0
        else:
            self.consec_cached += 1
        self.step = step
        forecaster = self.forecasters.get(key)
        cache = (self.cfg["warmup_steps"] <= step < self.steps - self.cfg["tail_actual_steps"]
                 and forecaster is not None and forecaster.ready()
                 and (self.consec_cached + 1) % max(1, math.floor(self.window)) != 0)
        self.mode = "cached" if cache else "actual"

    def forecaster(self, key, feat):
        f = self.forecasters.get(key)
        if f is not None and f.h[0].shape != feat.shape:
            f = None
        if f is None:
            cfg = self.cfg
            f = self.forecasters[key] = _Forecaster(cfg["cheby_degree"], cfg["ridge_lambda"], cfg["blend_w"],
                                                    self.steps, cfg["history_size"])
        return f


class _State:
    """Lives in transformer_options for one sampler call; holds the run of the current schedule."""

    def __init__(self, cfg):
        self.cfg, self.run, self._src = cfg, None, None
        self.done = []

    def run_for(self, sample_sigmas):
        if sample_sigmas is None:
            return None
        if sample_sigmas is not self._src:
            sigmas = [float(s) for s in sample_sigmas.tolist()]
            if self.run is None or sigmas != self.run.sigmas:
                if self.run is not None:
                    self.done.append(self.run)
                    self.run.forecasters = {}
                self.run = _Run(self.cfg, sigmas)
            self._src = sample_sigmas
        return self.run

    def finish(self):
        runs = self.done + ([self.run] if self.run is not None else [])
        for r in runs:
            r.forecasters = {}
        self.run = self._src = None
        return runs


def _batch_key(transformer_options, x):
    uuids = transformer_options.get("uuids") or ()
    cond_or_uncond = transformer_options.get("cond_or_uncond") or ()
    replaced = tuple(sorted(map(str, transformer_options.get("patches_replace", {}).get("dit", {}))))
    shape = tuple(tuple(t.shape) for t in x) if isinstance(x, (list, tuple)) else tuple(x.shape)  # H3: [video, audio]
    return tuple(cond_or_uncond), tuple(map(str, uuids)), replaced, shape


def _copy(t):
    # the copy must not come from the memory compiler's pool
    with pause_malloc_graph():
        return t.detach().clone()


class _Anima:
    """MiniTrainDIT (comfy/ldm/cosmos/predict2.py): the feature is the input of final_layer."""
    name = "Anima"
    needs = ("final_layer", "t_embedder", "t_embedding_norm", "unpatchify")

    @staticmethod
    def transformer_options(args, kwargs):
        return kwargs.get("transformer_options", {})

    @staticmethod
    def hook(dit, x, captured):
        return dit.final_layer.register_forward_pre_hook(lambda module, inputs: captured.append(_copy(inputs[0])))

    @staticmethod
    def feature(captured):
        return captured[0] if len(captured) == 1 else None

    @staticmethod
    def head(dit, feat, timesteps, x_shape):
        """The part of MiniTrainDIT._forward after the blocks, fed with a predicted feature."""
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = dit.t_embedder[1](dit.t_embedder[0](timesteps).to(feat.dtype))
        t_emb = dit.t_embedding_norm(t_emb)
        out = dit.unpatchify(dit.final_layer(feat, t_emb, adaln_lora_B_T_3D=adaln_lora))
        return out[:, :, :x_shape[-3], :x_shape[-2], :x_shape[-1]]

    @classmethod
    def predicted(cls, dit, feat, x, timesteps, call):
        return cls.head(dit, feat, timesteps, x.shape)


class _MossImage:
    """MOSSImage (comfy/ldm/moss_image/model.py): one sequence of image, text and reference tokens. The
    feature is the image tokens leaving the last block; there is no module call between the blocks and
    the head to hook, so the last block's output is copied instead."""
    name = "MOSS-Image"
    needs = ("layers", "time_embedding", "video_head_modulation", "final_norm_video", "final_linear_video")

    @staticmethod
    def transformer_options(args, kwargs):
        # MOSSImage.forward hands transformer_options to the wrappers positionally
        return args[0] if args and isinstance(args[0], dict) else kwargs.get("transformer_options", {})

    @staticmethod
    def hook(dit, x, captured):
        image_tokens = x.shape[-2] * x.shape[-1]
        return dit.layers[-1].register_forward_hook(
            lambda module, inputs, output: captured.append(_copy(output[:, :image_tokens])))

    @staticmethod
    def feature(captured):
        # preprocess_refs runs the blocks too when the reference keys are not precomputed; the main pass is last
        return captured[-1] if captured else None

    @staticmethod
    def head(dit, feat, timesteps, x_shape):
        """The part of MOSSImage._forward after the blocks, fed with a predicted feature."""
        from comfy.ldm.flux.layers import timestep_embedding
        from comfy.ldm.moss_image.model import modulated_rms_norm
        vec = dit.time_embedding(timestep_embedding(timesteps, 256).to(feat.dtype))
        head_shift, head_scale = dit.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        out = dit.final_linear_video(modulated_rms_norm(feat, dit.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(x_shape[0], dit.latent_channels, x_shape[-2], x_shape[-1])

    @classmethod
    def predicted(cls, dit, feat, x, timesteps, call):
        return cls.head(dit, feat, timesteps, x.shape)


class _H3:
    """MiniMaxH3Model / MiniMaxH3ImageModel (comfy/ldm/minimax/model.py): one packed sequence
    [text | references | audio | video], x is [video, audio]. The feature is the target rows (audio and
    video, the last segments) of the sequence entering final_layer. A predicted step runs the model's own
    forward with _run_blocks swapped for a function that drops the forecast into those rows, so the embed,
    the timestep table and the output heads stay ComfyUI's code instead of a copy here."""
    name = "MiniMax H3"
    needs = ("blocks", "final_layer", "token_refiner", "video_patch_proj", "_run_blocks")

    @staticmethod
    def transformer_options(args, kwargs):
        # forward hands transformer_options to the wrappers positionally
        return args[0] if args and isinstance(args[0], dict) else kwargs.get("transformer_options", {})

    @staticmethod
    def hook(dit, x, captured):
        def pre_hook(module, inputs):
            h, video_seg, audio_seg = inputs[0], inputs[2], inputs[3]  # final_layer(x, t_emb, video_seg, audio_seg, ...)
            start = video_seg[0] if audio_seg is None else min(video_seg[0], audio_seg[0])
            captured.append(_copy(h[start:]))
        return dit.final_layer.register_forward_pre_hook(pre_hook)

    @staticmethod
    def feature(captured):
        return captured[-1] if captured else None

    @staticmethod
    def predicted(dit, feat, x, timesteps, call):
        def run_blocks(h, t_emb, mod_segments, layout, transformer_options):
            h[h.shape[0] - feat.shape[0]:] = feat
            return h
        dit._run_blocks = run_blocks  # instance attribute shadows the method for this call only
        try:
            return call()
        finally:
            del dit._run_blocks


ARCHS = (_Anima, _MossImage, _H3)


def _diffusion_model_wrapper(arch):
    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        transformer_options = arch.transformer_options(args, kwargs)
        state = transformer_options.get(STATE_KEY)
        if state is None or transformer_options.get(LOWRES_KEY) or kwargs.get("control") is not None:
            return executor(x, timesteps, context, *args, **kwargs)
        run = state.run_for(transformer_options.get("sample_sigmas"))
        sigmas = transformer_options.get("sigmas")
        step = run.step_of(sigmas.flatten()[0].item()) if run is not None and sigmas is not None else None
        if step is None:
            return executor(x, timesteps, context, *args, **kwargs)

        dit = executor.class_obj
        key = _batch_key(transformer_options, x)
        run.begin_step(step, key)
        forecaster = run.forecasters.get(key)
        if run.mode == "cached" and forecaster is not None and forecaster.ready():
            run.forecast += 1
            with pause_malloc_graph():  # the forecast must not come from the memory compiler's pool
                feat = forecaster.predict(step)
            return arch.predicted(dit, feat, x, timesteps, lambda: executor(x, timesteps, context, *args, **kwargs))

        captured = []
        handle = arch.hook(dit, x, captured)
        try:
            out = executor(x, timesteps, context, *args, **kwargs)
        finally:
            handle.remove()
        run.actual += 1
        feat = arch.feature(captured)
        if feat is not None:
            f = run.forecaster(key, feat)
            if f.t and f.t[-1] == step:  # re-evaluation at the same sigma (Heun etc.): keep the latest
                f.h[-1] = feat
            else:
                f.update(step, feat)
        return out
    return wrapper


def _sampler_sample_wrapper(cfg):
    def wrapper(executor, guider, sigmas, extra_args, *args, **kwargs):
        name = getattr(getattr(executor.class_obj, "sampler_function", None), "__name__", "")
        if name in UNSUPPORTED_SAMPLERS:
            logger.warning("KludgeAnimaSpectrum: off for %s (SDE multistep samplers get color blotches with "
                           "Spectrum); use er_sde, euler, dpmpp_2m or euler_ancestral", name[len("sample_"):])
            return executor(guider, sigmas, extra_args, *args, **kwargs)
        transformer_options = extra_args["model_options"].setdefault("transformer_options", {})
        state = transformer_options[STATE_KEY] = _State(cfg)
        try:
            return executor(guider, sigmas, extra_args, *args, **kwargs)
        finally:
            transformer_options.pop(STATE_KEY, None)
            for r in state.finish():
                logger.info("KludgeAnimaSpectrum: %d steps, %d full model calls, %d predicted",
                            r.steps, r.actual, r.forecast)
    return wrapper


class KludgeAnimaSpectrum:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "warmup_steps": ("INT", {"default": 6, "min": 0, "max": 100,
                                     "tooltip": "First steps that always run the full model."}),
            "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.05,
                                      "tooltip": "Start window: one full model call every N steps."}),
            "flex_window": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "Window growth after each full call. Higher is faster."}),
            "tail_actual_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                          "tooltip": "Last steps that always run the full model."}),
            "blend_w": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "1.0 = pure Chebyshev fit, 0.0 = pure Taylor step."}),
            "cheby_degree": ("INT", {"default": 3, "min": 1, "max": 8}),
            "ridge_lambda": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
            "history_size": ("INT", {"default": 100, "min": 3, "max": 1000,
                                     "tooltip": "Full-call features kept per sampler call (VRAM)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima/kludge"
    DESCRIPTION = ("Spectrum: skip the transformer blocks on most steps and predict their output. "
                   "Step count comes from the sampler, so it works with any sampler, scheduler and denoise. "
                   "For Anima, MOSS-Image and MiniMax H3 models.")
    EXPERIMENTAL = True

    def patch(self, model, **cfg):
        if cfg["history_size"] < cfg["cheby_degree"] + 2:
            raise ValueError("history_size must be at least cheby_degree + 2")
        dit = model.get_model_object("diffusion_model")
        arch = next((a for a in ARCHS if all(hasattr(dit, name) for name in a.needs)), None)
        if arch is None:
            raise ValueError(f"KludgeAnimaSpectrum needs an Anima / Cosmos DiT, MOSS-Image or MiniMax H3 model, got {type(dit).__name__}")
        m = model.clone()
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, STATE_KEY, _diffusion_model_wrapper(arch))
        m.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, STATE_KEY, _sampler_sample_wrapper(cfg))
        return (m,)
