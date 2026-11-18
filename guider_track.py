"""Guider Track Model.

With dynamic VRAM, ComfyUI marks the models of the running prompt through PromptModelTracker.add
(comfy/model_patcher.py). It is called with every value that flows along a link into a node that is going to
execute, and it recognises a ModelPatcher, an object with `.patcher` (CLIP, VAE) and an object with
`get_models()`. A GUIDER (comfy.samplers.CFGGuider) only has `.model_patcher`. So when the loader and the guider
node are served from the cache and only SamplerCustomAdvanced runs again (same workflow, new seed), the sampled
model is not marked: dynamic_pins[device]["current_prompt"] stays False.

Pin eviction (comfy.model_management.pin_eviction_tiers) frees the "weights" RAM pins of unmarked models first.
For a model larger than VRAM those pins hold the part that is streamed to the GPU on every step. Once RAM is
short and anything asks for new pins (a video VAE decode pushing model pages out of VRAM is enough), the model
loses its own pins and re-reads that part from the model file on every step.

The node gives the guider a get_models(), so the tracker sees its model. Sampling itself is untouched.
"""


def _guider_models(guider):
    patcher = guider.model_patcher
    return [patcher, *patcher.model_patches_models(), *patcher.get_nested_additional_models()]


class KludgeGuiderTrackModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"guider": ("GUIDER",)}}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "track"
    CATEGORY = "kludge"
    DESCRIPTION = ("Keeps the guider's model marked as used by the current prompt when the workflow is re-run from "
                   "the cache (dynamic VRAM). Place between the guider node (BasicGuider, CFGGuider, ...) and "
                   "SamplerCustomAdvanced. Fixes sampling getting 2-3x slower from the second run on models larger "
                   "than VRAM. Output is unchanged. Not needed with KSampler.")

    def track(self, guider):
        if not callable(getattr(guider, "get_models", None)) and hasattr(guider, "model_patcher"):
            guider.get_models = lambda g=guider: _guider_models(g)
        return (guider,)
