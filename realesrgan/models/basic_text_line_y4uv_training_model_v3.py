"""Training wrapper for the isolated six-channel BASIC V2 Y4UV path."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch.nn import functional as F
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_mask_to_yuv444, y4uv_to_yuv444
from realesrgan.data.reference_loss_options import resolve_reference_loss_flags
from realesrgan.data.text_line_crop_v3 import polygon_to_quad
from realesrgan.losses.basic_text_line_y4uv_v3_losses import (
    Y4UVNativeMaskedL1LossV3,
    YUV444ReferenceMaskedL1LossV3,
    compare_y4uv_losses,
)
from realesrgan.utils import YUV2RGB
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses
from realesrgan.losses.text_gestalt_loss import MaskedTextGestaltLoss
from realesrgan.losses.thin_skeleton_fbeta_loss import ThinSkeletonFbetaLoss
from realesrgan.losses.stroke_geometry_losses import SoftCLDiceLoss, StrokeProbabilityBCELoss, StrokeWidthLoss
from realesrgan.losses.keep_loss import KeepLoss
from realesrgan.losses.gt_sdf_boundary_loss import GTSDFBoundaryLoss
from realesrgan.losses.normal_profile_loss import BoundaryNormalProfileLoss
from realesrgan.losses.stroke_core_overshoot_loss import StrokeCoreLoss, OvershootLoss
from realesrgan.losses.uv_structure_supervision_loss import UVStrongSupervisionLoss


@MODEL_REGISTRY.register()
class BasicTextLineY4UVTrainingModelV3(SRModel):
    """Train Y4UV residuals while logging native and decoded-reference losses."""

    def init_training_settings(self):
        """@shanlan-comment
        Input: model options containing native/reference/CP/gradient and optional SkelFbeta settings.
        Output: initialized loss modules, reference switches, optimizer, and scheduler.
        Key Steps: build configured losses, resolve legacy-true reference flags,
        and validate the non-negative SkelFbeta experiment weight.
        Key Params: ``train_reference_loss`` gates per-step decoded-reference
        work; validation has its own switch; ``skel_fbeta_weight`` gates the
        training-only skeleton objective.
        State/Dependencies: BasicSR loss registry and optimizer state are initialized in place.
        """
        self.net_g.train()
        train_opt = self.opt["train"]
        native_opt = train_opt.get(
            "y4uv_native_opt",
            {"type": "Y4UVNativeMaskedL1LossV3", "channel_weight": (1.0, 1.0, 1.0, 1.0, 0.25, 0.25)},
        )
        reference_opt = train_opt.get(
            "yuv444_reference_opt",
            {"type": "YUV444ReferenceMaskedL1LossV3", "channel_weight": (1.0, 0.25, 0.25)},
        )
        # Importing the isolated loss module above registers the two classes
        # with BasicSR before build_loss resolves their config names.
        self.cri_y4uv_native = build_loss(native_opt).to(self.device)
        self.cri_yuv444_reference = build_loss(reference_opt).to(self.device)
        self.cri_cp = build_loss(train_opt["cp_opt"]).to(self.device) if train_opt.get("cp_opt") else None
        self.cri_text_gestalt = build_loss(train_opt["text_gestalt_opt"]).to(self.device) if train_opt.get("text_gestalt_opt") else None
        self.cri_grad = build_loss(train_opt["grad_opt"]).to(self.device) if train_opt.get("grad_opt") else None
        uv_opt = train_opt.get("uv_strong_opt")
        self.cri_uv_strong = build_loss(uv_opt).to(self.device) if uv_opt else None
        self.uv_strong_weight = float(train_opt.get("uv_strong_weight", 0.0))
        if self.uv_strong_weight < 0.0:
            raise ValueError("uv_strong_weight must be non-negative")
        bg_lf_opt = train_opt.get("bg_lf_opt")
        if bg_lf_opt is None and float(train_opt.get("bg_lf_weight", 0.0)) > 0.0:
            bg_lf_opt = {"type": "BackgroundLowFrequencyLoss", "radius": 7, "sigma": 4.0, "channel": "luma"}
        self.cri_bg_lf = build_loss(bg_lf_opt).to(self.device) if bg_lf_opt else None
        # Radially weighted FFT amplitude term (ZoomSR RadialFreqLoss methodology): ``fft_opt`` builds the loss,
        # ``fft_weight`` is ramped by _schedule like the other auxiliary terms.
        fft_opt = train_opt.get("fft_opt")
        self.cri_fft = build_loss(fft_opt).to(self.device) if fft_opt else None
        self.fft_weight = float(train_opt.get("fft_weight", 0.0))
        if self.fft_weight < 0.0:
            raise ValueError("fft_weight must be non-negative")
        if self.fft_weight > 0.0 and self.cri_fft is None:
            raise ValueError("fft_weight is positive but fft_opt did not build a loss")
        # Keep the selected formulation visible in scalar traces.  The guard
        # variant is deliberately a separate registered loss so old BG-LF
        # runs retain their historical blur-before-mask behavior.
        self.bg_lf_guard_active = bool(getattr(self.cri_bg_lf, "variant", "") == "guard")
        skel_fbeta_opt = train_opt.get("skel_fbeta_opt")
        self.cri_skel_fbeta = build_loss(skel_fbeta_opt).to(self.device) if skel_fbeta_opt else None
        self.skel_fbeta_weight = float(train_opt.get("skel_fbeta_weight", 0.0))
        if self.skel_fbeta_weight < 0.0:
            raise ValueError("skel_fbeta_weight must be non-negative")
        # Stroke geometry losses are training-only and consume the existing
        # decoded RGB output plus detached GT structure masks.  No auxiliary
        # foreground head is added to the generator.
        cldice_opt = train_opt.get("cldice_opt", train_opt.get("stroke_cldice_opt"))
        self.cri_cldice = build_loss(cldice_opt).to(self.device) if cldice_opt else None
        self.cldice_weight = float(
            train_opt.get("cldice_weight", train_opt.get("stroke_cldice_weight", 0.0))
        )
        if self.cldice_weight < 0.0:
            raise ValueError("cldice_weight must be non-negative")
        # Both terms supervise the same GT skeleton recall/precision problem.
        # clDice is an alternative to, not an additive companion of, the
        # historical ThinSkeletonFbeta term; stacking them silently doubles a
        # topology objective and invalidates the planned ablation.
        if self.skel_fbeta_weight > 0.0 and self.cldice_weight > 0.0:
            raise ValueError(
                "Set either skel_fbeta_weight or cldice_weight to zero; "
                "SoftCLDiceLoss replaces ThinSkeletonFbetaLoss in this experiment."
            )
        width_opt = train_opt.get("stroke_width_opt", train_opt.get("width_opt"))
        self.cri_stroke_width = build_loss(width_opt).to(self.device) if width_opt else None
        self.stroke_width_weight = float(
            train_opt.get("stroke_width_weight", train_opt.get("width_weight", 0.0))
        )
        if self.stroke_width_weight < 0.0:
            raise ValueError("stroke_width_weight must be non-negative")
        self.stroke_prob_enabled = bool(train_opt.get("stroke_prob_enabled", False))
        prob_opt = train_opt.get("stroke_prob_opt")
        self.cri_stroke_prob = build_loss(prob_opt).to(self.device) if prob_opt else None
        self.stroke_prob_weight = float(train_opt.get("stroke_prob_weight", 0.0))
        if self.stroke_prob_weight < 0.0:
            raise ValueError("stroke_prob_weight must be non-negative")
        if self.stroke_prob_enabled and (self.cri_stroke_prob is None or self.stroke_prob_weight <= 0.0):
            raise ValueError("stroke_prob_enabled requires positive stroke_prob_weight and stroke_prob_opt")
        keep_opt = train_opt.get("keep_opt", train_opt.get("keep_loss_opt"))
        self.cri_keep = build_loss(keep_opt).to(self.device) if keep_opt else None
        self.keep_weight = float(train_opt.get("keep_weight", 0.0))
        if self.keep_weight < 0.0:
            raise ValueError("keep_weight must be non-negative")
        sdf_boundary_opt = train_opt.get("sdf_boundary_opt")
        self.cri_sdf_boundary = build_loss(sdf_boundary_opt).to(self.device) if sdf_boundary_opt else None
        self.sdf_boundary_weight = float(train_opt.get("sdf_boundary_weight", 0.0))
        if self.sdf_boundary_weight < 0.0:
            raise ValueError("sdf_boundary_weight must be non-negative")
        normal_profile_opt = train_opt.get("normal_profile_opt")
        self.cri_normal_profile = build_loss(normal_profile_opt).to(self.device) if normal_profile_opt else None
        self.normal_profile_weight = float(train_opt.get("normal_profile_weight", 0.0))
        if self.normal_profile_weight < 0.0:
            raise ValueError("normal_profile_weight must be non-negative")
        stroke_core_opt = train_opt.get("stroke_core_opt", train_opt.get("core_opt"))
        self.cri_stroke_core = build_loss(stroke_core_opt).to(self.device) if stroke_core_opt else None
        self.stroke_core_weight = float(
            train_opt.get("stroke_core_weight", train_opt.get("core_weight", 0.0))
        )
        if self.stroke_core_weight < 0.0:
            raise ValueError("stroke_core_weight must be non-negative")
        stroke_overshoot_opt = train_opt.get("stroke_overshoot_opt", train_opt.get("overshoot_opt"))
        self.cri_stroke_overshoot = build_loss(stroke_overshoot_opt).to(self.device) if stroke_overshoot_opt else None
        self.stroke_overshoot_weight = float(
            train_opt.get("stroke_overshoot_weight", train_opt.get("overshoot_weight", 0.0))
        )
        if self.stroke_overshoot_weight < 0.0:
            raise ValueError("stroke_overshoot_weight must be non-negative")
        self.loss_mode = str(train_opt.get("loss_mode", "native")).strip().lower()
        if self.loss_mode not in {"native", "reference"}:
            raise ValueError("train.loss_mode must be native or reference")
        self.train_reference_loss, self.validation_reference_loss = resolve_reference_loss_flags(train_opt)
        # This alias makes the state explicit in debug traces and supports
        # callers that inspect the historical single-switch name.
        self.reference_loss_enabled = self.train_reference_loss
        if self.loss_mode == "reference" and not self.train_reference_loss:
            raise ValueError("reference loss cannot be selected when train_reference_loss is false")
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        """@luna-comment
        Input: matching BCHW six-channel Y4UV, optional RGB/masks and segment metadata.
        Output: validated model batch on its device, including full-resolution masks.
        Key Steps: move tensors, check shapes, recover missing RGB masks, retain metadata.
        Key Params: optional original_lq and mask keys. State/Dependencies: device and pack helpers.
        """
        self.lq = data["lq"].to(self.device)
        self.original_lq = data.get("original_lq")
        if self.original_lq is not None:
            self.original_lq = self.original_lq.to(self.device)
        self.gt = data["gt"].to(self.device)
        if self.lq.ndim != 4 or self.lq.shape[1] != 6 or self.gt.shape != self.lq.shape:
            raise ValueError("Y4UV training batches require matching [B,6,H/2,W/2] lq and gt tensors")
        self.valid_mask = data.get("valid_mask")
        if self.valid_mask is not None:
            self.valid_mask = self.valid_mask.to(self.device)
        self.text_mask = data.get("text_mask")
        if self.text_mask is not None:
            self.text_mask = self.text_mask.to(self.device)
        self.gt_rgb = data.get("gt_rgb")
        if self.gt_rgb is not None:
            self.gt_rgb = self.gt_rgb.to(self.device)
        self.lq_rgb = data.get("lq_rgb")
        if self.lq_rgb is not None:
            self.lq_rgb = self.lq_rgb.to(self.device)
        self.valid_mask_rgb = data.get("valid_mask_rgb")
        if self.valid_mask_rgb is not None:
            self.valid_mask_rgb = self.valid_mask_rgb.to(self.device)
        elif self.gt_rgb is not None:
            self.valid_mask_rgb = self._full_resolution_mask(self.valid_mask, self.gt_rgb, "valid_mask")
        self.text_mask_rgb = data.get("text_mask_rgb")
        if self.text_mask_rgb is not None:
            self.text_mask_rgb = self.text_mask_rgb.to(self.device)
        elif self.gt_rgb is not None:
            self.text_mask_rgb = self._full_resolution_mask(self.text_mask, self.gt_rgb, "text_mask")
        self.structure_available = list(data.get("structure_available", []))
        self.segment_metadata = data

    def _forward_v5(self, input_y4uv, *, network=None, **kwargs):
        """@luna-comment
        Input: packed input, optional generator and forward keyword arguments.
        Output: generator prediction with the configured residual input route.
        Key Steps: detect external skip support; validate original input if requested.
        Key Params: residual_input_source selects skip. State/Dependencies: net_g and original_lq.
        """
        generator = self.net_g if network is None else network
        if not getattr(generator, "supports_external_skip_input", False):
            return generator(input_y4uv, **kwargs)
        skip = getattr(self, "original_lq", None)
        config = self.opt.get("lq_preprocess") or {}
        source = str(config.get("residual_input_source", "preprocessed")).strip().lower()
        if source == "original" and bool(config.get("enabled", False)) and skip is None:
            raise RuntimeError("original residual input requested but original_lq is unavailable")
        if source == "original" and bool(config.get("enabled", False)):
            return generator(skip, skip_input_y4uv=input_y4uv, **kwargs)
        return generator(input_y4uv, **kwargs)

    @torch.no_grad()
    def test(self):
        """@luna-comment
        Input: prepared model batch. Output: no-gradient prediction stored in output.
        Key Steps: prefer EMA generator; evaluate through the common skip route.
        Key Params: EMA presence selects generator. State/Dependencies: net_g, net_g_ema and lq.
        """
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            self.output = self._forward_v5(self.lq, network=self.net_g_ema)
        else:
            self.net_g.eval()
            self.output = self._forward_v5(self.lq)
            self.net_g.train()

    @staticmethod
    def _full_resolution_mask(mask, reference: torch.Tensor, name: str) -> torch.Tensor:
        """@luna-comment
        Input: optional native or packed mask and BCHW RGB reference. Output: aligned single-channel mask.
        Key Steps: synthesize missing coverage, retain native masks, decode/resize legacy packed coverage.
        Key Params: only exact full/half geometry accepted. State/Dependencies: torch pack helpers; no mutation.
        """
        if mask is None:
            return torch.ones_like(reference[:, :1])
        value = mask.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 3:
            value = value.unsqueeze(1)
        if value.ndim != 4 or value.shape[0] != reference.shape[0]:
            raise ValueError(f"{name} must be BCHW or BHW, got {tuple(value.shape)}")
        if value.shape[1] == 1:
            if value.shape[-2:] == reference.shape[-2:]:
                return value
            if value.shape[-2] * 2 == reference.shape[-2] and value.shape[-1] * 2 == reference.shape[-1]:
                return F.interpolate(value, size=reference.shape[-2:], mode="nearest")
        elif value.shape[1] == 6:
            decoded = y4uv_mask_to_yuv444(value, full_hw=tuple(reference.shape[-2:]))
            return decoded.mean(dim=1, keepdim=True)
        raise ValueError(f"{name} does not match full-resolution RGB target {tuple(reference.shape)}")

    def _background_masks(self):
        """@luna-comment
        Input: stored HR RGB targets and masks. Output: text/valid masks for background supervision.
        Key Steps: validate availability length; treat unavailable structure as no text for this guard.
        Key Params: availability is per sample. State/Dependencies: returns original or cloned text mask.
        """
        if self.gt_rgb is None or self.valid_mask_rgb is None or self.text_mask_rgb is None:
            raise RuntimeError("bgLF requires full-resolution gt_rgb, valid_mask_rgb, and text_mask_rgb")
        valid_rgb = self.valid_mask_rgb
        text_rgb = self.text_mask_rgb
        availability = getattr(self, "structure_available", [])
        if availability:
            if len(availability) != int(self.gt_rgb.shape[0]):
                raise ValueError("structure_available must have one entry per batch sample")
            # Remote58 represents missing structure as valid-area text for the
            # legacy native loss.  For bgLF, unknown structure is conservatively
            # treated as no text so that the sample still contributes valid far
            # background; no generated/inference tensor is used here.
            unavailable = [index for index, value in enumerate(availability) if not bool(value)]
            if unavailable:
                text_rgb = text_rgb.clone()
                text_rgb[unavailable] = 0.0
        return text_rgb, valid_rgb

    def _structure_masks(self):
        """@luna-comment
        Input: stored HR mask fields/availability. Output: verified text and valid masks.
        Key Steps: reject absent or explicitly unverified structure; return HR tensors unchanged.
        Key Params: every declared sample must be verified. State/Dependencies: current model batch only.
        """
        if not hasattr(self, "text_mask_rgb") or not hasattr(self, "valid_mask_rgb"):
            raise RuntimeError("structure auxiliary loss requires full-resolution text_mask_rgb and valid_mask_rgb")
        availability = getattr(self, "structure_available", [])
        if availability and not all(bool(value) for value in availability):
            raise RuntimeError("structure auxiliary loss received a sample without a verified structure mask")
        return self.text_mask_rgb, self.valid_mask_rgb

    @staticmethod
    def _struct_hf(pred_rgb, gt_rgb, text_mask, valid_mask, opt):
        """@luna-comment
        Input: aligned HR predictions/targets, masks and region weights. Output: weighted luma-gradient error.
        Key Steps: build inner/ring support, derive luma, compare orthogonal Scharr responses and normalize.
        Key Params: outer_ring_radius and text/boundary/background weights. State/Dependencies: torch ops only.
        """
        text = (text_mask > 0.5).float() * (valid_mask > 0.5).float()
        valid = (valid_mask > 0.5).float()
        radius = int(opt.get("outer_ring_radius", 3))
        ring = (F.max_pool2d(text, 2 * radius + 1, 1, radius) > 0.5).float() * (1.0 - text) * valid
        inner = (F.max_pool2d(1.0 - text, 3, 1, 1) < 0.5).float() * text
        boundary = text * (1.0 - inner)
        weight = valid * float(opt.get("background_weight", 0.25))
        weight = weight + text * (float(opt.get("stroke_weight", 1.0)) - float(opt.get("background_weight", 0.25)))
        weight = weight + boundary * (float(opt.get("boundary_weight", 5.0)) - float(opt.get("stroke_weight", 1.0)))
        weight = weight + ring * (float(opt.get("outer_ring_weight", 6.0)) - float(opt.get("background_weight", 0.25)))
        luma = lambda value: 0.299 * value[:, :1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]
        kx = pred_rgb.new_tensor(((-3., 0., 3.), (-10., 0., 10.), (-3., 0., 3.))).view(1, 1, 3, 3) / 16.
        dy = F.conv2d(luma(pred_rgb), kx, padding=1) - F.conv2d(luma(gt_rgb), kx, padding=1)
        dx = F.conv2d(luma(pred_rgb), kx.transpose(-1, -2), padding=1) - F.conv2d(luma(gt_rgb), kx.transpose(-1, -2), padding=1)
        return (torch.sqrt(dx.square() + dy.square() + 1e-6) * weight).sum() / weight.sum().clamp_min(1e-6)

    @staticmethod
    def _metadata_list(metadata: Mapping[str, Any], key: str, batch_size: int, default: Any) -> list[Any]:
        """@luna-comment
        Input: collated metadata, key, batch size and fallback value.
        Output: exactly batch_size values, or ValueError for incompatible metadata.
        Key Steps: fill absent values; validate lists; accept scalar only for singleton.
        Key Params: default controls absent metadata. State/Dependencies: none.
        """
        values = metadata.get(key)
        if values is None:
            return [default for _ in range(batch_size)]
        if isinstance(values, (list, tuple)):
            if len(values) != batch_size:
                raise ValueError(f"metadata[{key!r}] length {len(values)} != batch size {batch_size}")
            return list(values)
        if batch_size == 1:
            return [values]
        raise ValueError(f"metadata[{key!r}] must be a batch-length list for batch size {batch_size}")

    @classmethod
    def _build_cp_annotations(cls, metadata: Mapping[str, Any], batch_size: int) -> list[list[list[Any]]]:
        """@luna-comment
        Input: per-sample eligibility, polygons, validity, truncation and transcriptions.
        Output: per-sample CRNN quadrilateral records; invalid/ineligible entries are empty.
        Key Steps: normalize fields, filter unsuitable crops, convert accepted polygons.
        Key Params: cp_eligible/is_crop_truncated gate recognition. State/Dependencies: polygon_to_quad.
        """
        eligible = cls._metadata_list(metadata, "cp_eligible", batch_size, False)
        polygons = cls._metadata_list(metadata, "polygon_local", batch_size, [])
        polygon_valid = cls._metadata_list(metadata, "polygon_valid", batch_size, True)
        truncated = cls._metadata_list(metadata, "is_crop_truncated", batch_size, False)
        transcriptions = cls._metadata_list(metadata, "transcription", batch_size, "")
        output: list[list[list[Any]]] = []
        for is_eligible, polygon, is_valid, is_truncated, transcription in zip(
            eligible, polygons, polygon_valid, truncated, transcriptions
        ):
            if not bool(is_eligible) or not bool(is_valid) or bool(is_truncated):
                output.append([])
                continue
            try:
                quad = polygon_to_quad(polygon)
            except (TypeError, ValueError):
                output.append([])
                continue
            output.append([[float(value) for value in quad.reshape(-1).tolist()] + [str(transcription)]])
        return output

    @staticmethod
    def _append_cp_stats(logs: OrderedDict, reference: torch.Tensor, stats: Mapping[str, Any]) -> None:
        """@luna-comment
        Input: mutable step logs, reference tensor and CP statistics. Output: appended scalar CP diagnostics.
        Key Steps: copy supported counters and ratios to the reference device/dtype.
        Key Params: fixed supported-stat list. State/Dependencies: mutates log dictionary, not model weights.
        """
        for name in (
            "eligible_lines",
            "accepted_lines",
            "skipped_lines",
            "vertical_skipped",
            "overshoot_ratio",
            "clamp_ratio",
            "width_p50",
            "width_p95",
            "width_max",
        ):
            if name in stats:
                logs[f"cp_{name}"] = reference.new_tensor(float(stats[name]))
        for name, value in stats.get("layer_losses", {}).items():
            logs[f"cp_layer_{name}"] = reference.new_tensor(float(value))

    def _schedule(self, target: float, current_iter: int) -> float:
        """@luna-comment Return the configured deterministic auxiliary-loss ramp value."""
        train_opt = self.opt["train"]
        start = int(train_opt.get("aux_start_iter", 0))
        ramp = int(train_opt.get("aux_ramp_iters", 0))
        if current_iter < start:
            return 0.0
        if ramp <= 0:
            return float(target)
        return float(target) * min(1.0, (current_iter - start + 1) / float(ramp))

    def optimize_parameters(self, current_iter):
        """@shanlan-comment
        Input: one device-resident six-channel Y4UV batch and aligned masks/RGB CP targets.
        Output: updated generator parameters and scalar loss diagnostics in ``log_dict``.
        Key Steps: always compute native loss, optionally compute decoded-reference
        loss, add ramped CP/gradient/Recall-aware/bgLF terms, and finally add
        the verified-structure-gated SkelFbeta term.
        Key Params: ``train_reference_loss`` defaults true for legacy configs; native mode remains selected for cache runs.
        State/Dependencies: mutates generator/optimizer state and may decode YUV444 for CP or enabled reference loss.
        """
        self.optimizer_g.zero_grad(set_to_none=True)
        if self.stroke_prob_enabled:
            network_result = self._forward_v5(self.lq, return_stroke_logits=True)
            if not isinstance(network_result, tuple) or len(network_result) != 2:
                raise RuntimeError("stroke_prob_enabled network must return (y4uv_output, full-resolution logits)")
            self.output, self.stroke_logits = network_result
        else:
            self.output = self._forward_v5(self.lq)
            self.stroke_logits = None
        native_loss = self.cri_y4uv_native(
            self.output,
            self.gt,
            valid_mask=self.valid_mask,
            text_mask=self.text_mask,
        )
        if self.train_reference_loss:
            reference_loss = self.cri_yuv444_reference(
                self.output,
                self.gt,
                valid_mask=self.valid_mask,
                text_mask=self.text_mask,
            )
            compare_abs = (native_loss - reference_loss).abs()
        else:
            reference_loss = self.output.new_zeros(())
            compare_abs = self.output.new_zeros(())
        comparison = {
            "l_native": native_loss,
            "l_reference": reference_loss,
            "l_compare_abs": compare_abs,
        }
        selected = comparison["l_native"] if self.loss_mode == "native" else comparison["l_reference"]
        total = selected
        logs = OrderedDict(
            l_native=comparison["l_native"].detach(),
            l_reference=comparison["l_reference"].detach(),
            l_compare_abs=comparison["l_compare_abs"].detach(),
            l_selected=selected.detach(),
            loss_mode=self.output.new_tensor(0.0 if self.loss_mode == "native" else 1.0),
            reference_loss_enabled=self.output.new_tensor(float(self.train_reference_loss)),
            haar_switch=self.output.new_tensor(float(bool(getattr(self.net_g, "haar_switch", False)))),
        )

        train_opt = self.opt["train"]
        cp_weight = self._schedule(float(train_opt.get("cp_weight", 0.0)), current_iter)
        grad_weight = self._schedule(float(train_opt.get("grad_weight", 0.0)), current_iter)
        bg_lf_weight = self._schedule(float(train_opt.get("bg_lf_weight", 0.0)), current_iter)
        skel_fbeta_weight = self._schedule(self.skel_fbeta_weight, current_iter)
        logs["cp_scale"] = self.output.new_tensor(cp_weight)
        logs["grad_scale"] = self.output.new_tensor(grad_weight)
        logs["bg_lf_scale"] = self.output.new_tensor(bg_lf_weight)
        logs["bg_lf_guard_active"] = self.output.new_tensor(float(self.bg_lf_guard_active))
        logs["skel_fbeta_scale"] = self.output.new_tensor(skel_fbeta_weight)
        cldice_weight = self._schedule(self.cldice_weight, current_iter)
        stroke_width_weight = self._schedule(self.stroke_width_weight, current_iter)
        stroke_prob_weight = self._schedule(self.stroke_prob_weight, current_iter)
        logs["cldice_scale"] = self.output.new_tensor(cldice_weight)
        logs["stroke_width_scale"] = self.output.new_tensor(stroke_width_weight)
        logs["stroke_prob_scale"] = self.output.new_tensor(stroke_prob_weight)
        keep_weight = self._schedule(self.keep_weight, current_iter)
        logs["keep_scale"] = self.output.new_tensor(keep_weight)
        uv_strong_scale = self._schedule(self.uv_strong_weight, current_iter)
        logs["uv_strong_scale"] = self.output.new_tensor(uv_strong_scale)

        output444 = None
        output_rgb = None
        stroke_probability = None
        if self.stroke_prob_enabled and stroke_prob_weight > 0.0:
            text_rgb, valid_rgb = self._structure_masks()
            raw_prob, prob_details = self.cri_stroke_prob(
                self.stroke_logits, text_rgb, valid_rgb, return_details=True
            )
            total = total + stroke_prob_weight * raw_prob
            stroke_probability = self.stroke_logits.sigmoid()
            logs["l_stroke_prob_raw"] = raw_prob.detach()
            logs["l_stroke_prob"] = (stroke_prob_weight * raw_prob).detach()
            logs["stroke_prob_iou"] = prob_details["iou"].mean().detach()
            logs["stroke_prob_recall"] = prob_details["recall"].mean().detach()
            logs["stroke_prob_precision"] = prob_details["precision"].mean().detach()
        if self.cri_cp is not None and cp_weight > 0.0:
            if self.gt_rgb is None:
                raise RuntimeError("CP loss is enabled but the batch has no gt_rgb target")
            output444 = y4uv_to_yuv444(self.output)
            output_rgb = YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            if bool(getattr(self.cri_cp, "expects_segment_metadata", False)):
                cp_raw = self.cri_cp(output_rgb, self.gt_rgb, self.segment_metadata)
            else:
                lines = self._build_cp_annotations(self.segment_metadata, self.output.shape[0])
                cp_raw = self.cri_cp(output_rgb, self.gt_rgb, lines)
            total = total + cp_weight * cp_raw
            logs["l_cp_raw"] = cp_raw.detach()
            logs["l_cp"] = (cp_weight * cp_raw).detach()
            self._append_cp_stats(logs, self.output, getattr(self.cri_cp, "last_stats", {}))
        elif self.cri_cp is not None:
            logs["l_cp_raw"] = self.output.sum() * 0.0
            logs["l_cp"] = self.output.sum() * 0.0

        if uv_strong_scale > 0.0 and self.cri_uv_strong is None:
            raise RuntimeError("uv_strong_weight is positive but uv_strong_opt did not build a loss")
        if self.cri_uv_strong is not None and uv_strong_scale > 0.0:
            raw_uv, uv_details = self.cri_uv_strong(self.output, self.gt, self.valid_mask, self.text_mask, return_details=True)
            total = total + uv_strong_scale * raw_uv
            logs["l_uv_strong_raw"] = raw_uv.detach()
            logs["l_uv_strong"] = (uv_strong_scale * raw_uv).detach()
            for name, value in uv_details.items():
                logs["uv_" + name] = value

        if bg_lf_weight > 0.0 and self.cri_bg_lf is None:
            raise RuntimeError("bg_lf_weight is positive but bg_lf_opt did not build a loss")
        if self.cri_bg_lf is not None and bg_lf_weight > 0.0:
            text_rgb, valid_rgb = self._background_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_bg_lf, bg_lf_details = self.cri_bg_lf(
                output_rgb,
                self.gt_rgb,
                text_mask_rgb=text_rgb,
                valid_mask_rgb=valid_rgb,
                return_details=True,
            )
            total = total + bg_lf_weight * raw_bg_lf
            logs["l_bg_lf_raw"] = raw_bg_lf.detach()
            logs["l_bg_lf_weighted"] = (bg_lf_weight * raw_bg_lf).detach()
            logs["l_bg_lf"] = (bg_lf_weight * raw_bg_lf).detach()
            logs["bg_lf_far_coverage"] = bg_lf_details["far_coverage"].detach()
            logs["bg_lf_far_over_valid"] = bg_lf_details["far_over_valid"].detach()
            logs["bg_lf_guard_normalizer_min"] = bg_lf_details.get(
                "guard_normalizer_min", self.output.new_zeros(())
            ).detach()
            logs["bg_lf_guard_normalizer_mean"] = bg_lf_details.get(
                "guard_normalizer_mean", self.output.new_zeros(())
            ).detach()
        elif self.cri_bg_lf is not None:
            zero = self.output.sum() * 0.0
            logs["l_bg_lf_raw"] = zero
            logs["l_bg_lf_weighted"] = zero
            logs["l_bg_lf"] = zero
            logs["bg_lf_far_coverage"] = zero
            logs["bg_lf_far_over_valid"] = zero
            logs["bg_lf_guard_normalizer_min"] = zero
            logs["bg_lf_guard_normalizer_mean"] = zero

        structure_scale = self._schedule(float(train_opt.get("struct_hf_weight", 0.0)), current_iter)
        if structure_scale > 0.0:
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_struct = self._struct_hf(output_rgb, self.gt_rgb, text_rgb, valid_rgb, train_opt.get("struct_hf_opt", {}))
            total = total + structure_scale * raw_struct
            logs["l_struct_hf_raw"] = raw_struct.detach()
            logs["l_struct_hf"] = (structure_scale * raw_struct).detach()

        missing_scale = self._schedule(float(train_opt.get("recall_aware_weight_missing", 0.0)), current_iter)
        false_scale = self._schedule(float(train_opt.get("recall_aware_weight_false", 0.0)), current_iter)
        if missing_scale > 0.0 or false_scale > 0.0:
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            missing, false, details = recall_aware_structure_losses(output_rgb, self.gt_rgb, text_rgb, valid_rgb, **train_opt.get("recall_aware_opt", {}))
            total = total + missing_scale * missing + false_scale * false
            logs["l_missing_raw"], logs["l_false_raw"] = missing.detach(), false.detach()
            logs["l_missing"], logs["l_false"] = (missing_scale * missing).detach(), (false_scale * false).detach()
            logs["recall_active_ratio"] = details["active"].float().mean().detach()

        fft_scale = self._schedule(self.fft_weight, current_iter)
        if self.cri_fft is not None:
            logs["fft_scale"] = self.output.new_tensor(fft_scale)
        if self.cri_fft is not None and fft_scale > 0.0:
            if self.gt_rgb is None:
                raise RuntimeError("fft loss requires full-resolution gt_rgb")
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_fft, fft_details = self.cri_fft(output_rgb, self.gt_rgb, valid_mask=self.valid_mask_rgb, return_details=True)
            total = total + fft_scale * raw_fft
            logs["l_fft_raw"] = raw_fft.detach()
            logs["l_fft"] = (fft_scale * raw_fft).detach()
            logs["fft_hf_ratio_pred"] = fft_details["hf_ratio_pred"].detach()
            logs["fft_hf_ratio_gt"] = fft_details["hf_ratio_gt"].detach()
        elif self.cri_fft is not None:
            zero = self.output.sum() * 0.0
            logs["l_fft_raw"] = zero
            logs["l_fft"] = zero

        if skel_fbeta_weight > 0.0 and self.cri_skel_fbeta is None:
            raise RuntimeError("skel_fbeta_weight is positive but skel_fbeta_opt did not build a loss")
        if self.cri_skel_fbeta is not None and skel_fbeta_weight > 0.0:
            if self.gt_rgb is None or self.text_mask_rgb is None or self.valid_mask_rgb is None:
                raise RuntimeError("SkelFbeta requires full-resolution gt_rgb and verified RGB masks")
            text_rgb, valid_rgb = self.text_mask_rgb, self.valid_mask_rgb
            availability = getattr(self, "structure_available", [])
            if availability:
                if len(availability) != int(self.gt_rgb.shape[0]):
                    raise ValueError("structure_available must have one entry per batch sample")
                unavailable = [index for index, value in enumerate(availability) if not bool(value)]
                if unavailable:
                    # Unknown structure rows are fully gated for this loss;
                    # no fabricated skeleton is allowed to affect gradients.
                    text_rgb = text_rgb.clone()
                    valid_rgb = valid_rgb.clone()
                    text_rgb[unavailable] = 0.0
                    valid_rgb[unavailable] = 0.0
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_skel_fbeta, skel_details = self.cri_skel_fbeta(
                output_rgb,
                self.gt_rgb,
                text_mask_rgb=text_rgb,
                valid_mask_rgb=valid_rgb,
                return_details=True,
            )
            total = total + skel_fbeta_weight * raw_skel_fbeta
            logs["l_skel_fbeta_raw"] = raw_skel_fbeta.detach()
            logs["l_skel_fbeta_weighted"] = (skel_fbeta_weight * raw_skel_fbeta).detach()
            logs["l_skel_fbeta"] = (skel_fbeta_weight * raw_skel_fbeta).detach()
            logs["skel_fbeta_recall"] = skel_details["skeleton_recall"].mean().detach()
            logs["skel_fbeta_precision"] = skel_details["occupancy_precision"].mean().detach()
            logs["skel_fbeta_fbeta"] = skel_details["fbeta"].mean().detach()
            logs["skel_fbeta_active_ratio"] = skel_details["active"].float().mean().detach()
            logs["skel_fbeta_skeleton_pixels"] = skel_details["skeleton_pixels"].mean().detach()
            logs["skel_fbeta_tube_pixels"] = skel_details["tube_pixels"].mean().detach()
        elif self.cri_skel_fbeta is not None:
            zero = self.output.sum() * 0.0
            logs["l_skel_fbeta_raw"] = zero
            logs["l_skel_fbeta_weighted"] = zero
            logs["l_skel_fbeta"] = zero
            logs["skel_fbeta_recall"] = zero
            logs["skel_fbeta_precision"] = zero
            logs["skel_fbeta_fbeta"] = zero
            logs["skel_fbeta_active_ratio"] = zero
            logs["skel_fbeta_skeleton_pixels"] = zero
            logs["skel_fbeta_tube_pixels"] = zero

        # L_clDice and L_width share the same verified structure-mask contract
        # as SkelFbeta.  Unknown rows are fully zeroed; no inferred mask is
        # allowed to create topology or width gradients.
        geometry_enabled = cldice_weight > 0.0 or stroke_width_weight > 0.0
        if geometry_enabled:
            if self.gt_rgb is None or self.text_mask_rgb is None or self.valid_mask_rgb is None:
                raise RuntimeError("stroke geometry losses require full-resolution gt_rgb and verified RGB masks")
            text_rgb, valid_rgb = self.text_mask_rgb, self.valid_mask_rgb
            availability = getattr(self, "structure_available", [])
            if availability:
                if len(availability) != int(self.gt_rgb.shape[0]):
                    raise ValueError("structure_available must have one entry per batch sample")
                unavailable = [index for index, value in enumerate(availability) if not bool(value)]
                if unavailable:
                    text_rgb = text_rgb.clone()
                    valid_rgb = valid_rgb.clone()
                    text_rgb[unavailable] = 0.0
                    valid_rgb[unavailable] = 0.0
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)

        if cldice_weight > 0.0 and self.cri_cldice is None:
            raise RuntimeError("cldice_weight is positive but cldice_opt did not build a loss")
        if self.cri_cldice is not None and cldice_weight > 0.0:
            raw_cldice, cldice_details = self.cri_cldice(
                output_rgb,
                self.gt_rgb,
                text_mask_rgb=text_rgb,
                valid_mask_rgb=valid_rgb,
                pred_probability=stroke_probability,
                return_details=True,
            )
            total = total + cldice_weight * raw_cldice
            logs["l_cldice_raw"] = raw_cldice.detach()
            logs["l_cldice_weighted"] = (cldice_weight * raw_cldice).detach()
            logs["l_cldice"] = (cldice_weight * raw_cldice).detach()
            logs["cldice_gt_coverage"] = cldice_details["gt_coverage"].mean().detach()
            logs["cldice_pred_precision"] = cldice_details["pred_precision"].mean().detach()
            logs["cldice_score"] = cldice_details["cldice_score"].mean().detach()
            logs["cldice_active_ratio"] = cldice_details["active"].float().mean().detach()
            logs["cldice_gt_skeleton_pixels"] = cldice_details["gt_skeleton_pixels"].mean().detach()
            logs["cldice_pred_skeleton_mass"] = cldice_details["pred_skeleton_mass"].mean().detach()
        elif self.cri_cldice is not None:
            zero = self.output.sum() * 0.0
            logs["l_cldice_raw"] = zero
            logs["l_cldice_weighted"] = zero
            logs["l_cldice"] = zero
            logs["cldice_gt_coverage"] = zero
            logs["cldice_pred_precision"] = zero
            logs["cldice_score"] = zero
            logs["cldice_active_ratio"] = zero
            logs["cldice_gt_skeleton_pixels"] = zero
            logs["cldice_pred_skeleton_mass"] = zero

        if stroke_width_weight > 0.0 and self.cri_stroke_width is None:
            raise RuntimeError("stroke_width_weight is positive but stroke_width_opt did not build a loss")
        if self.cri_stroke_width is not None and stroke_width_weight > 0.0:
            raw_width, width_details = self.cri_stroke_width(
                output_rgb,
                self.gt_rgb,
                text_mask_rgb=text_rgb,
                valid_mask_rgb=valid_rgb,
                pred_probability=stroke_probability,
                return_details=True,
            )
            total = total + stroke_width_weight * raw_width
            logs["l_stroke_width_raw"] = raw_width.detach()
            logs["l_stroke_width_weighted"] = (stroke_width_weight * raw_width).detach()
            logs["l_stroke_width"] = (stroke_width_weight * raw_width).detach()
            logs["stroke_width_abs_error"] = width_details["width_abs_error"].mean().detach()
            logs["stroke_width_active_ratio"] = width_details["active"].float().mean().detach()
            logs["stroke_width_gt_skeleton_pixels"] = width_details["gt_skeleton_pixels"].mean().detach()
            logs["stroke_width_predicted_mean"] = width_details["predicted_width_at_gt_skeleton"].mean().detach()
            logs["stroke_width_gt_mean"] = width_details["gt_width_at_gt_skeleton"].mean().detach()
        elif self.cri_stroke_width is not None:
            zero = self.output.sum() * 0.0
            logs["l_stroke_width_raw"] = zero
            logs["l_stroke_width_weighted"] = zero
            logs["l_stroke_width"] = zero
            logs["stroke_width_abs_error"] = zero
            logs["stroke_width_active_ratio"] = zero
            logs["stroke_width_gt_skeleton_pixels"] = zero
            logs["stroke_width_predicted_mean"] = zero
            logs["stroke_width_gt_mean"] = zero

        if keep_weight > 0.0 and self.cri_keep is None:
            raise RuntimeError("keep_weight is positive but keep_opt did not build a loss")
        if self.cri_keep is not None and keep_weight > 0.0:
            if self.gt_rgb is None or self.lq_rgb is None:
                raise RuntimeError("Keep loss requires full-resolution gt_rgb and lq_rgb")
            if self.text_mask_rgb is None or self.valid_mask_rgb is None:
                raise RuntimeError("Keep loss requires full-resolution verified RGB masks")
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_keep, keep_details = self.cri_keep(
                output_rgb,
                self.gt_rgb,
                self.lq_rgb,
                text_mask_rgb=self.text_mask_rgb,
                valid_mask_rgb=self.valid_mask_rgb,
                structure_available=getattr(self, "structure_available", []),
                return_details=True,
            )
            total = total + keep_weight * raw_keep
            logs["l_keep_raw"] = raw_keep.detach()
            logs["l_keep_weighted"] = (keep_weight * raw_keep).detach()
            logs["l_keep"] = (keep_weight * raw_keep).detach()
            for name in (
                "keep_support_count",
                "keep_support_coverage",
                "keep_gt_skeleton_count",
                "keep_visible_skeleton_count",
                "keep_gt_support_count",
                "keep_visible_support_rate",
                "keep_polarity_consistency_rate",
                "keep_visible_to_lost_rate",
                "keep_thin_recall",
                "keep_mean_lq_contrast",
                "keep_mean_pred_contrast",
                "keep_false_positive_rate",
                "keep_spur_rate",
                "keep_background_activation_rate",
                "keep_valid_pixel_count",
                "keep_polarity_bright_fraction",
            ):
                if name in keep_details:
                    logs[name] = keep_details[name].detach()
        elif self.cri_keep is not None:
            zero = self.output.sum() * 0.0
            logs["l_keep_raw"] = zero
            logs["l_keep_weighted"] = zero
            logs["l_keep"] = zero

        sdf_boundary_scale = self._schedule(self.sdf_boundary_weight, current_iter)
        if sdf_boundary_scale > 0.0 and self.cri_sdf_boundary is None:
            raise RuntimeError("sdf_boundary_weight is positive but sdf_boundary_opt did not build a loss")
        if self.cri_sdf_boundary is not None and sdf_boundary_scale > 0.0:
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_sdf, sdf_details = self.cri_sdf_boundary(
                output_rgb, self.gt_rgb, text_rgb, valid_rgb, return_details=True
            )
            total = total + sdf_boundary_scale * raw_sdf
            logs["l_sdf_boundary_raw"] = raw_sdf.detach()
            logs["l_sdf_boundary"] = (sdf_boundary_scale * raw_sdf).detach()
            logs["sdf_boundary_active_pixels"] = sdf_details["active_pixels"].mean().detach()
            logs["sdf_boundary_active_ratio"] = sdf_details["active_band"].mean().detach()
        elif self.cri_sdf_boundary is not None:
            zero = self.output.sum() * 0.0
            logs["l_sdf_boundary_raw"] = zero
            logs["l_sdf_boundary"] = zero
            logs["sdf_boundary_active_pixels"] = zero
            logs["sdf_boundary_active_ratio"] = zero

        normal_profile_scale = self._schedule(self.normal_profile_weight, current_iter)
        if normal_profile_scale > 0.0 and self.cri_normal_profile is None:
            raise RuntimeError("normal_profile_weight is positive but normal_profile_opt did not build a loss")
        if self.cri_normal_profile is not None and normal_profile_scale > 0.0:
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_profile, profile_details = self.cri_normal_profile(
                output_rgb, self.gt_rgb, text_rgb, valid_rgb, return_details=True
            )
            total = total + normal_profile_scale * raw_profile
            logs["l_normal_profile_raw"] = raw_profile.detach()
            logs["l_normal_profile"] = (normal_profile_scale * raw_profile).detach()
            logs["normal_profile_boundary_weight"] = profile_details["boundary_weight_sum"].detach()
            logs["normal_profile_valid_fraction"] = profile_details["valid_profile_fraction"].detach()
        elif self.cri_normal_profile is not None:
            zero = self.output.sum() * 0.0
            logs["l_normal_profile_raw"] = zero
            logs["l_normal_profile"] = zero
            logs["normal_profile_boundary_weight"] = zero
            logs["normal_profile_valid_fraction"] = zero

        stroke_core_scale = self._schedule(self.stroke_core_weight, current_iter)
        stroke_overshoot_scale = self._schedule(self.stroke_overshoot_weight, current_iter)
        logs["stroke_core_scale"] = self.output.new_tensor(stroke_core_scale)
        logs["stroke_overshoot_scale"] = self.output.new_tensor(stroke_overshoot_scale)
        if stroke_core_scale > 0.0 and self.cri_stroke_core is None:
            raise RuntimeError("stroke_core_weight is positive but stroke_core_opt did not build a loss")
        if stroke_overshoot_scale > 0.0 and self.cri_stroke_overshoot is None:
            raise RuntimeError("stroke_overshoot_weight is positive but stroke_overshoot_opt did not build a loss")

        if self.cri_stroke_core is not None and stroke_core_scale > 0.0:
            if self.gt_rgb is None:
                raise RuntimeError("StrokeCore loss requires full-resolution gt_rgb")
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(
                output444, format="RGB", is_train=True, uv_normalized=True
            )
            raw_core, core_details = self.cri_stroke_core(
                output_rgb,
                self.gt_rgb,
                text_rgb,
                valid_rgb,
                return_details=True,
            )
            total = total + stroke_core_scale * raw_core
            logs["l_stroke_core_raw"] = raw_core.detach()
            logs["l_stroke_core_weighted"] = (stroke_core_scale * raw_core).detach()
            logs["l_stroke_core"] = (stroke_core_scale * raw_core).detach()
            logs["stroke_core_skeleton_pixels"] = core_details["skeleton_pixels"].mean().detach()
            logs["stroke_core_active_samples"] = core_details["active_samples"].mean().detach()
            logs["stroke_core_abs_error"] = core_details["core_abs_error"].mean().detach()
        elif self.cri_stroke_core is not None:
            zero = self.output.sum() * 0.0
            logs["l_stroke_core_raw"] = zero
            logs["l_stroke_core_weighted"] = zero
            logs["l_stroke_core"] = zero
            logs["stroke_core_skeleton_pixels"] = zero
            logs["stroke_core_active_samples"] = zero
            logs["stroke_core_abs_error"] = zero

        if self.cri_stroke_overshoot is not None and stroke_overshoot_scale > 0.0:
            if self.gt_rgb is None:
                raise RuntimeError("Stroke overshoot loss requires full-resolution gt_rgb")
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(
                output444, format="RGB", is_train=True, uv_normalized=True
            )
            raw_overshoot, overshoot_details = self.cri_stroke_overshoot(
                output_rgb,
                self.gt_rgb,
                text_rgb,
                valid_rgb,
                return_details=True,
            )
            total = total + stroke_overshoot_scale * raw_overshoot
            logs["l_stroke_overshoot_raw"] = raw_overshoot.detach()
            logs["l_stroke_overshoot_weighted"] = (stroke_overshoot_scale * raw_overshoot).detach()
            logs["l_stroke_overshoot"] = (stroke_overshoot_scale * raw_overshoot).detach()
            logs["stroke_overshoot_active_pixels"] = overshoot_details["active_pixels"].mean().detach()
            logs["stroke_overshoot_upper_fraction"] = overshoot_details["upper_active_fraction"].mean().detach()
            logs["stroke_overshoot_lower_fraction"] = overshoot_details["lower_active_fraction"].mean().detach()
        elif self.cri_stroke_overshoot is not None:
            zero = self.output.sum() * 0.0
            logs["l_stroke_overshoot_raw"] = zero
            logs["l_stroke_overshoot_weighted"] = zero
            logs["l_stroke_overshoot"] = zero
            logs["stroke_overshoot_active_pixels"] = zero
            logs["stroke_overshoot_upper_fraction"] = zero
            logs["stroke_overshoot_lower_fraction"] = zero

        gestalt_scale = self._schedule(float(train_opt.get("text_gestalt_weight", 0.0)), current_iter)
        if self.cri_text_gestalt is not None and gestalt_scale > 0.0:
            text_rgb, valid_rgb = self._structure_masks()
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            output_rgb = output_rgb if output_rgb is not None else YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
            raw_gestalt = self.cri_text_gestalt(output_rgb, self.gt_rgb, text_rgb, valid_rgb)
            total = total + gestalt_scale * raw_gestalt
            logs["l_text_gestalt_raw"], logs["l_text_gestalt"] = raw_gestalt.detach(), (gestalt_scale * raw_gestalt).detach()

        if self.cri_grad is not None and grad_weight > 0.0:
            output444 = output444 if output444 is not None else y4uv_to_yuv444(self.output)
            target444 = y4uv_to_yuv444(self.gt)
            raw_grad = self.cri_grad(output444, target444, valid_mask=None, text_mask=None)
            total = total + grad_weight * raw_grad
            logs["l_grad_raw"] = raw_grad.detach()
            logs["l_grad"] = (grad_weight * raw_grad).detach()
        elif self.cri_grad is not None:
            logs["l_grad_raw"] = self.output.sum() * 0.0
            logs["l_grad"] = self.output.sum() * 0.0

        total.backward()
        self.optimizer_g.step()
        logs["l_total"] = total.detach()
        self.log_dict = self.reduce_loss_dict(logs)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """@shanlan-comment
        Input: validation loader yielding six-channel Y4UV batches and the current iteration.
        Output: logged native/reference validation means without RGB image serialization.
        Key Steps: run inference, compute native loss, and optionally retain decoded-reference diagnostics.
        Key Params: ``validation_reference_loss`` controls validation-only reference comparison.
        State/Dependencies: uses the configured losses and TensorBoard/root logger when supplied.
        """
        dataset_name = dataloader.dataset.opt["name"]
        sums = {
            "l_native": 0.0,
            "l_reference": 0.0,
            "l_compare_abs": 0.0,
        }
        sample_count = 0
        for val_data in dataloader:
            self.feed_data(val_data)
            self.test()
            native_loss = self.cri_y4uv_native(
                self.output,
                self.gt,
                valid_mask=self.valid_mask,
                text_mask=self.text_mask,
            )
            if self.validation_reference_loss:
                reference_loss = self.cri_yuv444_reference(
                    self.output,
                    self.gt,
                    valid_mask=self.valid_mask,
                    text_mask=self.text_mask,
                )
                compare_abs = (native_loss - reference_loss).abs()
            else:
                reference_loss = self.output.new_zeros(())
                compare_abs = self.output.new_zeros(())
            comparison = {
                "l_native": native_loss,
                "l_reference": reference_loss,
                "l_compare_abs": compare_abs,
            }
            batch_size = int(self.output.shape[0])
            sample_count += batch_size
            for name in sums:
                sums[name] += float(comparison[name].detach().float().item()) * batch_size
            del self.lq, self.output
            if hasattr(self, "original_lq"):
                del self.original_lq
            if hasattr(self, "gt"):
                del self.gt
            torch.cuda.empty_cache()

        if sample_count == 0:
            raise RuntimeError(f"Y4UV validation dataset {dataset_name!r} is empty")
        values = {name: value / sample_count for name, value in sums.items()}
        logger = get_root_logger()
        logger.info(
            "Validation %s [Y4UV] iter %d: l_native=%.6e l_reference=%.6e "
            "l_compare_abs=%.6e samples=%d",
            dataset_name,
            current_iter,
            values["l_native"],
            values["l_reference"],
            values["l_compare_abs"],
            sample_count,
        )
        if tb_logger:
            for name, value in values.items():
                tb_logger.add_scalar(f"val/{dataset_name}/{name}", value, current_iter)
        if save_img:
            logger.warning("Y4UV validation ignores save_img because [B,6,H/2,W/2] is not an RGB image")


__all__ = ["BasicTextLineY4UVTrainingModelV3", "resolve_reference_loss_flags"]
