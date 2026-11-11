from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.basic_loss import CharbonnierLoss, L1Loss, MSELoss, PerceptualLoss, WeightedTVLoss
from basicsr.losses.gan_loss import GANLoss, g_path_regularize, gradient_penalty_loss, r1_penalty
from .Recognition_loss import RecognitionLoss, RecognitionV2Loss
from .DiffCropRecog_loss import OCRCTCLoss, PPOCRTextPercepLoss
from .aesop_loss import AutoEncoderLoss
from .LDL_loss import compute_gradient_penalty
from .Gradient_loss import SobelGradientLoss
from .relativeGAN_loss import RelativisticSoftplusGANLoss
from .masked_l1_loss import MaskedL1Loss
from .masked_l1_loss_v2 import TextRegionPerceptualLossV2
from .y_only_sobel_loss import YOnlySobelLoss
from .fft_loss import FFTAmplitudeLoss
from .radial_freq_loss import RadialFreqLoss
from .crnn_content_perceptual_loss import CRNNTextLineContentLoss
from .basic_text_line_p0_losses import (
    BoxAwareMaskedL1LossV2,
    YOnlySobelMaskedLossV2,
    CPLineEligibilityV2,
    EligibleCPAdapterV2,
)
from .background_low_frequency_loss import BackgroundLowFrequencyGuardLoss, BackgroundLowFrequencyLoss, far_background_mask
from .keep_loss import KeepLoss
from .thin_skeleton_fbeta_loss import ThinSkeletonFbetaLoss, SkelFbetaLoss
from .stroke_geometry_losses import SoftCLDiceLoss, StrokeProbabilityBCELoss, StrokeWidthLoss
from .normal_profile_loss import BoundaryNormalProfileLoss, GTBoundaryNormalProfileLoss, NormalProfileLoss
from .gt_sdf_boundary_loss import (
    GTSDFBoundaryLoss,
    GTSDFNarrowBandLoss,
    SignedDistanceBoundaryLoss,
    build_gt_sdf_band,
    soft_ink_probability as gt_sdf_soft_ink_probability,
)
from .stroke_core_overshoot_loss import StrokeCoreLoss, OvershootLoss, StrokeOvershootLoss

# __all__ = [
#     'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss',
#     'r1_penalty', 'g_path_regularize', 'compute_gradient_penalty',
# ]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
