import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_adamu_dual_example() -> dict:
    """Creates a random input example for the AdamU dual-camera policy."""
    return {
        "observation/state": np.random.rand(31),
        "observation/camera_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/camera_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AdamuDualInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For the AdamU dual-camera dataset, this class handles:
    - Dual camera views (camera_0 and camera_1)
    - 31D state space (joint positions)
    - 31D action space (joint targets)
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) since LeRobot may store as float32 (C,H,W)
        # For dual-camera AdamU, we have two camera views
        camera_1 = _parse_image(data["observation/camera_1"])
        camera_2 = _parse_image(data["observation/camera_2"])

        # Create inputs dict. Do not change the keys in the dict below.
        # Pi0 models support three image inputs: one third-person view and two wrist views.
        # For dual-camera AdamU:
        # - camera_1 → base_0_rgb (third-person view)
        # - camera_2 → left_wrist_0_rgb (second view)
        # - right_wrist_0_rgb remains padded with zeros
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": camera_1,
                "left_wrist_0_rgb": camera_2,
                # Pad missing third camera with zero-array
                "right_wrist_0_rgb": np.zeros_like(camera_1),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask out padding image for pi0 model (not for pi0-FAST)
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AdamuDualOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. It is
    used for inference only.

    For AdamU, we unpad actions from 32D model dimension back to 31D robot dimension.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 31 actions -- since we padded actions to fit the model action
        # dimension (32D), we need to now parse out the correct number of actions (31D).
        return {"actions": np.asarray(data["actions"][:, :31])}