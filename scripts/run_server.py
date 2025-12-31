#!/usr/bin/env python3

"""
DexHand Dual-Camera Policy Inference Server.

This script starts a ZMQ-based inference server for the DexHand dual-camera policy.
The server loads a trained OpenPi checkpoint and exposes it via ZMQ for robot integration.

Endpoints:
    - "ping": Check if server is alive
    - "kill": Shut down the server
    - "get_action": Get action for observations
    - "get_modality_config": Get policy modality configuration

Example usage:
    python run_server.py \\
        --checkpoint /disk0/zwp/WORK_DIR/openpi-NLL/.../10000 \\
        --port 5555

    # From another machine:
    python client_example.py --host localhost --port 5555
    
Yiqing Wang Exanmple
    - pi05_adamu
    # Example usage:
    python scripts/run_server.py \
        --checkpoint /mnt/pfs/scalelab/yiqing/openpi/checkpoints/pi05_adamu_continue/pi05_adamu_v4_continue/7000 \
        --config pi05_adamu_continue \
        --host 0.0.0.0 \
        --port 5555 \
        --camera-mode dual \
        --default-prompt "pick up the bottle"
"""

import argparse
import dataclasses
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.rtc_processor import RTCConfig, RTCAttentionSchedule
from openpi.training import config as _config
from dexhand_dual_service import DexHandDualInferenceServer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DexHandPolicyWrapper:
    """
    Wrapper around OpenPi policy to provide get_action and get_modality_config methods.
    Supports both single-camera and dual-camera models.
    """

    def __init__(
        self,
        policy: _policy.Policy,
        camera_mode: str = "dual",
        default_prompt: Optional[str] = None,
    ):
        """
        Initialize the policy wrapper.

        Args:
            policy: OpenPi policy instance
            camera_mode: Camera configuration mode
                - "single": Single camera (observation/image)
                - "dual": DexHand dual camera (camera_1, camera_2)
                - "dual_adamu": AdamU dual camera (camera_0, camera_1)
            default_prompt: Default prompt to use if not provided
        """
        if camera_mode not in ["single", "dual", "dual_adamu"]:
            raise ValueError(f"camera_mode must be 'single', 'dual', or 'dual_adamu', got: {camera_mode}")

        self.policy = policy
        self.camera_mode = camera_mode
        self.default_prompt = default_prompt

        logger.info(f"Policy wrapper initialized with camera_mode={camera_mode}")

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action from policy for given observations.

        Args:
            observations: Dict with keys (robot format):
                - "state.state": joint positions (1, 13) or (13,)
                - "video.camera_1_rgb": camera 1 image (1, H, W, 3) or (H, W, 3)
                - "video.camera_2_rgb": camera 2 image (1, H, W, 3) or (H, W, 3)
                - "annotation.human.action.task_description": task description (list or string)

            Alternative formats also supported:
                - "observation/camera_1", "observation/camera_2", "observation/state", "prompt"

        Returns:
            Dict with key "actions": action array (13,)
        """
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Key mapping (robot format → policy format)
        # ═══════════════════════════════════════════════════════════════
        if self.camera_mode == "single":
            # Single-camera model uses "observation/image" key
            key_mapping = {
                # Robot format
                'state.state': 'observation/state',
                'video.camera_1_rgb': 'observation/image',  # Map camera_1 to image for single-camera
                'video.image_rgb': 'observation/image',
                'annotation.human.action.task_description': 'prompt',

                # Alternative formats
                'camera_1': 'observation/image',
                'image': 'observation/image',
                'state': 'observation/state',
                'task': 'prompt',
            }
        elif self.camera_mode == "dual":
            # DexHand dual camera: camera_1 and camera_2 → policy camera_1 and camera_2
            key_mapping = {
                # Robot format - DexHand convention (camera_1, camera_2)
                'state.state': 'observation/state',
                'video.camera_1_rgb': 'observation/camera_1',
                'video.camera_2_rgb': 'observation/camera_2',
                'video.camera.1.rgb': 'observation/camera_1',  # Alternative dot notation
                'video.camera.2.rgb': 'observation/camera_2',
                'annotation.human.action.task_description': 'prompt',

                # Alternative formats
                'camera_1': 'observation/camera_1',
                'camera_2': 'observation/camera_2',
                'camera.1': 'observation/camera_1',
                'camera.2': 'observation/camera_2',
                'state': 'observation/state',
                'task': 'prompt',
            }
        elif self.camera_mode == "dual_adamu":
            # AdamU dual camera: camera_0 and camera_1 → policy camera_1 and camera_2
            key_mapping = {
                # Robot format - AdamU convention (camera_0, camera_1)
                'state.state': 'observation/state',
                'video.camera_0_rgb': 'observation/camera_1',  # camera_0 → policy camera_1
                'video.camera_1_rgb': 'observation/camera_2',  # camera_1 → policy camera_2
                'video.camera.0.rgb': 'observation/camera_1',  # Alternative dot notation
                'video.camera.1.rgb': 'observation/camera_2',
                'annotation.human.action.task_description': 'prompt',

                # Alternative formats
                'camera_0': 'observation/camera_1',
                'camera_1': 'observation/camera_2',
                'camera.0': 'observation/camera_1',
                'camera.1': 'observation/camera_2',
                'state': 'observation/state',
                'task': 'prompt',
            }
        else:
            raise ValueError(f"Unknown camera_mode: {self.camera_mode}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Transform observations
        # ═══════════════════════════════════════════════════════════════
        obs = {}

        # Extract RTC parameters (if present) BEFORE key mapping
        # These are for Training-Time RTC support - client sends previously executed actions
        prev_chunk_left_over = observations.pop('prev_chunk_left_over', None)
        inference_delay = observations.pop('inference_delay', 0)
        execution_horizon = observations.pop('execution_horizon', None)

        # Also check for deprecated 'prev_action' key (fallback)
        if prev_chunk_left_over is None and 'prev_action' in observations:
            prev_chunk_left_over = observations.pop('prev_action')

        for client_key, value in observations.items():
            # Convert lists to numpy arrays
            if isinstance(value, list):
                # Special handling for prompt (list of strings)
                if 'task_description' in client_key or 'prompt' in client_key:
                    # Extract first string from list
                    value = value[0] if len(value) > 0 else ""
                else:
                    value = np.array(value)

            # Map key to policy format
            policy_key = key_mapping.get(client_key, client_key)

            # Transform data based on key
            if isinstance(value, np.ndarray):
                # Remove batch dimension if present (squeeze first dim if size is 1)
                if value.ndim > 1 and value.shape[0] == 1:
                    value = value.squeeze(0)
                    logger.debug(f"Removed batch dimension from {policy_key}: shape after squeeze = {value.shape}")

                # Handle camera images
                if 'camera' in policy_key:
                    # Ensure uint8 [0, 255] format
                    if value.dtype in [np.float32, np.float64]:
                        if value.max() <= 1.0:
                            value = (value * 255).astype(np.uint8)

                    # Note: Policy can handle different image sizes, no need to resize
                    # If you need specific size, uncomment below:
                    # import cv2
                    # if value.shape[:2] != (224, 224):
                    #     value = cv2.resize(value, (224, 224))

                    logger.debug(f"{policy_key}: shape={value.shape}, dtype={value.dtype}")

                # Handle state
                elif 'state' in policy_key:
                    # Ensure float32
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    logger.debug(f"{policy_key}: shape={value.shape}, dtype={value.dtype}")

            obs[policy_key] = value

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Handle default prompt
        # ═══════════════════════════════════════════════════════════════
        if "prompt" not in obs:
            if self.default_prompt:
                obs["prompt"] = self.default_prompt
            else:
                raise ValueError("prompt must be provided in observations or via default_prompt")

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Validate transformed observations
        # ═══════════════════════════════════════════════════════════════
        logger.debug(f"Client keys: {list(observations.keys())}")
        logger.debug(f"Policy keys: {list(obs.keys())}")
        logger.debug(f"Getting action for prompt: {obs.get('prompt')}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Call policy with RTC parameters
        # ═══════════════════════════════════════════════════════════════
        # Pass RTC parameters to enable Training-Time RTC during inference
        result = self.policy.infer(
            obs,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        actions = result["actions"]

        # Ensure actions are in the right format
        if isinstance(actions, np.ndarray):
            actions = actions.astype(np.float32)
        else:
            actions = np.array(actions, dtype=np.float32)

        logger.debug(f"Action shape: {actions.shape}, dtype: {actions.dtype}")

        return {"actions": actions}

    def get_modality_config(self) -> Dict[str, Any]:
        """
        Get the modality configuration of the policy.

        Returns:
            Dict with policy metadata and modality info
        """
        metadata = self.policy.metadata

        if self.camera_mode == "single":
            return {
                "metadata": str(metadata),
                "action_dim": 13,
                "state_dim": 13,
                "camera_count": 1,
                "camera_mode": "single",
                "image_modalities": ["image"],
            }
        else:  # dual
            return {
                "metadata": str(metadata),
                "action_dim": 13,
                "state_dim": 13,
                "camera_count": 2,
                "camera_mode": "dual",
                "image_modalities": ["camera_1", "camera_2"],
            }


def apply_nll_enhancements_overrides(config, args):
    """
    Apply NLL enhancement parameter overrides to the config.

    All parameters must be specified manually via command-line arguments.
    No automatic detection from checkpoint names or config.json files.

    Args:
        config: Training configuration from _config.get_config()
        args: Command-line arguments with override values

    Returns:
        Modified config with overridden model parameters
    """
    model_overrides = {}

    # Handle vanilla mode (disable all NLL enhancements)
    if args.vanilla or (args.use_nll_enhancements is False):
        logger.info("\n🔧 VANILLA MODE: Disabling all NLL enhancements")
        model_overrides.update({
            'use_nll_enhancements': False,
            'use_wasserstein_attention': False,
            'use_mi_loss': False,
            'refinement_iterations': 1,
            'num_projections': 0,
            'enhancement_weight': 0.0,
            'mi_loss_weight': 0.0,
            'mi_temperature': 0.0,
        })
    # Handle explicit enable/disable of enhancements
    elif args.use_nll_enhancements is True:
        logger.info("\n✅ NLL ENHANCEMENTS: Explicitly enabled")
        model_overrides.update({
            'use_nll_enhancements': True,
            'use_wasserstein_attention': True,  # Critical: must enable this too!
        })

    # Collect explicit parameter overrides
    if args.action_horizon is not None:
        model_overrides['action_horizon'] = args.action_horizon
        logger.info(f"Override: action_horizon = {args.action_horizon}")
    if args.refinement_iterations is not None:
        model_overrides['refinement_iterations'] = args.refinement_iterations
        logger.info(f"Override: refinement_iterations = {args.refinement_iterations}")
    if args.num_projections is not None:
        model_overrides['num_projections'] = args.num_projections
        logger.info(f"Override: num_projections = {args.num_projections}")
    if args.enhancement_weight is not None:
        model_overrides['enhancement_weight'] = args.enhancement_weight
        logger.info(f"Override: enhancement_weight = {args.enhancement_weight}")
    if args.mi_loss_weight is not None:
        model_overrides['mi_loss_weight'] = args.mi_loss_weight
        logger.info(f"Override: mi_loss_weight = {args.mi_loss_weight}")
    if args.mi_temperature is not None:
        model_overrides['mi_temperature'] = args.mi_temperature
        logger.info(f"Override: mi_temperature = {args.mi_temperature}")

    # Apply RTC configuration if enabled
    if args.rtc_enabled:
        logger.info("\n🤖 RTC Configuration:")
        logger.info(f"  Schedule: {args.rtc_schedule}")
        logger.info(f"  Max guidance weight: {args.rtc_max_guidance_weight}")
        logger.info(f"  Execution horizon: {args.rtc_execution_horizon}")

        # Create RTC config
        schedule_map = {
            "zeros": RTCAttentionSchedule.ZEROS,
            "ones": RTCAttentionSchedule.ONES,
            "linear": RTCAttentionSchedule.LINEAR,
            "exp": RTCAttentionSchedule.EXP,
        }

        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule_map[args.rtc_schedule],
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
        )

        model_overrides['rtc_config'] = rtc_config

    # If no overrides, return original config
    if not model_overrides:
        return config

    # Create new model config with overrides (frozen dataclass requires replace)
    new_model = dataclasses.replace(config.model, **model_overrides)
    new_config = dataclasses.replace(config, model=new_model)

    return new_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DexHand Dual-Camera Policy Inference Server (ZMQ)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config name (auto-detected from camera-mode if not provided)"
    )

    parser.add_argument(
        "--camera-mode",
        type=str,
        choices=["single", "dual", "dual_adamu"],
        default="dual",
        help="Camera mode: 'single' (one camera), 'dual' (DexHand: camera_1/2), 'dual_adamu' (AdamU: camera_0/1)"
    )

    parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Use vanilla baseline model (disable all NLL enhancements)"
    )

    parser.add_argument(
        "--use-nll-enhancements",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Explicitly enable/disable NLL enhancements (true/false)"
    )

    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt if not provided in request"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to serve on"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="Host to bind to ('*' for all interfaces)"
    )

    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="API token for authentication (optional)"
    )

    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Action horizon in steps (e.g., 16). Extract from checkpoint dir name."
    )

    # NLL Enhancement Parameters (must match training configuration!)
    parser.add_argument(
        "--refinement-iterations",
        type=int,
        default=None,
        help="Number of iterative refinement steps (e.g., 1, 3, 5). Extract from checkpoint dir name."
    )
    parser.add_argument(
        "--num-projections",
        type=int,
        default=None,
        help="Number of Sliced Wasserstein projections (e.g., 5, 8, 32). Extract from checkpoint dir name."
    )
    parser.add_argument(
        "--enhancement-weight",
        type=float,
        default=None,
        help="Weight for mixing enhanced features (e.g., 0.01, 0.1, 0.3). Extract from checkpoint dir name."
    )
    parser.add_argument(
        "--mi-loss-weight",
        type=float,
        default=None,
        help="InfoNCE mutual information loss weight (e.g., 0, 0.0001). Extract from checkpoint dir name."
    )
    parser.add_argument(
        "--mi-temperature",
        type=float,
        default=None,
        help="InfoNCE temperature for contrastive learning (e.g., 0.1). Extract from checkpoint dir name."
    )

    # RTC (Real-Time Chunking) parameters - inference only
    parser.add_argument(
        "--rtc-enabled",
        action="store_true",
        help="Enable RTC for smooth chunk transitions (inference only)"
    )
    parser.add_argument(
        "--rtc-schedule",
        type=str,
        default="exp",
        choices=["zeros", "ones", "linear", "exp"],
        help="RTC attention schedule (default: exp)"
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=5.0,
        help="RTC maximum guidance weight (beta parameter, default: 5.0)"
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=5,
        help="RTC execution horizon - actions executed per chunk (default: 5 for H=10)"
    )

    args = parser.parse_args()

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint directory does not exist: {args.checkpoint}")
        return 1

    # # Auto-detect config name based on camera mode if not provided
    # if args.config is None:
    #     if args.camera_mode == "single":
    #         args.config = "pi05_dexhand_nll_enhanced"
    #     else:  # dual
    #         args.config = "pi05_dexhand_dual_nll_enhanced"
    #     logger.info(f"Auto-detected config: {args.config} (from camera_mode={args.camera_mode})")

    # Print header
    logger.info("=" * 70)
    logger.info(f"DexHand Policy Inference Server (Camera Mode: {args.camera_mode.upper()})")
    if args.vanilla:
        logger.info("Mode: VANILLA BASELINE (No NLL Enhancements)")
    elif args.use_nll_enhancements is False:
        logger.info("Mode: VANILLA BASELINE (NLL Enhancements Disabled)")
    elif args.use_nll_enhancements is True:
        logger.info("Mode: NLL-ENHANCED")
    else:
        logger.info("Mode: Using config defaults")
    logger.info("=" * 70)

    # Load policy with NLL enhancement overrides
    logger.info(f"Loading config: {args.config}")
    config = _config.get_config(args.config)
    logger.info(f"Config loaded: {config.name}")

    # Apply NLL enhancement parameter overrides (if provided)
    logger.info("\nApplying NLL enhancement parameter overrides...")
    config = apply_nll_enhancements_overrides(config, args)

    logger.info(f"\nLoading checkpoint from: {args.checkpoint}")
    policy = _policy_config.create_trained_policy(
        config,
        args.checkpoint,
        default_prompt=args.default_prompt
    )
    logger.info("Policy loaded successfully!")

    # Wrap policy
    logger.info(f"Wrapping policy with camera_mode={args.camera_mode}...")
    wrapped_policy = DexHandPolicyWrapper(
        policy=policy,
        camera_mode=args.camera_mode,
        default_prompt=args.default_prompt,
    )

    # Create and start server
    logger.info("=" * 70)
    logger.info(f"Starting ZMQ server on tcp://{args.host}:{args.port}")
    logger.info("=" * 70)

    server = DexHandDualInferenceServer(
        policy=wrapped_policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
    )

    # Print server information
    logger.info("\nEndpoints:")
    logger.info("  - 'ping':              Health check (requires_input=False)")
    logger.info("  - 'kill':              Shutdown server (requires_input=False)")
    logger.info("  - 'get_action':        Get action for observations")
    logger.info("  - 'get_modality_config': Get policy modality configuration")

    logger.info("\nObservation format (for 'get_action'):")
    logger.info("  {")
    if args.camera_mode == "single":
        logger.info("      'observation/image': image array (H, W, 3)")
    else:  # dual
        logger.info("      'observation/camera_1': image array (H, W, 3)")
        logger.info("      'observation/camera_2': image array (H, W, 3)")
    logger.info("      'observation/state': state array (13,)")
    logger.info("      'prompt': task description string")
    logger.info("  }")

    logger.info("\nAction response:")
    logger.info("  {'actions': array (chunk_size, 13)}  # Action chunking enabled")
    logger.info("  Typical usage: actions[0] for first action")

    logger.info("\nConfiguration:")
    logger.info(f"  Camera mode: {args.camera_mode}")
    logger.info(f"  Vanilla mode: {args.vanilla}")
    logger.info(f"  NLL enhancements: {'disabled' if args.vanilla else 'check config'}")

    logger.info("\nPress Ctrl+C to stop the server")
    logger.info("=" * 70 + "\n")

    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
