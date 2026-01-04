import os
import argparse
import base64
import importlib.util
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

try:
    import decord
except Exception:
    decord = None

try:
    from PIL import Image
except Exception:
    Image = None

OpenAI = None
if importlib.util.find_spec("openai") is not None:
    from openai import OpenAI as _OpenAI

    OpenAI = _OpenAI

# ==========================================
# VLM Interface Definition
# ==========================================

class VLMClient:
    """Simple Vision-Language Model client (OpenAI-style multimodal chat)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.gptoai.top/v1")
        self.client = None

        if self.api_key and OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _frame_to_image_url(frame: Any) -> Optional[str]:
        if isinstance(frame, str):
            if frame.startswith("data:image") or frame.startswith("http"):
                return frame
            if os.path.exists(frame):
                with open(frame, "rb") as f:
                    payload = base64.b64encode(f.read()).decode("utf-8")
                suffix = Path(frame).suffix.lower().lstrip(".") or "png"
                return f"data:image/{suffix};base64,{payload}"
            return None

        if hasattr(frame, "save"):
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{payload}"

        return None

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        video_frames: List[Any],
    ) -> str:
        """Call the VLM with text prompts and visual inputs (video frames)."""
        if not self.client:
            print(
                f"[VLM Mock] System: {system_prompt[:50]}... | "
                f"User: {user_prompt[:50]}... | Frames: {len(video_frames)}"
            )
            return "Mock VLM Response"

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for frame in video_frames:
            url = self._frame_to_image_url(frame)
            if url:
                user_content.append({"type": "image_url", "image_url": {"url": url}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# ==========================================
# Data Structures
# ==========================================

@dataclass
class InteractionOutput:
    """Output of the Hand-Object Interaction Understanding Model (inferred from video)."""
    object_name: str
    functional_affordance: str
    contact_description: str
    palm_contact: bool
    raw_response: str

# ==========================================
# OmniDexReasoner Modules (Video-Based)
# ==========================================

class HandObjectInteractionModel:
    """
    1) Hand–Object Interaction Understanding Model
    
    Modified to work with Video Input only.
    Uses VLM to infer object properties, affordance, and contact details.
    """
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def predict(self, video_frames: List[Any]) -> InteractionOutput:
        system_prompt = """You are an expert in robotic grasping and human-object interaction understanding. 
Your task is to analyze a first-person video of a human hand grasping an object."""
        
        user_prompt = """Analyze the provided video frames.
1. Identify the object being grasped.
2. Determine the functional affordance being served by the grasp (e.g., Handle, Wrap, Press, Pour).
3. Describe the contact points between the hand and the object (which fingers, which parts).
4. Determine if the palm is in contact with the object (Yes/No).

Output the result in JSON format with keys: "object_name", "functional_affordance", "contact_description", "palm_contact" (boolean)."""

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )
        
        # In a real implementation, parse the JSON response.
        # Here we return mock data or try to parse if the VLM was real.
        try:
            # Mocking the parsing logic for the placeholder response
            if response_text == "Mock VLM Response":
                return InteractionOutput(
                    object_name="Unknown Object",
                    functional_affordance="Unknown Affordance",
                    contact_description="Unknown Contact",
                    palm_contact=False,
                    raw_response=response_text
                )
            data = json.loads(response_text)
            return InteractionOutput(
                object_name=data.get("object_name", "Unknown"),
                functional_affordance=data.get("functional_affordance", "None"),
                contact_description=data.get("contact_description", ""),
                palm_contact=data.get("palm_contact", False),
                raw_response=response_text
            )
        except:
            return InteractionOutput(
                object_name="Error Parsing",
                functional_affordance="Error",
                contact_description="Error",
                palm_contact=False,
                raw_response=response_text
            )


class GraspTaxonomyModel:
    """
    2) Grasp Taxonomy Understanding with Multimodal Reasoning
    
    Modified to work with Video Input + Interaction Context.
    Uses VLM to classify the grasp type based on the taxonomy.
    """
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def predict(self, video_frames: List[Any], interaction_context: InteractionOutput) -> str:
        system_prompt = """You are an expert in Grasp Taxonomy. 
You need to classify the grasp type into a hierarchical taxonomy:
- Coarse Level: Power, Precision, Intermediate.
- Fine Level: e.g., Tripod, Pinch, Large Diameter, Sphere 3-Finger, etc."""

        user_prompt = f"""Based on the video frames and the following interaction context:
- Object: {interaction_context.object_name}
- Affordance: {interaction_context.functional_affordance}
- Contact Details: {interaction_context.contact_description}
- Palm Contact: {interaction_context.palm_contact}

Classify the grasp type. 
Provide the output as "CoarseType:FineType" (e.g., "Precision:Tripod")."""

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )
        
        return response_text.strip()


class SemanticDescriptionGenerator:
    """
    3) Dexterous Grasp Multi-Dimensional Semantic Description Generation Model
    
    Modified to generate description from Video + Inferred Semantics.
    """
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def generate(self, 
                 video_frames: List[Any], 
                 interaction_context: InteractionOutput, 
                 grasp_type: str) -> str:
        system_prompt = """You are a semantic description generator for robotic grasping.
Your goal is to produce a natural language description that characterizes the grasp action."""

        user_prompt = f"""Generate a concise but descriptive sentence about the grasp shown in the video.
Integrate the following semantic information:
- Object: {interaction_context.object_name}
- Functional Affordance: {interaction_context.functional_affordance}
- Grasp Type: {grasp_type}
- Contact Details: {interaction_context.contact_description}

Example format: "Two fingertips pinch the handle of the mug, performing a Precision:Tripod grasp to lift it."
"""

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )
        
        return response_text


class OmniDexReasoner:
    """
    Main framework integrating the three modules for Video-Only input.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.vlm_client = VLMClient(api_key=api_key)
        
        # Initialize modules with the VLM client
        self.interaction_model = HandObjectInteractionModel(self.vlm_client)
        self.taxonomy_model = GraspTaxonomyModel(self.vlm_client)
        self.description_generator = SemanticDescriptionGenerator(self.vlm_client)

    def reason(self, video_frames: List[Any]) -> Dict[str, Any]:
        """
        Main reasoning pipeline.
        
        Args:
            video_frames: A list of image data representing the first-person video.
        
        Returns:
            Dictionary containing the inferred semantics and final description.
        """
        
        # 1. Understand Interaction (VLM Call 1)
        # Infers object, affordance, and contact from video
        interaction_result = self.interaction_model.predict(video_frames)
        
        # 2. Reason about Grasp Taxonomy (VLM Call 2)
        # Uses video + interaction context to classify grasp
        grasp_type = self.taxonomy_model.predict(video_frames, interaction_result)
        
        # 3. Generate Semantic Description (VLM Call 3)
        # Synthesizes all info into a natural language description
        description = self.description_generator.generate(
            video_frames, 
            interaction_result, 
            grasp_type
        )
        
        return {
            "object_name": interaction_result.object_name,
            "affordance": interaction_result.functional_affordance,
            "grasp_type": grasp_type,
            "description": description,
            "palm_contact": interaction_result.palm_contact,
            "contact_details": interaction_result.contact_description
        }


def _load_frames_from_dir(frames_dir: str, max_frames: int = 8) -> List[str]:
    directory = Path(frames_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))
    paths = sorted(paths)[:max_frames]
    return [str(path) for path in paths]


def _extract_frames_from_video(
    video_path: str,
    extract_fps: int = 2,
    max_frames: int = 15,
) -> List[Any]:
    if decord is None or Image is None:
        raise RuntimeError(
            "Video frame extraction requires `decord` and `Pillow`. "
            "Install with: pip install decord pillow"
        )

    vr = decord.VideoReader(video_path)
    local_fps = vr.get_avg_fps()
    if local_fps <= 0:
        local_fps = 20

    duration_sec = len(vr) / local_fps
    total_frames = len(vr)
    if total_frames <= 0:
        return []
    if total_frames == 1:
        frame_indexes = [0] * max_frames
    else:
        step = (total_frames - 1) / max(1, max_frames - 1)
        frame_indexes = [int(round(i * step)) for i in range(max_frames)]

    frames = vr.get_batch(frame_indexes).asnumpy()
    images: List[Any] = []
    for frame in frames:
        images.append(Image.fromarray(frame))
    return images


def _resolve_video_from_dir(video_dir: str) -> str:
    directory = Path(video_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    candidates = sorted(directory.glob("*.mp4"))
    if not candidates:
        candidates = sorted(directory.glob("*.avi"))
    if not candidates:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    return str(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniDexReasoner on a video.")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to a video file (mp4/avi).",
    )
    source_group.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing a video file (mp4/avi).",
    )
    source_group.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory containing image frames (png/jpg/webp).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=15,
        help="Maximum number of frames to send to the VLM.",
    )
    parser.add_argument(
        "--extract-fps",
        type=int,
        default=2,
        help="FPS used to sample frames from the video.",
    )
    args = parser.parse_args()

    video_frames: List[Any] = []
    if args.video_path:
        video_frames = _extract_frames_from_video(
            args.video_path,
            extract_fps=args.extract_fps,
            max_frames=args.max_frames,
        )
    elif args.video_dir:
        resolved_path = _resolve_video_from_dir(args.video_dir)
        video_frames = _extract_frames_from_video(
            resolved_path,
            extract_fps=args.extract_fps,
            max_frames=args.max_frames,
        )
    elif args.frames_dir:
        video_frames = _load_frames_from_dir(args.frames_dir, max_frames=args.max_frames)

    reasoner = OmniDexReasoner()
    result = reasoner.reason(video_frames)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
