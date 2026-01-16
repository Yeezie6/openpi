import os
import argparse
import base64
import importlib.util
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from .grasp_knowledge_base import GraspKnowledgeBase
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from tqdm import tqdm

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

# Configure logging (default to INFO to avoid verbose HTTP debug)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File logging will be configured later in `main()` when CLI args are available.

# Reduce verbosity for HTTP and SDK libraries that log request/response bodies
for noisy in ("httpx", "httpcore", "openai", "urllib3", "stainless"): 
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Dedicated logger for VLM conversations
vlm_logger = logging.getLogger("omnidex.vlm")
vlm_logger.setLevel(logging.INFO)
vlm_logger.propagate = False # Don't send VLM logs to the root/console by default

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write to prevent progress bar corruption."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-IF2Cfo0egVGW5RgYpsDFhw9MvBKSw7sptMNgzpOo7SCheoMs")
        self.model = model or os.environ.get("OPENAI_MODEL", "gemini-2.5-flash-lite")
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
            # Avoid printing large data (frames). Log minimal info instead.
            logger.info(f"[VLM Mock] user_prompt={user_prompt[:80]!r} frames={len(video_frames)}")
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
        content = response.choices[0].message.content.strip()

        # Log the conversation to the dedicated VLM logger
        vlm_logger.info("="*40)
        vlm_logger.info(f"MODEL: {self.model}")
        vlm_logger.info(f"SYSTEM PROMPT:\n{system_prompt}")
        vlm_logger.info(f"USER PROMPT:\n{user_prompt}")
        vlm_logger.info(f"RESPONSE:\n{content}")
        vlm_logger.info("="*40)

        return content


# ==========================================
# Data Structures
# ==========================================

@dataclass
class InteractionOutput:
    """Output of the Hand-Object Interaction Understanding Model (video-inferred)."""
    # grasp_category removed as requested
    opposition_type: str  # Opposition: Palm | Pad | Side
    thumb_position: str  # Thumb position: Abd | Add
    virtual_fingers: str  # Virtual finger grouping text description
    raw_response: str

# ==========================================
# OmniDexReasoner Modules (Video-Based)
# ==========================================

class HandObjectInteractionModel:
    """
    1) Hand–Object Interaction Understanding Model
    
    Modified to work with Video Input only.
    Uses VLM to infer grasp taxonomy attributes: Type, Opposition, Thumb posture, Virtual Fingers.
    """
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def detect_active_hands(self, video_frames: List[Any], debug: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Step 1.0: Identify which hands (Left/Right) are actively manipulating objects.
        """
        system_prompt = """You are an expert in analyzing human hand-object interactions in first-person videos.
The left side of the video frame corresponds to the Left Hand. The right side corresponds to the Right Hand.

RULES:
1. Analyze the video and determine which hands are actively manipulating an object.
2. Ignore hands that are visible but NOT interacting with any object.
3. Output format: A single JSON object with the key "active_hands" containing a list of strings (allowed: "left", "right").
4. Examples: {"active_hands": ["right"]}, {"active_hands": ["left", "right"]}, or {"active_hands": []}.
"""
        user_prompt = "Analyze the provided video frames and identify the active hands."
        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )
        
        parsed_active: List[str] = []
        try:
            import re
            m = re.search(r"\{.*\}", response_text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                active = data.get("active_hands", [])
                # Normalize and validate
                valid = []
                for h in active:
                    h_lower = str(h).lower()
                    if "left" in h_lower:
                        valid.append("left")
                    if "right" in h_lower:
                        valid.append("right")
                # Remove duplicates and sort
                parsed_active = sorted(list(set(valid)))
        except Exception:
            logger.warning(f"Failed to parse active hands from: {response_text}")
        
        # Fallback: assume right hand if uncertain, or return empty? 
        parsed_active = parsed_active or ["right"]
        if debug is not None:
            debug.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": response_text,
                    "parsed_active_hands": parsed_active,
                }
            )
        return parsed_active

    def predict(self, video_frames: List[Any], hand_side: str = "right", debug: Optional[Dict[str, Any]] = None) -> InteractionOutput:
        system_prompt = f"""You are an expert in human grasping and grasp taxonomy. 
Focus ONLY on the {hand_side.upper()} HAND (visible on the {hand_side} side of the image).

TASK:
Infer the following attributes for the {hand_side.upper()} HAND:
1) Opposition type: Palm (wrap/containment), Pad (fingertips), Side (lateral pinch).
2) Thumb position: Abd (abducted/open), Add (adducted/close to palm).
3) Virtual fingers (VF): Identify digits acting together (1=thumb, 2=index, 3=middle, 4=ring, 5=little). Format: "VF2: 1 vs 2-5".

INSTRUCTIONS:
- Chain of Thought: Integrate evidence across frames internally. Prefer the dominant/stable grasp.
- Uncertainty: Use "Unknown" if evidence is occluded or contradictory.
- Output Format: ONE valid JSON object ONLY:
{{
  "opposition_type": "Palm|Pad|Side|Unknown",
  "thumb_position": "Abd|Add|Unknown",
  "virtual_fingers": "string"
}}
"""
        user_prompt = f"Analyze the {hand_side.upper()} HAND in the provided video frames."


        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )

        # Debug: Log the raw VLM response
        logger.debug(f"VLM Response: {response_text}")

        if debug is not None:
            debug.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": response_text,
                }
            )
        
        try:
            # Try to robustly extract a JSON object from the response using regex.
            import re

            m = re.search(r"\{.*\}", response_text, re.DOTALL)

            if m:
                json_text = m.group(0)
                data = json.loads(json_text)
                parsed_output = InteractionOutput(
                    opposition_type=data.get("opposition_type", "Unknown"),
                    thumb_position=data.get("thumb_position", "Unknown"),
                    virtual_fingers=data.get("virtual_fingers", "Unknown"),
                    raw_response=response_text,
                )
                if debug is not None:
                    debug["parsed"] = asdict(parsed_output)
                return parsed_output

            # Fallback: attempt to read simple key: value patterns
            def _extract_kv(key_name: str) -> Optional[str]:
                # match both 'Key: value' and '"key": "value"'
                pat1 = re.search(rf"{key_name}\s*[:\-]\s*(.+)", response_text, re.IGNORECASE)
                if pat1:
                    val = pat1.group(1).strip()
                    # stop at end of line or semicolon
                    val = val.split('\n')[0].split(';')[0].strip(' \"')
                    return val
                pat2 = re.search(rf'"{key_name}"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
                if pat2:
                    return pat2.group(1)
                return None

            opposition_type = _extract_kv('opposition_type') or 'Unknown'
            thumb_position = _extract_kv('thumb_position') or 'Unknown'
            virtual_fingers = _extract_kv('virtual_fingers') or 'Unknown'
            if opposition_type == 'Unknown' and thumb_position == 'Unknown' and virtual_fingers == 'Unknown':
                logger.error("Failed to parse VLM response as JSON or key-value pairs.")
                parsed_output = InteractionOutput(
                    opposition_type="Error",
                    thumb_position="Error",
                    virtual_fingers="Error",
                    raw_response=response_text,
                )
                if debug is not None:
                    debug["parsed"] = asdict(parsed_output)
                return parsed_output

            parsed_output = InteractionOutput(
                opposition_type=opposition_type,
                thumb_position=thumb_position,
                virtual_fingers=virtual_fingers,
                raw_response=response_text,
            )
            if debug is not None:
                debug["parsed"] = asdict(parsed_output)
            return parsed_output
        except Exception as e:
            logger.exception("Unhandled exception while parsing VLM response")
            parsed_output = InteractionOutput(
                opposition_type="Error",
                thumb_position="Error",
                virtual_fingers="Error",
                raw_response=response_text,
            )
            if debug is not None:
                debug["parsed"] = asdict(parsed_output)
            return parsed_output


class GraspTaxonomyModel:
    """
    2) Grasp Taxonomy Understanding with Multimodal Reasoning
    
    Now augmented with a tiny RAG-style knowledge base + CoT prompting.
    """
    def __init__(self, vlm_client: VLMClient, knowledge_base: Optional[GraspKnowledgeBase] = None):
        self.vlm_client = vlm_client
        self.knowledge_base = knowledge_base or GraspKnowledgeBase()

    def _build_knowledge_context(self, query: str, top_k: int = 3) -> str:
        hits = self.knowledge_base.retrieve(query, top_k=top_k)
        blocks = []
        for h in hits:
            blocks.append(
                f"- Label: {h.get('name')}\n  Coarse Type: {h.get('coarse')}\n  Fine Type: {h.get('fine')}\n  Opposition: {h.get('opposition')}\n  Thumb: {h.get('thumb')}\n  Virtual Fingers: {h.get('virtual_fingers')}"
            )
        return "\n".join(blocks)

    def predict(self, video_frames: List[Any], interaction_context: InteractionOutput, trace: bool = False, debug: Optional[Dict[str, Any]] = None) -> str:
        # Build allowed labels from the small knowledge base
        allowed = [e.get('name') for e in self.knowledge_base.entries if e.get('name')]
        allowed_list_text = ', '.join(allowed)

        system_prompt = f"""You are an expert in Grasp Taxonomy. 
Classify the grasp type based on provided visual context and knowledge snippets.

RULES:
1) Use the provided attributes + retrieved knowledge as evidence.
2) Reason step-by-step internally.
3) Output MUST end with a single line in this exact format: "Final Grasp: Coarse:Fine".
4) Allowed labels: {allowed_list_text}
"""

        query_text = (
            f"Opposition: {interaction_context.opposition_type}\n"
            f"Thumb Position: {interaction_context.thumb_position}\n"
            f"Virtual Fingers: {interaction_context.virtual_fingers}"
        )
        knowledge_ctx = self._build_knowledge_context(query_text)

        user_prompt = f"""Context from video:
- Opposition Type: {interaction_context.opposition_type}
- Thumb Position: {interaction_context.thumb_position}
- Virtual Fingers: {interaction_context.virtual_fingers}

Reference Knowledge Snippets:
{knowledge_ctx}

Identify the best matching grasp label from the video frames and context above.
"""

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )

        if trace:
            logger.info("[Taxonomy] Prompted with knowledge:\n%s", knowledge_ctx)
            logger.info("[Taxonomy] Raw VLM response: %s", response_text)

        if debug is not None:
            debug.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "knowledge_context": knowledge_ctx,
                    "allowed_labels": allowed,
                    "raw_response": response_text,
                }
            )

        # Clean and validate the response to extract Coarse:Fine
        import re

        text = response_text.strip()

        # Strip lightweight markdown emphasis that can block regex word boundaries
        def _preclean(s: str) -> str:
            cleaned = re.sub(r"[*_`]+", "", s)
            cleaned = cleaned.replace("—", "-").replace("–", "-")
            return cleaned

        # Prefer an explicit "Final Grasp:" line if present
        final_line = None
        for line in text.splitlines():
            if "final" in line.lower() and "grasp" in line.lower():
                final_line = line
                break

        # Normalization helper to collapse newlines/extra spaces that can break regex matches
        def _normalize(s: str) -> str:
            return " ".join(s.split())

        candidate_text = final_line or text
        normalized_candidate = _normalize(_preclean(candidate_text))
        normalized_full = _normalize(_preclean(text))

        def _extract(label_text: str) -> Optional[str]:
            m_primary = re.search(r"\b(Power|Precision|Intermediate)\s*[:\-]\s*([A-Za-z0-9_ \-]+)\b", label_text, re.IGNORECASE)
            if m_primary:
                coarse = m_primary.group(1).capitalize()
                fine = m_primary.group(2).strip()
                parsed_label = f"{coarse}:{fine}"
                if debug is not None:
                    debug["parsed_label"] = parsed_label
                return parsed_label

            m_secondary = re.search(r"\b(Power|Precision|Intermediate)\b[\s\-:]*([A-Za-z0-9_\-]+)?", label_text, re.IGNORECASE)
            if m_secondary:
                coarse = m_secondary.group(1).capitalize()
                fine = (m_secondary.group(2) or "").strip()
                if fine:
                    return f"{coarse}:{fine}"
                return f"{coarse}:Unknown"
            return None

        parsed = _extract(normalized_candidate) or _extract(normalized_full)

        if parsed:
            # Validate and map parsed result to allowed labels only
            try:
                import difflib

                coarse, fine = parsed.split(':', 1)
                coarse = coarse.strip().capitalize()
                fine = fine.strip().replace(' ', '_')

                # Build allowed map: {Coarse: [Fine1, Fine2, ...]}
                allowed_map: Dict[str, List[str]] = {}
                for a in allowed:
                    if ':' in a:
                        c, f = a.split(':', 1)
                        allowed_map.setdefault(c.capitalize(), []).append(f)

                # If coarse not in allowed_map, try fuzzy match to known coarse categories
                if coarse not in allowed_map:
                    coarse_candidates = list(allowed_map.keys())
                    coarse_match = difflib.get_close_matches(coarse, coarse_candidates, n=1, cutoff=0.6)
                    if coarse_match:
                        coarse = coarse_match[0]

                fines_for_coarse = allowed_map.get(coarse, [])
                # Exact accept
                if fine in fines_for_coarse:
                    return f"{coarse}:{fine}"
                # Fuzzy match fine label to known fines
                close = difflib.get_close_matches(fine, fines_for_coarse, n=1, cutoff=0.6) if fines_for_coarse else []
                if close:
                    parsed_label = f"{coarse}:{close[0]}"
                    if debug is not None:
                        debug["parsed_label"] = parsed_label
                    return parsed_label
                # If fine unknown, return coarse:Unknown to avoid invented labels
                parsed_label = f"{coarse}:Unknown"
                if debug is not None:
                    debug["parsed_label"] = parsed_label
                return parsed_label
            except Exception:
                logger.exception("Error validating parsed grasp_type")
                if debug is not None:
                    debug["parsed_label"] = parsed
                return parsed

        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", normalized_candidate)
        if ":" in cleaned:
            if debug is not None:
                debug["parsed_label"] = cleaned
            return cleaned

        logger.warning(f"Unexpected grasp_type format. Response: {response_text}")
        if debug is not None:
            debug["parsed_label"] = "Error: Invalid Format"
        return "Error: Invalid Format"


class SemanticDescriptionGenerator:
    """
    3) Dexterous Grasp Multi-Dimensional Semantic Description Generation Model
    
    Modified to generate description from Video + Inferred Semantics.
    """
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def generate(
        self,
        video_frames: List[Any],
        hands_data: Dict[str, Any],
        active_hands: Optional[List[str]] = None,
        debug: Optional[Dict[str, Any]] = None,
    ) -> str:
        active_hands = active_hands or sorted(hands_data.keys())
        system_prompt = f"""You are a video annotation AI specialized in human manipulation tasks in first-person videos.

TASK: Generate a single-sentence task description summarizing the video.

REQUIREMENTS:
1. Objects & Affordances: Mention the objects being manipulated, their features, and SPECIFICALLY where they are grasped (functional affordance, e.g., "the handle of the mug", "the rim of the bowl").
2. Actions & Spatial Context: Describe the actions (e.g., pick, place, pour) including spatial transitions if visible (e.g., "from the table to the shelf", "out of the box").
3. Hands & Grasps: Mention specific hands ({' and '.join(active_hands)}) and their inferred grasp types.
4. Natural Grasp Names: Use the "natural" version of the grasp taxonomy (e.g., use "prismatic 3 finger" instead of "Prismatic_3_Finger"). Integrate it smoothly into the sentence.
5. Mandatory inclusion: You MUST explicitly mention the grasp type for each active hand.
6. Style: Imperative style (e.g., "Use the right hand to...").
7. Output format: Plain text only. NO JSON, NO explanations. ONE sentence only.

EXAMPLE:
Use the right hand with a prismatic 3 finger grasp to pick up the handle of a blue mug and use the left hand with a lateral grasp to steady the mug's base.
"""
        
        # Build context string based on hands_data
        context_str = ""
        
        for side in active_hands:
            info = hands_data[side]
            interaction = info['interaction']
            grasp_type = info['grasp_type']
            fine_grasp = grasp_type.split(":", 1)[1] if ":" in grasp_type else grasp_type
            # Create a more natural version for VLM to use in description
            natural_grasp = fine_grasp.replace("_", " ").lower()
            
            context_str += f"\n[{side.upper()} HAND Info]:\n"
            context_str += f"- Grasp Taxonomy (technical): {fine_grasp}\n"
            context_str += f"- Grasp Taxonomy (natural): {natural_grasp}\n"
            context_str += f"- Opposition: {interaction.opposition_type}\n"
            context_str += f"- Thumb Position: {interaction.thumb_position}\n"
            context_str += f"- Virtual Fingers: {interaction.virtual_fingers}\n"

        user_prompt = f"""Inferred Grasp Attributes for Active Hands:
{context_str}

Analyze the video and describe the task as requested.
"""

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )

        if debug is not None:
            debug.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": response_text,
                    "context_str": context_str,
                }
            )

        return response_text


class OmniDexReasoner:
    """
    Main framework integrating the three modules for Video-Only input.
    """
    def __init__(self, api_key: Optional[str] = None, trace: bool = False):
        self.trace = trace
        self.vlm_client = VLMClient(api_key=api_key)
        
        # Initialize modules with the VLM client
        self.interaction_model = HandObjectInteractionModel(self.vlm_client)
        self.taxonomy_model = GraspTaxonomyModel(self.vlm_client)
        self.description_generator = SemanticDescriptionGenerator(self.vlm_client)

    def reason(self, video_frames: List[Any], return_trace: bool = False) -> Any:
        """
        Main reasoning pipeline.
        
        Args:
            video_frames: A list of image data representing the first-person video.
        
        Returns:
            Dictionary containing the inferred semantics and final description.
        """
        
        trace_data: Dict[str, Any] = {"detect": {}, "hands": {}, "description": {}}

        # 1.0 Detect Active Hands
        detect_debug: Dict[str, Any] = {}
        active_hands = self.interaction_model.detect_active_hands(video_frames, debug=detect_debug)
        trace_data["detect"] = detect_debug
        if self.trace:
            logger.info(f"[Detect] Active hands: {active_hands}")
        
        hands_results = {}
        
        # If no active hands detected, fallback to right hand processing or return empty?
        # Let's fallback to 'right' to handle cases where detection fails but action exists.
        if not active_hands:
            active_hands = ["right"]

        for side in active_hands:
            # 1. Understand Interaction (VLM Call)
            interaction_debug: Dict[str, Any] = {}
            interaction_result = self.interaction_model.predict(video_frames, side, debug=interaction_debug)
            
            # 2. Reason about Grasp Taxonomy (VLM Call)
            taxonomy_debug: Dict[str, Any] = {}
            grasp_type = self.taxonomy_model.predict(video_frames, interaction_result, trace=self.trace, debug=taxonomy_debug)
            
            # Extract coarse category
            if ":" in grasp_type:
                coarse = grasp_type.split(":", 1)[0]
            else:
                coarse = "Unknown"
                
            hands_results[side] = {
                "interaction": interaction_result,
                "grasp_type": grasp_type,
                "grasp_category": coarse
            }

            trace_data["hands"][side] = {
                "interaction": interaction_debug,
                "taxonomy": taxonomy_debug,
            }

        # 3. Generate Semantic Description (VLM Call 3)
        description_debug: Dict[str, Any] = {}
        description = self.description_generator.generate(
            video_frames, 
            hands_results,
            active_hands=active_hands,
            debug=description_debug,
        )
        trace_data["description"] = description_debug
        
        if self.trace:
            logger.info("[Description] %s", description)
        
        # Transform hands_results to JSON-serializable dict
        serializable_hands = {}
        for side, data in hands_results.items():
            serializable_hands[side] = {
                "grasp_type": data["grasp_type"],
                "grasp_category": data["grasp_category"],
                "opposition_type": data["interaction"].opposition_type,
                "thumb_position": data["interaction"].thumb_position,
                "virtual_fingers": data["interaction"].virtual_fingers,
            }

        result = {
            "active_hands": active_hands,
            "hands": serializable_hands,
            "description": description,
        }

        if return_trace:
            return result, trace_data
        return result


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
    max_frames: int = 15,
) -> List[Any]:
    if decord is None or Image is None:
        raise RuntimeError(
            "Video frame extraction requires `decord` and `Pillow`. "
            "Install with: pip install decord pillow"
        )

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
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


def _process_video_sequential(
    reasoner: OmniDexReasoner,
    video_path: str,
    out_dir: str,
    max_workers: int = 4
) -> None:
    if decord is None or Image is None:
        raise RuntimeError("Sequential processing requires `decord` and `Pillow`.")

    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    except Exception as e:
        logger.error(f"Failed to open video {video_path}: {e}")
        return

    total_frames = len(vr)
    logger.info(f"Processing {video_path}: {total_frames} frames found. Parallel segments: {max_workers}")

    local_fps = 30
    
    # Pre-calculate all segment indices
    tasks = []
    current_sec = 0
    while True:
        base_idx = current_sec * local_fps
        idx_0 = base_idx
        idx_mid = base_idx + 15
        idx_end = base_idx + 30
        
        if idx_end >= total_frames:
            # Check edge case: if we have enough frames for a partial second?
            # Original code breaks here.
            break
            
        frame_indices = [idx_0, idx_mid, idx_end]
        tasks.append((current_sec, frame_indices))
        current_sec += 1

    if not tasks:
        logger.warning(f"No valid segments for {video_path}")
        return

    results = []
    vr_lock = threading.Lock()

    def _process_segment(sec: int, indices: List[int]) -> Optional[Dict]:
        try:
            # decord.VideoReader is not thread-safe for concurrent get_batch calls
            # on the same instance. Use a lock to protect the read.
            with vr_lock:
                frames_arr = vr.get_batch(indices).asnumpy()
            
            pil_frames = [Image.fromarray(f) for f in frames_arr]
            
            out, trace_data = reasoner.reason(pil_frames, return_trace=True)
            
            logger.info(f"  [Video {Path(video_path).name}] Analyzed sec {sec}")
            return {
                "second": sec,
                "frame_indices": indices,
                "analysis": out,
                "trace": trace_data,
            }
        except Exception as e:
            logger.error(f"Error processing second {sec} for {video_path}: {e}")
            return None

    # Execute efficiently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_segment, t[0], t[1]): t[0] for t in tasks}
        
        # Add tqdm for progress tracking across segments
        with tqdm(total=len(tasks), desc=f"Analyzing segments: {Path(video_path).name}", leave=False, dynamic_ncols=True) as pbar:
            for future in as_completed(future_map):
                res = future.result()
                if res:
                    results.append(res)
                pbar.update(1)
    
    # Sort results by second
    results.sort(key=lambda x: x["second"])

    if not results:
        logger.warning(f"No results generated for {video_path}")
        return

    # Split results into data and trace
    results_only = []
    traces_only = []
    for item in results:
        # Shallow copy to avoid modifying original if reused (though it's not)
        data_item = {k: v for k, v in item.items() if k != "trace"}
        trace_item = {"second": item["second"], "trace": item.get("trace")}
        results_only.append(data_item)
        traces_only.append(trace_item)

    # Save results
    stem = Path(video_path).stem
    out_path = Path(out_dir) / f"{stem}_sequential.json"
    trace_path = Path(out_dir) / f"{stem}_sequential_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_only, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sequential analysis to {out_path}")

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(traces_only, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sequential trace to {trace_path}")


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
        "--batch-dir",
        type=str,
        default=None,
        help="Directory containing multiple video files to process in batch.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="omnidex_outputs",
        help="Directory to write per-video JSON outputs (created if missing).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, search for videos recursively under --batch-dir.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="If set, log inputs/outputs of the three modules to terminal.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Enable sequential per-second processing (chunks of 300 frames, 15fps assumption).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a log file to write runtime logs (uses rotating handler).",
    )
    parser.add_argument(
        "--vlm-log-file",
        type=str,
        default=None,
        help="Path to a dedicated log file for VLM prompts and raw responses.",
    )
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=1,
        help="Number of parallel videos to process in batch mode.",
    )
    parser.add_argument(
        "--seg-workers",
        type=int,
        default=4,
        help="Number of parallel segments to process per video (sequential mode).",
    )
    args = parser.parse_args()

    # Configure file logging if requested
    if args.log_file:
        try:
            from logging.handlers import RotatingFileHandler

            log_dir = Path(args.log_file).parent
            if str(log_dir) and not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)

            fh = RotatingFileHandler(str(args.log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
            fh.setLevel(logging.DEBUG if args.trace else logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
            # Avoid adding duplicate handlers if main() called multiple times
            has_fh = any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == str(args.log_file) for h in logger.handlers)
            if not has_fh:
                logging.getLogger().addHandler(fh)
        except Exception:
            logger.exception(f"Failed to initialize log file handler: {args.log_file}")

    # Configure VLM-specific logging if requested
    if args.vlm_log_file:
        try:
            from logging.handlers import RotatingFileHandler
            vlm_log_dir = Path(args.vlm_log_file).parent
            if str(vlm_log_dir) and not vlm_log_dir.exists():
                vlm_log_dir.mkdir(parents=True, exist_ok=True)
            vfh = RotatingFileHandler(str(args.vlm_log_file), maxBytes=20 * 1024 * 1024, backupCount=5, encoding='utf-8')
            vfh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            vlm_logger.addHandler(vfh)
            logger.info(f"VLM conversation logs will be written to {args.vlm_log_file}")
        except Exception:
            logger.exception(f"Failed to initialize VLM log file: {args.vlm_log_file}")

    # If trace requested, set root logger to DEBUG for console output
    if args.trace:
        logging.getLogger().setLevel(logging.DEBUG)

    # Replace default handler with TqdmLoggingHandler for cleaner progress bars
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    th = TqdmLoggingHandler()
    th.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.root.addHandler(th)

    # Initialize Reasoner (will be shared across threads if thread-safe, or instantiated per thread if needed)
    # Since VLMClient uses openai client which is thread safe, sharing reasoner is fine.
    # However, for maximum safety if reasoner has state (it doesn't seem to have per-call state that persists), we share it.
    reasoner = OmniDexReasoner(trace=args.trace)

    def _process_and_save(frames: List[Any], name: str, out_dir: str):
        try:
            res, trace_data = reasoner.reason(frames, return_trace=True)
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            file_name = f"{name}.json"
            with open(out_path / file_name, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote output for {name} -> {out_path / file_name}")

            trace_name = f"{name}_trace.json"
            with open(out_path / trace_name, "w", encoding="utf-8") as f_trace:
                json.dump({"name": name, "trace": trace_data}, f_trace, ensure_ascii=False, indent=2)
            logger.info(f"Wrote trace log for {name} -> {out_path / trace_name}")
        except Exception:
            logger.exception(f"Failed processing {name}")

    # Helper for batch video processing
    def process_single_video(vid_path: Path):
        try:
            if args.sequential:
                 _process_video_sequential(reasoner, str(vid_path), args.out_dir, max_workers=args.seg_workers)
            else:
                 frames = _extract_frames_from_video(str(vid_path), max_frames=args.max_frames)
                 _process_and_save(frames, vid_path.stem, args.out_dir)
        except Exception:
            logger.exception(f"Failed to process video {vid_path}")

    # Batch mode: process multiple video files under --batch-dir
    if args.batch_dir:
        base = Path(args.batch_dir)
        if not base.exists():
            raise FileNotFoundError(f"Batch directory not found: {args.batch_dir}")
        pattern = "**/*.mp4" if args.recursive else "*.mp4"
        files = sorted(base.glob(pattern))
        if not files:
            # try other extensions
            pattern2 = "**/*.avi" if args.recursive else "*.avi"
            files = sorted(base.glob(pattern2))
        if not files:
            logger.warning(f"No video files found in {args.batch_dir}")
            return
            
        logger.info(f"Found {len(files)} videos. Processing with {args.batch_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=args.batch_workers) as executor:
            # Wrap files in tqdm for video-level progress
            futures = {executor.submit(process_single_video, vid): vid for vid in files}
            with tqdm(total=len(files), desc="Overall Progress", dynamic_ncols=True) as pbar:
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        vid = futures[f]
                        logger.error(f"Batch worker error on {vid}: {e}")
                    pbar.update(1)
        return

    # Single source modes
    video_frames: List[Any] = []
    if args.video_path:
        if args.sequential:
             _process_video_sequential(reasoner, args.video_path, args.out_dir, max_workers=args.seg_workers)
        else:
             video_frames = _extract_frames_from_video(
                args.video_path,
                max_frames=args.max_frames,
             )
             _process_and_save(video_frames, Path(args.video_path).stem, args.out_dir)
    elif args.video_dir:
        resolved_path = _resolve_video_from_dir(args.video_dir)
        if args.sequential:
             _process_video_sequential(reasoner, resolved_path, args.out_dir, max_workers=args.seg_workers)
        else:
            video_frames = _extract_frames_from_video(
                resolved_path,
                max_frames=args.max_frames,
            )
            _process_and_save(video_frames, Path(resolved_path).stem, args.out_dir)
    elif args.frames_dir:
        frames_list = _load_frames_from_dir(args.frames_dir, max_frames=args.max_frames)
        _process_and_save(frames_list, Path(args.frames_dir).name, args.out_dir)


if __name__ == "__main__":
    main()

"""
python src/openpi/reasoner/omnidex_reasoner.py \
    --video-dir PickPlaceBottle/PickPlaceBottle_Merged_v4/videos/chunk-000/observation.camera_0.rgb/ \
    --max-frames 15
    
python src/openpi/reasoner/omnidex_reasoner.py \
    --batch-dir PickPlaceBottle/PickPlaceBottle_Merged_v4/videos/chunk-000/observation.camera_0.rgb/ \
    --out-dir ./omni_outputs \
    --max-frames 15 --recursive
    
python src/openpi/reasoner/omnidex_reasoner.py \
    --video-dir egodex/ \
    --out-dir ./omni_outputs/egodex0_test_1/ \
    --sequential \
    --seg-workers 10 \
    --log-file ./omni_outputs/egodex0_test_1/omnidex_log.txt
"""