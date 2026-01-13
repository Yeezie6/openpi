import os
import argparse
import base64
import importlib.util
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
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
        return response.choices[0].message.content.strip()


# ==========================================
# Lightweight Grasp Knowledge Base (for RAG)
# ==========================================

class GraspKnowledgeBase:
    """In-memory grasp taxonomy (33 classes) used for RAG grounding."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, str]] = [
            self._entry("Power", "Large Diameter", "Palm", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Power", "Small Diameter", "Palm", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Power", "Medium Wrap", "Palm", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Power", "Adducted Thumb", "Palm", "Add", "VF2: 2-5; VF3: 1"),
            self._entry("Power", "Light Tool", "Palm", "Add", "VF2: 2-5; VF3: (1)"),
            self._entry("Precision", "Prismatic 4 Finger", "Pad", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Precision", "Prismatic 3 Finger", "Pad", "Abd", "VF2: 2-4; VF3: none"),
            self._entry("Precision", "Prismatic 2 Finger", "Pad", "Abd", "VF2: 2-3; VF3: none"),
            self._entry("Precision", "Palmar Pinch", "Pad", "Abd", "VF2: 2; VF3: none"),
            self._entry("Power", "Power Disk", "Palm", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Power", "Power Sphere", "Palm", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Precision", "Precision Disk", "Pad", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Precision", "Precision Sphere", "Pad", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Precision", "Tripod", "Pad", "Abd", "VF2: 2-3; VF3: none"),
            self._entry("Power", "Fixed Hook", "Palm", "Add", "VF2: 2-5; VF3: none"),
            self._entry("Intermediate", "Lateral", "Side", "Add", "VF2: 2; VF3: none"),
            self._entry("Power", "Index Finger Extension", "Palm", "Add", "VF2: 3-5; VF3: 2"),
            self._entry("Power", "Extension Type", "Pad", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Power", "Distal Type", "Pad", "Abd", "VF2: 2-5; VF3: none"),
            self._entry("Precision", "Writing Tripod", "Side", "Abd", "VF2: 2; VF3: none"),
            self._entry("Intermediate", "Tripod Variation", "Side", "Abd", "VF2: 3-4; VF3: none"),
            self._entry("Precision", "Parallel Extension", "Pad", "Add", "VF2: 2-5; VF3: none"),
            self._entry("Intermediate", "Adduction Grip", "Side", "Abd", "VF2: 2; VF3: none"),
            self._entry("Precision", "Tip Pinch", "Pad", "Abd", "VF2: 2; VF3: none"),
            self._entry("Intermediate", "Lateral Tripod", "Side", "Add", "VF2: 3; VF3: none"),
            self._entry("Power", "Sphere 4 Finger", "Pad", "Abd", "VF2: 2-4; VF3: none"),
            self._entry("Precision", "Quadpod", "Pad", "Abd", "VF2: 2-4; VF3: none"),
            self._entry("Power", "Sphere 3 Finger", "Pad", "Abd", "VF2: 2-3; VF3: none"),
            self._entry("Intermediate", "Stick", "Side", "Add", "VF2: 2; VF3: none"),
            self._entry("Power", "Palmar", "Palm", "Add", "VF2: 2-5; VF3: none"),
            self._entry("Power", "Ring", "Pad", "Abd", "VF2: 2; VF3: none"),
            self._entry("Intermediate", "Ventral", "Side", "Add", "VF2: 2; VF3: none"),
            self._entry("Precision", "Inferior Pincer", "Pad", "Abd", "VF2: 2; VF3: none"),
        ]

    @staticmethod
    def _entry(coarse: str, fine: str, opposition: str, thumb: str, vf: str) -> Dict[str, str]:
        fine_slug = fine.replace(" ", "_")
        return {
            "name": f"{coarse}:{fine_slug}",
            "coarse": coarse,
            "fine": fine,
            "opposition": opposition,
            "thumb": thumb,
            "virtual_fingers": vf,
        }

    @staticmethod
    def _score(entry: Dict[str, str], text: str) -> int:
        text_l = text.lower()
        score = 0
        for field in ("name", "coarse", "fine", "opposition", "thumb", "virtual_fingers"):
            val = entry.get(field, "").lower()
            tokens = val.replace(":", " ").replace(";", " ").replace("-", " ").split()
            for tok in tokens:
                if tok and tok in text_l:
                    score += 1
        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not query:
            return self.entries[:top_k]
        scored = [
            (self._score(entry, query), entry)
            for entry in self.entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scored[:top_k] if s > 0] or [e for _, e in scored[:top_k]]


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

    def detect_active_hands(self, video_frames: List[Any], trace: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Step 1.0: Identify which hands (Left/Right) are actively manipulating objects.
        """
        system_prompt = """You are an expert in analyzing hand-object interactions in first-person videos.
The left side of the video frame corresponds to the Left Hand.
The right side of the video frame corresponds to the Right Hand.
"""
        user_prompt = """
Analyze the video frames and determine which hands are actively manipulating an object.
Ignore hands that are visible but NOT interacting with any object.

Output a single JSON object with a key "active_hands" containing a list of strings.
Allowed values in the list: "left", "right".

Examples:
- If only the right hand is holding/moving an object: {"active_hands": ["right"]}
- If only the left hand is holding/moving an object: {"active_hands": ["left"]}
- If both hands are manipulating objects: {"active_hands": ["left", "right"]}
- If no hands are manipulating: {"active_hands": []}
"""
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
        if not parsed_active:
            parsed_active = ["right"]

        if trace is not None:
            trace.append({
                "step": "detect_active_hands",
                "hand": "both",
                "frame_count": len(video_frames),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": response_text,
                "parsed_active_hands": parsed_active,
            })

        return parsed_active

    def predict(self, video_frames: List[Any], hand_side: str = "right", trace: Optional[List[Dict[str, Any]]] = None) -> InteractionOutput:
        system_prompt = f"""You are an expert in robotic grasping and hand taxonomy. 
Focus ONLY on the {hand_side.upper()} HAND (visible on the {hand_side} side of the image) and the grasp taxonomy signals (Opposition, Thumb posture, Virtual Fingers)."""
        
        user_prompt = f"""
You are a grasp-attribute annotator. Given a sequence of video frames showing a hand interacting with an object, infer ONLY the following grasp attributes for the {hand_side.upper()} HAND:

1) Opposition type:
   - Palm: the object is opposed against the palm / palm wrap / containment.
   - Pad: opposition mainly via finger pads / fingertips (pad-to-pad or thumb pad to finger pad).
   - Side: lateral/side pinch where finger side(s) oppose (e.g., key pinch).

2) Thumb position:
   - Abd: thumb abducted/open (clearly away from palm, large thumb-index spread).
   - Add: thumb adducted/closer to palm (thumb tucked in, small thumb-index spread).

3) Virtual fingers (VF):
   - Identify which digits of the {hand_side.upper()} HAND are effectively acting together.
   - Use digit IDs: 1=thumb, 2=index, 3=middle, 4=ring, 5=little.
   - Output format examples:
     - "VF2: 1 vs 2-5"
     - "VF2: 1 vs 2-3"
     - "VF3: 2-3, (4) optional"
   - Use parentheses for occasional/optional contact that is not consistently present across frames.

CoT requirement (IMPORTANT):
- Think step-by-step in a private scratchpad to integrate evidence across frames (contact locations, palm involvement, digit engagement, thumb spread, object stability).
- DO NOT reveal your step-by-step reasoning.
- Use temporal consistency: prefer the dominant/stable grasp during the hold/manipulation phase; ignore brief transitions unless no stable grasp exists.

Uncertainty rules:
- If the key evidence is occluded, too small, motion-blurred, or contradictory across frames, use "Unknown" for that field.
- If the grasp changes mid-clip, label the grasp that is most stable/longest during object control. If none is stable, use "Unknown" where appropriate.

Output rules (STRICT):
- Output ONE valid JSON object ONLY (no extra text, no markdown), with ALL keys present:
{{
  "opposition_type": "Palm|Pad|Side|Unknown",
  "thumb_position": "Abd|Add|Unknown",
  "virtual_fingers": "<text description or Unknown>"
}}
"""


        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )

        # Debug: Log the raw VLM response
        logger.debug(f"VLM Response: {response_text}")
        
        try:
            # Try to robustly extract a JSON object from the response using regex.
            import re

            m = re.search(r"\{.*\}", response_text, re.DOTALL)

            if m:
                json_text = m.group(0)
                data = json.loads(json_text)
                out = InteractionOutput(
                    opposition_type=data.get("opposition_type", "Unknown"),
                    thumb_position=data.get("thumb_position", "Unknown"),
                    virtual_fingers=data.get("virtual_fingers", "Unknown"),
                    raw_response=response_text,
                )
                if trace is not None:
                    trace.append({
                        "step": "interaction_model",
                        "hand": hand_side,
                        "frame_count": len(video_frames),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "raw_response": response_text,
                        "parsed": asdict(out),
                    })
                return out

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
                out = InteractionOutput(
                    opposition_type="Error",
                    thumb_position="Error",
                    virtual_fingers="Error",
                    raw_response=response_text,
                )
                if trace is not None:
                    trace.append({
                        "step": "interaction_model",
                        "hand": hand_side,
                        "frame_count": len(video_frames),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "raw_response": response_text,
                        "parsed": asdict(out),
                    })
                return out

            out = InteractionOutput(
                opposition_type=opposition_type,
                thumb_position=thumb_position,
                virtual_fingers=virtual_fingers,
                raw_response=response_text,
            )
            if trace is not None:
                trace.append({
                    "step": "interaction_model",
                    "hand": hand_side,
                    "frame_count": len(video_frames),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": response_text,
                    "parsed": asdict(out),
                })
            return out
        except Exception as e:
            logger.exception("Unhandled exception while parsing VLM response")
            out = InteractionOutput(
                opposition_type="Error",
                thumb_position="Error",
                virtual_fingers="Error",
                raw_response=response_text,
            )
            if trace is not None:
                trace.append({
                    "step": "interaction_model",
                    "hand": hand_side,
                    "frame_count": len(video_frames),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": response_text,
                    "parsed": asdict(out),
                })
            return out


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

    def predict(self, video_frames: List[Any], interaction_context: InteractionOutput, trace: bool = False, trace_log: Optional[List[Dict[str, Any]]] = None) -> str:
        system_prompt = """You are an expert in Grasp Taxonomy. 
Use the provided knowledge snippets to classify the grasp type.
Return the final grasp as Coarse:Fine."""

        query_text = (
            f"Opposition: {interaction_context.opposition_type}\n"
            f"Thumb Position: {interaction_context.thumb_position}\n"
            f"Virtual Fingers: {interaction_context.virtual_fingers}"
        )
        knowledge_ctx = self._build_knowledge_context(query_text)

        # Build allowed labels from the small knowledge base and include them in the prompt
        allowed = [e.get('name') for e in self.knowledge_base.entries if e.get('name')]
        allowed_list_text = ', '.join(allowed)

        user_prompt = f"""Context from video frames:
    - Opposition Type: {interaction_context.opposition_type}
    - Thumb Position: {interaction_context.thumb_position}
    - Virtual Fingers: {interaction_context.virtual_fingers}

    Reference grasp knowledge (retrieved):
    {knowledge_ctx}

    Allowed grasp labels (choose exactly one): {allowed_list_text}

    Reason step by step:
    1) Use the provided grasp attributes + knowledge snippets as evidence.
    2) List 2-3 candidate grasp types with brief justification.
    3) Pick the best and output ONLY one line in this exact format (must be one of the allowed labels):
       Final Grasp: Coarse:Fine
    """

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )

        if trace:
            logger.info("[Taxonomy] Prompted with knowledge:\n%s", knowledge_ctx)
            logger.info("[Taxonomy] Raw VLM response: %s", response_text)

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
                return f"{coarse}:{fine}"

            m_secondary = re.search(r"\b(Power|Precision|Intermediate)\b[\s\-:]*([A-Za-z0-9_\-]+)?", label_text, re.IGNORECASE)
            if m_secondary:
                coarse = m_secondary.group(1).capitalize()
                fine = (m_secondary.group(2) or "").strip()
                if fine:
                    return f"{coarse}:{fine}"
                return f"{coarse}:Unknown"
            return None

        parsed = _extract(normalized_candidate) or _extract(normalized_full)
        final_result: str

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
                    final_result = f"{coarse}:{fine}"
                else:
                    # Fuzzy match fine label to known fines
                    close = difflib.get_close_matches(fine, fines_for_coarse, n=1, cutoff=0.6) if fines_for_coarse else []
                    if close:
                        final_result = f"{coarse}:{close[0]}"
                    else:
                        # If fine unknown, return coarse:Unknown to avoid invented labels
                        final_result = f"{coarse}:Unknown"
            except Exception:
                logger.exception("Error validating parsed grasp_type")
                final_result = parsed
        else:
            cleaned = re.sub(r"^[^A-Za-z0-9]+", "", normalized_candidate)
            if ":" in cleaned:
                final_result = cleaned
            else:
                logger.warning(f"Unexpected grasp_type format. Response: {response_text}")
                final_result = "Error: Invalid Format"

        if trace_log is not None:
            try:
                trace_log.append({
                    "step": "taxonomy_model",
                    "frame_count": len(video_frames),
                    "interaction_context": asdict(interaction_context),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "knowledge_context": knowledge_ctx,
                    "allowed_labels": allowed,
                    "raw_response": response_text,
                    "parsed_grasp": parsed,
                    "final_grasp": final_result,
                })
            except Exception:
                logger.exception("Failed to record taxonomy trace")

        return final_result


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
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        system_prompt = """You are a video annotation AI specialized in identifying high-level manipulation tasks performed by human hands in first-person videos with a series of grasp information."""
        
        # Build context string based on hands_data
        context_str = ""
        active_hands = sorted(hands_data.keys())
        
        for side in active_hands:
            info = hands_data[side]
            interaction = info['interaction']
            grasp_type = info['grasp_type']
            fine_grasp = grasp_type.split(":", 1)[1] if ":" in grasp_type else grasp_type
            
            context_str += f"\n[{side.upper()} HAND Info]:\n"
            context_str += f"- Grasp Taxonomy: {fine_grasp}\n"
            context_str += f"- Opposition: {interaction.opposition_type}\n"
            context_str += f"- Thumb Position: {interaction.thumb_position}\n"
            context_str += f"- Virtual Fingers: {interaction.virtual_fingers}\n"

        user_prompt = f"""
Input Context (Inferred Grasp Attributes for Active Hands):
{context_str}

Task:
Generate a single-sentence task description that summarizes the main manipulation task performed in the video.

The task description MUST include:
1. The objects being manipulated, including distinctive features.
2. The specific actions (e.g., pick, place, pour, open) with start/end locations if applicable.
3. Mention specific hands ({' and '.join(active_hands)}) and their grasp types (from context).

Guidelines:
- If only one hand is active, describe its action clearly.
- If TWO hands are active, describe how they coordinate or what each is doing.
- Use imperative style (e.g., "Use the right hand to... and the left hand to...").
- Assume real human hands.
- One sentence only.

Output format:
Return plain text only. Do not include explanations or JSON.
"""

        response_text = self.vlm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_frames=video_frames
        )
        if trace is not None:
            hands_context = {}
            for side_key, info in hands_data.items():
                interaction_obj = info.get("interaction")
                interaction_payload = asdict(interaction_obj) if hasattr(interaction_obj, "__dataclass_fields__") else interaction_obj
                hands_context[side_key] = {
                    "grasp_type": info.get("grasp_type"),
                    "grasp_category": info.get("grasp_category"),
                    "interaction": interaction_payload,
                }

            trace.append({
                "step": "description_generator",
                "frame_count": len(video_frames),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": response_text,
                "active_hands": active_hands,
                "hands_context": hands_context,
            })

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

    def reason(self, video_frames: List[Any]) -> Dict[str, Any]:
        """
        Main reasoning pipeline.
        
        Args:
            video_frames: A list of image data representing the first-person video.
        
        Returns:
            Dictionary containing the inferred semantics and final description.
        """
        
        trace_log: List[Dict[str, Any]] = []

        # 1.0 Detect Active Hands
        active_hands = self.interaction_model.detect_active_hands(video_frames, trace=trace_log)
        if self.trace:
            logger.info(f"[Detect] Active hands: {active_hands}")
        
        hands_results = {}
        
        # If no active hands detected, fallback to right hand processing or return empty?
        # Let's fallback to 'right' to handle cases where detection fails but action exists.
        if not active_hands:
            active_hands = ["right"]

        for side in active_hands:
            # 1. Understand Interaction (VLM Call)
            interaction_result = self.interaction_model.predict(video_frames, side, trace=trace_log)
            
            # 2. Reason about Grasp Taxonomy (VLM Call)
            grasp_type = self.taxonomy_model.predict(video_frames, interaction_result, trace=self.trace, trace_log=trace_log)
            
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

        # 3. Generate Semantic Description (VLM Call 3)
        description = self.description_generator.generate(
            video_frames, 
            hands_results,
            trace=trace_log
        )
        
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

        return {
            "active_hands": active_hands,
            "hands": serializable_hands,
            "description": description,
            "trace_log": trace_log,
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
            
            out = reasoner.reason(pil_frames)
            
            logger.info(f"  [Video {Path(video_path).name}] Analyzed sec {sec}")
            return {
                "second": sec,
                "frame_indices": indices,
                "analysis": out
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

    # Save results
    stem = Path(video_path).stem
    out_path = Path(out_dir) / f"{stem}_sequential.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sequential analysis to {out_path}")

    # Save segment-level trace logs separately
    trace_entries = [
        {
            "second": r.get("second"),
            "frame_indices": r.get("frame_indices"),
            "trace_log": r.get("analysis", {}).get("trace_log", []),
        }
        for r in results
    ]
    trace_path = Path(out_dir) / f"{stem}_sequential_trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_entries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sequential trace log to {trace_path}")


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
            res = reasoner.reason(frames)
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            file_name = f"{name}.json"
            with open(out_path / file_name, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote output for {name} -> {out_path / file_name}")

            # Persist trace log separately for easier inspection
            trace_data = res.get("trace_log", [])
            trace_name = f"{name}_trace.json"
            with open(out_path / trace_name, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote trace log for {name} -> {out_path / trace_name}")
        except Exception:
            logger.exception(f"Failed processing {name}")

    # Helper for batch video processing
    def process_single_video(vid_path: Path):
        try:
            if args.sequential:
                 _process_video_sequential(reasoner, str(vid_path), args.out_dir, max_workers=args.seg_workers)
            else:
                 frames = _extract_frames_from_video(str(vid_path), extract_fps=args.extract_fps, max_frames=args.max_frames)
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
                extract_fps=args.extract_fps,
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
                extract_fps=args.extract_fps,
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
    --out-dir ./omni_outputs/egodex0_50/ \
    --max-frames 15
"""