from dataclasses import dataclass
from typing import List, Dict, Optional
import re
import random


@dataclass(frozen=True)
class KBEntry:
    name: str
    coarse: str
    fine: str
    opposition: str
    thumb: str
    vf2: str
    vf3: str
    object_shape: str
    contact_region: str
    size_hint: str
    key_cues: str
    avoid_if: str
    examples: str

    @property
    def hand_signature(self) -> str:
        return f"{self.coarse}|{self.opposition}|{self.thumb}|VF2={self.vf2}|VF3={self.vf3}"


class GraspKnowledgeBase:
    """Confusion-aware in-memory grasp KB for contrast-set RAG."""

    def __init__(self) -> None:
        self.entries: List[KBEntry] = [
            # High-confusion cluster with explicit disambiguation cues
            self._entry(
                "Power",
                "Large Diameter",
                "Palm",
                "Abd",
                "2-5",
                "none",
                object_shape="cylinder",
                contact_region="sidewall",
                size_hint="large",
                key_cues="Wrap around a thick cylindrical/cuboid body; broad palm contact; fingers spread around sidewall.",
                avoid_if="Object is a thin plate/lid (disk) or near-spherical; contact mainly on rim/face or uniformly on a round surface.",
                examples="bottle body, cup body, thick tool handle",
            ),
            self._entry(
                "Power",
                "Small Diameter",
                "Palm",
                "Abd",
                "2-5",
                "none",
                object_shape="cylinder",
                contact_region="sidewall",
                size_hint="small",
                key_cues="Cylindrical body is narrow; fingers can overlap/close deeply; strong wrap with high finger flexion.",
                avoid_if="Flat disk/plate/lid OR sphere-like object; if aperture looks wide, prefer Large/Medium.",
                examples="thin bottle, marker, narrow handle",
            ),
            self._entry(
                "Power",
                "Medium Wrap",
                "Palm",
                "Abd",
                "2-5",
                "none",
                object_shape="cylinder",
                contact_region="sidewall",
                size_hint="medium",
                key_cues="Cylindrical/cuboid body; wrap is clear but not extreme large/small; moderate aperture and finger flexion.",
                avoid_if="Thin disk/plate/lid OR near-spherical; if object obviously very thick, prefer Large; if very narrow, prefer Small.",
                examples="mug body, medium bottle, moderate handle",
            ),
            self._entry(
                "Power",
                "Power Disk",
                "Palm",
                "Abd",
                "2-5",
                "none",
                object_shape="disk",
                contact_region="rim",
                size_hint="unknown",
                key_cues="Object is plate/lid/disk-like; hand stabilizes or twists a flat object; contact often on rim/edge and/or face.",
                avoid_if="Main object is cylindrical sidewall wrap OR uniformly round sphere.",
                examples="plate, lid, flat cap, disk-like cover",
            ),
            self._entry(
                "Power",
                "Power Sphere",
                "Palm",
                "Abd",
                "2-5",
                "none",
                object_shape="sphere",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Object is sphere-like; palm + fingers envelop a round surface from multiple directions; no dominant 'sidewall' axis.",
                avoid_if="Clear cylindrical sidewall wrap OR thin flat disk.",
                examples="ball, apple/orange, round knob",
            ),
            # Remaining entries with placeholder disambiguation cues
            self._entry(
                "Power",
                "Adducted Thumb",
                "Palm",
                "Add",
                "2-5",
                "1",
                object_shape="unknown",
                contact_region="unknown",
                size_hint="unknown",
                key_cues="Thumb is adducted; thumb stabilizes against palm while VF2 wraps.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Power",
                "Light Tool",
                "Palm",
                "Add",
                "2-5",
                "(1)",
                object_shape="unknown",
                contact_region="unknown",
                size_hint="unknown",
                key_cues="Power-like grasp on a light tool; thumb may optionally contact.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Precision",
                "Prismatic 4 Finger",
                "Pad",
                "Abd",
                "2-5",
                "none",
                object_shape="prismatic",
                contact_region="edge",
                size_hint="unknown",
                key_cues="Prismatic object; fingertip pads align on parallel faces/edges.",
                avoid_if="",
                examples="box edge, small block",
            ),
            self._entry(
                "Precision",
                "Prismatic 3 Finger",
                "Pad",
                "Abd",
                "2-4",
                "none",
                object_shape="prismatic",
                contact_region="edge",
                size_hint="unknown",
                key_cues="3-finger prismatic pinch; pads oppose on edges/faces.",
                avoid_if="",
                examples="small block",
            ),
            self._entry(
                "Precision",
                "Prismatic 2 Finger",
                "Pad",
                "Abd",
                "2-3",
                "none",
                object_shape="prismatic",
                contact_region="edge",
                size_hint="unknown",
                key_cues="2-finger prismatic pinch; fine control on edges.",
                avoid_if="",
                examples="thin block",
            ),
            self._entry(
                "Precision",
                "Palmar Pinch",
                "Pad",
                "Abd",
                "2",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="unknown",
                key_cues="Thumb + index pad pinch with palmar stabilization.",
                avoid_if="",
                examples="coin, small part",
            ),
            self._entry(
                "Precision",
                "Precision Disk",
                "Pad",
                "Abd",
                "2-5",
                "none",
                object_shape="disk",
                contact_region="rim",
                size_hint="unknown",
                key_cues="Precision handling of disk-like object; pads dominate.",
                avoid_if="If power wrap/palm dominates, prefer Power Disk.",
                examples="lid, coin-like",
            ),
            self._entry(
                "Precision",
                "Precision Sphere",
                "Pad",
                "Abd",
                "2-5",
                "none",
                object_shape="sphere",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Precision handling of a small sphere-like object using pads.",
                avoid_if="If palm wraps strongly, prefer Power Sphere.",
                examples="small ball",
            ),
            self._entry(
                "Precision",
                "Tripod",
                "Pad",
                "Abd",
                "2-3",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Thumb opposes index+middle pads; 3-point precision control.",
                avoid_if="",
                examples="pen tip, small cap",
            ),
            self._entry(
                "Power",
                "Fixed Hook",
                "Palm",
                "Add",
                "2-5",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="unknown",
                key_cues="Hook-like support; load-bearing with palm/ulnar side.",
                avoid_if="",
                examples="bag handle",
            ),
            self._entry(
                "Intermediate",
                "Lateral",
                "Side",
                "Add",
                "2",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Lateral pinch using thumb pad vs index side.",
                avoid_if="",
                examples="key, card",
            ),
            self._entry(
                "Power",
                "Index Finger Extension",
                "Palm",
                "Add",
                "3-5",
                "2",
                object_shape="unknown",
                contact_region="face",
                size_hint="unknown",
                key_cues="Index extends for interaction while other fingers stabilize.",
                avoid_if="",
                examples="button press",
            ),
            self._entry(
                "Power",
                "Extension Type",
                "Pad",
                "Abd",
                "2-5",
                "none",
                object_shape="unknown",
                contact_region="face",
                size_hint="unknown",
                key_cues="Pad-dominant extension; interaction on face/plane.",
                avoid_if="",
                examples="press surface",
            ),
            self._entry(
                "Power",
                "Distal Type",
                "Pad",
                "Abd",
                "2-5",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="unknown",
                key_cues="Distal contact focus; pad contact near fingertips.",
                avoid_if="",
                examples="small edge",
            ),
            self._entry(
                "Precision",
                "Writing Tripod",
                "Side",
                "Abd",
                "2",
                "none",
                object_shape="stick",
                contact_region="surface",
                size_hint="small",
                key_cues="Pen-like manipulation; side contact and stable tripod.",
                avoid_if="",
                examples="pen/pencil",
            ),
            self._entry(
                "Intermediate",
                "Tripod Variation",
                "Side",
                "Abd",
                "3-4",
                "none",
                object_shape="unknown",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Tripod-like but side-biased; variation in supporting digits.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Precision",
                "Parallel Extension",
                "Pad",
                "Add",
                "2-5",
                "none",
                object_shape="unknown",
                contact_region="face",
                size_hint="unknown",
                key_cues="Pads aligned; thumb adducted; parallel extension posture.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Intermediate",
                "Adduction Grip",
                "Side",
                "Abd",
                "2",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="unknown",
                key_cues="Side grip with adduction-like stabilization.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Precision",
                "Tip Pinch",
                "Pad",
                "Abd",
                "2",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Thumb tip opposes index tip; minimal contact area.",
                avoid_if="",
                examples="needle, small pin",
            ),
            self._entry(
                "Intermediate",
                "Lateral Tripod",
                "Side",
                "Add",
                "3",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Tripod-like but lateral; thumb adducted.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Power",
                "Sphere 4 Finger",
                "Pad",
                "Abd",
                "2-4",
                "none",
                object_shape="sphere",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Sphere-like with 4 fingers; pads dominate.",
                avoid_if="",
                examples="round knob",
            ),
            self._entry(
                "Precision",
                "Quadpod",
                "Pad",
                "Abd",
                "2-4",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Thumb + 3 fingers precision; 4-point control.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Power",
                "Sphere 3 Finger",
                "Pad",
                "Abd",
                "2-3",
                "none",
                object_shape="sphere",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Sphere-like with 3 fingers; pad dominant.",
                avoid_if="",
                examples="small ball",
            ),
            self._entry(
                "Intermediate",
                "Stick",
                "Side",
                "Add",
                "2",
                "none",
                object_shape="stick",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Stick/rod-like object stabilized by side contact.",
                avoid_if="",
                examples="rod, screwdriver",
            ),
            self._entry(
                "Power",
                "Palmar",
                "Palm",
                "Add",
                "2-5",
                "none",
                object_shape="unknown",
                contact_region="face",
                size_hint="unknown",
                key_cues="Palmar stabilization; thumb adducted with palm dominance.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Power",
                "Ring",
                "Pad",
                "Abd",
                "2",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Ring-like holding; pad contact with index dominance.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Intermediate",
                "Ventral",
                "Side",
                "Add",
                "2",
                "none",
                object_shape="unknown",
                contact_region="surface",
                size_hint="unknown",
                key_cues="Ventral side contact; thumb adducted.",
                avoid_if="",
                examples="",
            ),
            self._entry(
                "Precision",
                "Inferior Pincer",
                "Pad",
                "Abd",
                "2",
                "none",
                object_shape="unknown",
                contact_region="edge",
                size_hint="small",
                key_cues="Inferior pincer; pad pinch from below/under-side.",
                avoid_if="",
                examples="",
            ),
        ]

    @staticmethod
    def _entry(
        coarse: str,
        fine: str,
        opposition: str,
        thumb: str,
        vf2: str,
        vf3: str,
        object_shape: str,
        contact_region: str,
        size_hint: str,
        key_cues: str,
        avoid_if: str,
        examples: str,
    ) -> KBEntry:
        fine_slug = fine.replace(" ", "_")
        return KBEntry(
            name=f"{coarse}:{fine_slug}",
            coarse=coarse,
            fine=fine,
            opposition=opposition,
            thumb=thumb,
            vf2=vf2,
            vf3=vf3,
            object_shape=object_shape,
            contact_region=contact_region,
            size_hint=size_hint,
            key_cues=key_cues,
            avoid_if=avoid_if,
            examples=examples,
        )

    @staticmethod
    def _parse_query(query: str) -> Dict[str, str]:
        q = query.lower()

        def _grab(pat: str) -> str:
            m = re.search(pat, q)
            return m.group(1) if m else ""

        return {
            "opposition": _grab(r"opposition\s*:\s*(palm|pad|side)"),
            "thumb": _grab(r"thumb\s*position\s*:\s*(abd|add)"),
            "vf2": _grab(r"vf2\s*:\s*([0-9\-\s]+)"),
            "vf3": _grab(r"vf3\s*:\s*([0-9\(\)]+|none)"),
        }

    @staticmethod
    def _match_signature(e: KBEntry, q: Dict[str, str]) -> bool:
        ok = True
        if q.get("opposition"):
            ok &= e.opposition.lower() == q["opposition"]
        if q.get("thumb"):
            ok &= e.thumb.lower() == q["thumb"]
        if q.get("vf2"):
            vf2_norm = q["vf2"].replace(" ", "")
            ok &= e.vf2.replace(" ", "") == vf2_norm
        if q.get("vf3"):
            vf3_norm = q["vf3"].replace(" ", "")
            ok &= e.vf3.replace(" ", "") == vf3_norm
        return ok

    @staticmethod
    def _score(e: KBEntry, query: str) -> float:
        q = query.lower()
        s = 0.0
        for tok in [e.coarse, e.fine, e.opposition, e.thumb, e.object_shape, e.contact_region]:
            if tok and tok.lower() in q:
                s += 1.0
        for w in e.examples.lower().split(","):
            w = w.strip()
            if w and w in q:
                s += 0.5
        return s

    def retrieve(self, query: str, top_k: int = 6, seed: Optional[int] = None) -> List[KBEntry]:
        q = self._parse_query(query)
        pool = [e for e in self.entries if self._match_signature(e, q)] or list(self.entries)
        pool.sort(key=lambda e: self._score(e, query), reverse=True)
        best_sig = pool[0].hand_signature
        confusers = [e for e in self.entries if e.hand_signature == best_sig]

        seen = set()
        merged: List[KBEntry] = []
        for e in confusers + pool[:top_k]:
            if e.name not in seen:
                merged.append(e)
                seen.add(e.name)

        rnd = random.Random(seed)
        rnd.shuffle(merged)
        return merged
