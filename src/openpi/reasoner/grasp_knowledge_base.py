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
    object_shape: str
    contact_region: str
    avoid_if: str

    @property
    def hand_signature(self) -> str:
        return f"{self.coarse}|{self.opposition}|{self.thumb}|VF2={self.vf2}"


class GraspKnowledgeBase:
    """Confusion-aware in-memory grasp KB for contrast-set RAG."""

    def __init__(self) -> None:
        self.entries: List[KBEntry] = [
            # Large Diameter
            self._entry("Power", "Large Diameter", "Palm", "Abd", "2-5", "Cylinder, Tapered Cylinder, Cuboid/Prismatic, Handle, Ring, Torus, Ring-Plate, Rim/Edge", "sidewall/rim", ""),
            # Ring 
            self._entry("Power", "Ring", "Palm", "Abd", "2", "Cylinder, Tapered Cylinder, Cuboid/Prismatic, Handle, Ring, Torus, Ring-Plate, Rim/Edge", "sidewall/rim", ""),
            # Prismatic 4 Finger
            # Added Cylinder/Round objects as this category absorbs Circular 4 Finger in simplified taxonomies
            self._entry("Precision", "Prismatic 4 Finger", "Pad", "Abd", "2-5", "Cuboid, Prismatic, Plate, Edge, Cylinder, Round Object", "edge/sidewall/face", ""),

            # Inferior Pincer
            self._entry("Precision", "Inferior Pincer", "Pad", "Abd", "2", "Small Feature/Edge, Tab/Pull-Tab, Rim/Edge, Cavity/Hole, Cylinder", "edge/sidewall", ""),

            # Precision Sphere
            self._entry("Precision", "Precision Sphere", "Pad", "Abd", "2-5", "Sphere/Ellipsoid, Knob, Small Ball-like Object", "surface", ""),

            # Lateral
            self._entry("Intermediate", "Lateral", "Side", "Add", "2", "Plate/Sheet, Edge, Tab/Pull-Tab, Prismatic", "edge/face", ""),

            # Adducted Thumb
            self._entry("Power", "Adducted Thumb", "Palm", "Add", "2-5", "Cylinder/Cuboid, Handle, Prismatic, Knob", "sidewall", ""),
        ]

    @staticmethod
    def _entry(
        coarse: str,
        fine: str,
        opposition: str,
        thumb: str,
        vf2: str,
        object_shape: str,
        contact_region: str,
        avoid_if: str,
    ) -> KBEntry:
        fine_slug = fine.replace(" ", "_")
        return KBEntry(
            name=f"{coarse}:{fine_slug}",
            coarse=coarse,
            fine=fine,
            opposition=opposition,
            thumb=thumb,
            vf2=vf2,
            object_shape=object_shape,
            contact_region=contact_region,
            avoid_if=avoid_if,
        )

    @staticmethod
    def _parse_query(query: str) -> Dict[str, str]:
        q = query.lower()

        def _grab(pat: str) -> str:
            m = re.search(pat, q, re.IGNORECASE)
            return m.group(1).strip() if m else ""

        # Improved VF2 extraction to handle "Virtual Fingers: VF2: 1 vs 2-5" or "Virtual Fingers: 2-5"
        vf2_raw = _grab(r"(?:virtual fingers|vf2)\s*[:\-]\s*(?:vf2\s*[:\-]\s*)?(?:1\s*vs\s*)?([0-9\-]+)")

        return {
            "opposition": _grab(r"opposition\s*:\s*(palm|pad|side)"),
            "thumb": _grab(r"thumb\s*position\s*:\s*(abd|add)"),
            "vf2": vf2_raw,
            "object_shape": _grab(r"object shape\s*:\s*([a-z0-9\-\s/]+)"),
            "contact_region": _grab(r"contact region\s*:\s*([a-z0-9\-\s/]+)"),
        }

    @staticmethod
    def _match_signature(e: KBEntry, q: Dict[str, str]) -> bool:
        ok = True
        if q.get("opposition") and q["opposition"] != "unknown":
            ok &= e.opposition.lower() == q["opposition"]
        if q.get("thumb") and q["thumb"] != "unknown":
            ok &= e.thumb.lower() == q["thumb"]
        if q.get("vf2") and q["vf2"] not in ["", "unknown"]:
            # Normalize: "2-5" should match "2-5", "2" should match "2"
            q_vf2 = q["vf2"].replace(" ", "")
            e_vf2 = e.vf2.replace(" ", "")
            # Allow partial match if needed? For now strict after normalization
            ok &= (e_vf2 == q_vf2)
        
        # Mandatory check for Object Shape compatibility
        if ok and q.get("object_shape") and q["object_shape"] not in ["", "unknown", "error"]:
            # KB format: "Cylinder, Tapered Cylinder"
            kb_shapes = [s.strip() for s in re.split(r'[,/]', e.object_shape.lower())]
            q_shapes = [s.strip() for s in re.split(r'[,/]', q["object_shape"].lower())]
            
            match_shape = False
            for qs in q_shapes:
                if any(qs in ks or ks in qs for ks in kb_shapes):
                    match_shape = True
                    break
            ok &= match_shape

        # Mandatory check for Contact Region compatibility
        if ok and q.get("contact_region") and q["contact_region"] not in ["", "unknown", "error"]:
            kb_regions = [r.strip() for r in re.split(r'[,/]', e.contact_region.lower())]
            q_regions = [r.strip() for r in re.split(r'[,/]', q["contact_region"].lower())]
            
            match_region = False
            for qr in q_regions:
                if any(qr in kr or kr in qr for kr in kb_regions):
                    match_region = True
                    break
            ok &= match_region

        return ok

    @staticmethod
    def _score(e: KBEntry, query: str) -> float:
        q = query.lower()
        s = 0.0
        for tok in [e.coarse, e.fine, e.opposition, e.thumb, e.object_shape, e.contact_region]:
            if tok and tok.lower() in q:
                s += 1.0
        return s

    def retrieve(self, query: str, top_k: int = 6, seed: Optional[int] = None) -> List[KBEntry]:
        q = self._parse_query(query)
        
        # Level 1: Strict Match (Hand Signature + Shape + Region)
        pool = [e for e in self.entries if self._match_signature(e, q)]
        
        # Level 2: Hand Signature Match (Opposition + Thumb + VF2) - ignore geometry mismatch if strict failed
        if not pool:
            q_sig = {k: v for k, v in q.items() if k in ["opposition", "thumb", "vf2"]}
            pool = [e for e in self.entries if self._match_signature(e, q_sig)]

        # Level 3: Opposition Match Only (Critical anatomical constraint)
        if not pool and q.get("opposition") and q["opposition"] not in ["", "unknown"]:
            q_opp = {"opposition": q["opposition"]}
            pool = [e for e in self.entries if self._match_signature(e, q_opp)]

        # Level 4: Fallback to all (if even opposition is unknown or no match)
        pool = pool or list(self.entries)

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
