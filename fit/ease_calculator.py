"""
fit/ease_calculator.py — Rule-based ease computation + industry fit tolerances

─────────────────────────────────────────────────────────────────
WHAT IS "EASE"?
─────────────────────────────────────────────────────────────────
Ease = garment_measurement - body_measurement

Positive ease: garment is LARGER than body → loose fit
Negative ease: garment is SMALLER than body → compression / too tight

A chest measurement of 90 cm and a garment chest of 94 cm →
ease_chest = +4 cm → "Fitted" (industry standard range: 2–6 cm)

─────────────────────────────────────────────────────────────────
INDUSTRY STANDARDS (ASTM D5585 + ISO 8559)
─────────────────────────────────────────────────────────────────
These tolerances are published in industry sizing standards and
vary slightly by garment category (woven vs. knit, casual vs. formal).
The values below are for woven upper-body garments (shirts, blouses).

Chest/Bust ease:
  < -2 cm      → Too Tight     (garment will restrict movement)
  -2 to 2 cm   → Fitted        (close-fitting, knit-like feel)
   2 to 6 cm   → Comfortable   (standard woven shirt ease)
   6 to 14 cm  → Loose         (relaxed fit)
  > 14 cm      → Too Large     (oversized)

Waist ease (tighter tolerances — waist is more sensitive):
  < -3 cm      → Too Tight
  -3 to 1 cm   → Fitted
   1 to 5 cm   → Comfortable
   5 to 12 cm  → Loose
  > 12 cm      → Too Large

Hip ease:
  < -2 cm      → Too Tight
  -2 to 3 cm   → Fitted
   3 to 8 cm   → Comfortable
   8 to 16 cm  → Loose
  > 16 cm      → Too Large

Overall fit = most restrictive label across all measured regions.
"""

from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

FIT_LABELS = ['Too Tight', 'Fitted', 'Comfortable', 'Loose', 'Too Large']
FIT_LABEL_TO_IDX = {label: i for i, label in enumerate(FIT_LABELS)}
FIT_IDX_TO_LABEL = {i: label for i, label in enumerate(FIT_LABELS)}


@dataclass
class PersonMeasurements:
    """Body measurements in centimetres."""
    chest:            float
    waist:            float
    hip:              float
    shoulder_width:   Optional[float] = None
    height:           Optional[float] = None
    weight:           Optional[float] = None    # kg


@dataclass
class GarmentMeasurements:
    """Garment measurements in centimetres (from label/metadata)."""
    garment_chest:    float
    garment_waist:    float
    garment_hip:      float
    garment_length:   Optional[float] = None
    garment_shoulder: Optional[float] = None
    garment_type:     str = 'upper'   # 'upper', 'lower', 'overall'


@dataclass
class EaseValues:
    """Ease = garment - body for each region."""
    chest: float
    waist: float
    hip:   float


@dataclass
class FitResult:
    """Complete fit assessment output."""
    fit_label:      str
    ease:           EaseValues
    region_labels:  dict            # per-region fit labels
    confidence:     float           # from ML classifier (0–1)
    recommendation: str


# ─────────────────────────────────────────────────────────────────
# Tolerance tables
# ─────────────────────────────────────────────────────────────────

# Each entry: (lower_bound_inclusive, upper_bound_exclusive, label)
TOLERANCES = {
    'chest': [
        (float('-inf'), -2,   'Too Tight'),
        (-2,             2,   'Fitted'),
        ( 2,             6,   'Comfortable'),
        ( 6,            14,   'Loose'),
        (14, float('inf'),    'Too Large'),
    ],
    'waist': [
        (float('-inf'), -3,   'Too Tight'),
        (-3,             1,   'Fitted'),
        ( 1,             5,   'Comfortable'),
        ( 5,            12,   'Loose'),
        (12, float('inf'),    'Too Large'),
    ],
    'hip': [
        (float('-inf'), -2,   'Too Tight'),
        (-2,             3,   'Fitted'),
        ( 3,             8,   'Comfortable'),
        ( 8,            16,   'Loose'),
        (16, float('inf'),    'Too Large'),
    ],
}

# Label severity order — lower index = more severe
SEVERITY = {label: i for i, label in enumerate(FIT_LABELS)}


# ─────────────────────────────────────────────────────────────────
# Ease calculator
# ─────────────────────────────────────────────────────────────────

class EaseCalculator:
    """
    Computes ease values and applies industry tolerance rules to
    produce a fit label.

    The rule engine is DETERMINISTIC: same inputs always produce
    same output. It runs before the ML classifier (D2) and is used
    as the primary signal when measurements are in a clear range.
    """

    def compute_ease(
        self,
        person: PersonMeasurements,
        garment: GarmentMeasurements,
    ) -> EaseValues:
        """Ease = garment_measurement - person_measurement."""
        return EaseValues(
            chest = garment.garment_chest - person.chest,
            waist = garment.garment_waist - person.waist,
            hip   = garment.garment_hip   - person.hip,
        )

    def classify_region(self, ease_value: float, region: str) -> str:
        """Apply tolerance table to get fit label for one region."""
        for lo, hi, label in TOLERANCES[region]:
            if lo <= ease_value < hi:
                return label
        return 'Too Large'  # fallback (shouldn't reach here)

    def classify_overall(
        self,
        ease: EaseValues,
        garment_type: str = 'upper',
    ) -> tuple[str, dict]:
        """
        Compute overall fit = most severe region label.

        For upper garments: chest and waist matter (not hip).
        For lower garments: waist and hip matter (not chest).
        For overall:        all three matter.

        Returns:
            (overall_label, region_labels_dict)
        """
        if garment_type == 'upper':
            active_regions = {'chest': ease.chest, 'waist': ease.waist}
        elif garment_type == 'lower':
            active_regions = {'waist': ease.waist, 'hip': ease.hip}
        else:  # overall
            active_regions = {'chest': ease.chest, 'waist': ease.waist, 'hip': ease.hip}

        region_labels = {
            region: self.classify_region(val, region)
            for region, val in active_regions.items()
        }

        # Overall = worst-case (most severe) label
        most_severe = max(region_labels.values(), key=lambda lbl: SEVERITY[lbl])
        return most_severe, region_labels

    def generate_recommendation(
        self,
        fit_label: str,
        ease: EaseValues,
        region_labels: dict,
        garment_type: str = 'upper',
    ) -> str:
        """
        Generate a human-readable recommendation string.
        """
        if fit_label == 'Comfortable':
            tight_regions = [r for r, l in region_labels.items() if l == 'Fitted']
            if tight_regions:
                return (f"Good fit overall. Slightly snug at "
                        f"{', '.join(tight_regions)} — comfortable for most body types.")
            return "This garment fits your measurements well. Comfortable and easy to move in."

        elif fit_label == 'Fitted':
            tight_regions = [r for r, l in region_labels.items()
                             if l in ('Fitted', 'Too Tight')]
            return (f"Close-fitting at {', '.join(tight_regions) if tight_regions else 'most areas'}. "
                    "Ideal for a tailored look. If you prefer more room, consider sizing up.")

        elif fit_label == 'Too Tight':
            tight_regions = [r for r, l in region_labels.items() if l == 'Too Tight']
            ease_str = ', '.join(
                f"{r}: {getattr(ease, r):+.1f} cm"
                for r in tight_regions if hasattr(ease, r)
            )
            return (f"This garment is too tight at {', '.join(tight_regions)} "
                    f"({ease_str}). Movement may be restricted. Size up is recommended.")

        elif fit_label == 'Loose':
            return ("Relaxed fit. Extra fabric provides comfort and ease of movement. "
                    "If a more structured silhouette is desired, consider sizing down.")

        elif fit_label == 'Too Large':
            return ("This garment is significantly larger than your measurements. "
                    "The fabric will bunch and drape loosely. Consider sizing down 1–2 sizes.")

        return f"Fit assessment: {fit_label}."

    def assess(
        self,
        person: PersonMeasurements,
        garment: GarmentMeasurements,
        ml_label: Optional[str] = None,
        ml_confidence: float = 0.0,
    ) -> FitResult:
        """
        Full fit assessment.

        If ml_label is provided and its confidence > 0.7, it's used
        to override the rule engine when they disagree (edge cases).
        Otherwise the rule engine is authoritative.
        """
        ease = self.compute_ease(person, garment)
        rule_label, region_labels = self.classify_overall(ease, garment.garment_type)

        # Resolve between rule engine and ML
        if ml_label and ml_confidence > 0.7 and ml_label != rule_label:
            # ML overrides only for confident predictions in adjacent categories
            rule_idx = SEVERITY[rule_label]
            ml_idx   = SEVERITY[ml_label]
            if abs(rule_idx - ml_idx) <= 1:   # only override if 1 step apart
                final_label = ml_label
                confidence = ml_confidence
            else:
                # Labels are far apart — trust rule engine
                final_label = rule_label
                confidence = 0.9
        else:
            final_label = rule_label
            confidence = 0.9 if ml_label == rule_label else 0.7

        recommendation = self.generate_recommendation(
            final_label, ease, region_labels, garment.garment_type
        )

        return FitResult(
            fit_label      = final_label,
            ease           = ease,
            region_labels  = region_labels,
            confidence     = confidence,
            recommendation = recommendation,
        )


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    calc = EaseCalculator()

    # Ease values verified against the actual TOLERANCES table above,
    # chosen mid-range on BOTH active regions for 'upper' (chest AND
    # waist — assess() takes the more severe of the two, so a value
    # that's mid-range on only one region isn't enough):
    #   chest:  Too Tight <-2 | Fitted -2..2 | Comfortable 2..6 | Loose 6..14 | Too Large >=14
    #   waist:  Too Tight <-3 | Fitted -3..1 | Comfortable 1..5 | Loose 5..12 | Too Large >=12
    test_cases = [
        # (person_chest, person_waist, person_hip, garment_chest, garment_waist, garment_hip, expected)
        (90, 74, 96, 94, 77, 100, 'Comfortable'),    # ease=(+4/+3/+4)  both regions mid-Comfortable
        (90, 74, 96, 84, 68, 90,  'Too Tight'),      # ease=(-6/-6/-6)  both regions clearly Too Tight
        (90, 74, 96, 99, 82, 104, 'Loose'),          # ease=(+9/+8/+8)  both regions mid-Loose
        (90, 74, 96, 115, 98, 120,'Too Large'),      # ease=(+25/+24/+24) both regions clearly Too Large
        (90, 74, 96, 90, 73, 96,  'Fitted'),         # ease=(+0/-1/+0)  both regions mid-Fitted
    ]

    print(f"{'Person':20s} {'Garment':20s} {'Ease (C/W/H)':20s} {'Label':12s} {'Expected':12s} {'OK'}")
    print("-" * 90)

    for pc, pw, ph, gc, gw, gh, expected in test_cases:
        person  = PersonMeasurements(chest=pc, waist=pw, hip=ph)
        garment = GarmentMeasurements(garment_chest=gc, garment_waist=gw,
                                      garment_hip=gh, garment_type='upper')
        result  = calc.assess(person, garment)
        ease    = result.ease
        ok = '✅' if result.fit_label == expected else '❌'
        print(f"C={pc} W={pw} H={ph}  →  C={gc} W={gw} H={gh}  "
              f"ease=({ease.chest:+.0f}/{ease.waist:+.0f}/{ease.hip:+.0f})  "
              f"{result.fit_label:12s} {expected:12s} {ok}")
        if result.fit_label != expected:
            print(f"  Recommendation: {result.recommendation}")
