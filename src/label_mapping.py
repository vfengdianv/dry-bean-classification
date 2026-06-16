"""
Label cleaning mapping table — centralized constant management.

Covers all label variants discovered in the Dry Bean Dataset:
- Case variants (dermason → DERMASON)
- Character-substitution attacks (D3RMAS0N → DERMASON, H0R0Z → HOROZ, etc.)
- Trailing whitespace variants (handled via str.strip() before lookup)
"""

# Complete label mapping covering all known dirty variants across train/test/val
LABEL_MAP = {
    # === Standard forms (identity mapping) ===
    'DERMASON': 'DERMASON',
    'SIRA': 'SIRA',
    'SEKER': 'SEKER',
    'HOROZ': 'HOROZ',
    'CALI': 'CALI',
    'BARBUNYA': 'BARBUNYA',
    'BOMBAY': 'BOMBAY',
    # === Lowercase variants ===
    'dermason': 'DERMASON',
    'sira': 'SIRA',
    'seker': 'SEKER',
    'horoz': 'HOROZ',
    'cali': 'CALI',
    'barbunya': 'BARBUNYA',
    'bombay': 'BOMBAY',
    # === Character-substitution attacks (3→E, 0→O) ===
    'D3RMAS0N': 'DERMASON',
    'S3K3R': 'SEKER',
    'H0R0Z': 'HOROZ',
    'B0MBAY': 'BOMBAY',
    # === Trailing-whitespace variants (listed for documentation; strip() handles these) ===
    'DERMASON ': 'DERMASON',
    'SIRA ': 'SIRA',
    'SEKER ': 'SEKER',
    'HOROZ ': 'HOROZ',
    'CALI ': 'CALI',
    'BARBUNYA ': 'BARBUNYA',
    'BOMBAY ': 'BOMBAY',
    # === Header-row leakage sentinel ===
    'Class': None,  # Discard — header row mistakenly included as data
}

# Valid class labels after cleaning
CLEAN_CLASSES = ['DERMASON', 'SIRA', 'SEKER', 'HOROZ', 'CALI', 'BARBUNYA', 'BOMBAY']


def clean_label(raw_label: str) -> str:
    """
    Clean a single raw label: strip whitespace, then lookup in LABEL_MAP.

    Raises ValueError for unrecognized labels — never silently discards.
    Returns the cleaned canonical label string.
    """
    stripped = raw_label.strip()
    if stripped not in LABEL_MAP:
        raise ValueError(
            f"Unrecognized label: raw='{raw_label}' → stripped='{stripped}'. "
            f"Add it to LABEL_MAP in label_mapping.py."
        )
    result = LABEL_MAP[stripped]
    if result is None:
        raise ValueError(
            f"Label '{stripped}' maps to None (header-row sentinel). "
            f"Check if the CSV header leaked into the data."
        )
    return result
