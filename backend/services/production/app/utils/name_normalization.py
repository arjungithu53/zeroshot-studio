#!/usr/bin/env python3
"""
Name Normalization Utilities
=============================
Consistent asset naming across Phase 1 and Phase 2 for character/location mapping.
"""


def normalize_asset_name(name: str) -> str:
    """
    Normalize asset name to UPPERCASE_WITH_UNDERSCORES format.

    This ensures consistent naming across CSV parsing, Phase 1 asset generation,
    and Phase 2 asset library lookups.

    Args:
        name: Original asset name (e.g., "Alex Smith", "Dense Forest", "black lab puppy")

    Returns:
        Normalized name (e.g., "ALEX_SMITH", "DENSE_FOREST", "BLACK_LAB_PUPPY")

    Examples:
        >>> normalize_asset_name("Alex Smith")
        'ALEX_SMITH'
        >>> normalize_asset_name("Dense Forest")
        'DENSE_FOREST'
        >>> normalize_asset_name("  jungle  ")
        'JUNGLE'
    """
    if not name:
        return ""
    return name.strip().upper().replace(' ', '_')


def denormalize_asset_name(normalized_name: str) -> str:
    """
    Convert normalized name back to human-readable format.

    Args:
        normalized_name: Normalized name (e.g., "ALEX_SMITH", "DENSE_FOREST")

    Returns:
        Human-readable name (e.g., "Alex Smith", "Dense Forest")

    Examples:
        >>> denormalize_asset_name("ALEX_SMITH")
        'Alex Smith'
        >>> denormalize_asset_name("DENSE_FOREST")
        'Dense Forest'
    """
    if not normalized_name:
        return ""
    return normalized_name.replace('_', ' ').title()


def normalize_list(names: list) -> list:
    """
    Normalize a list of names.

    Args:
        names: List of original names

    Returns:
        List of normalized names

    Examples:
        >>> normalize_list(["Alex", "Sarah Thompson", "DOG"])
        ['ALEX', 'SARAH_THOMPSON', 'DOG']
    """
    return [normalize_asset_name(name) for name in names if name and name.strip()]


def parse_location_with_variation(location_str: str) -> tuple:
    """
    Parse location string to extract base name and directional variation.

    Supports directional suffixes: _NORTH, _SOUTH, _EAST, _WEST

    Args:
        location_str: Location string from CSV (e.g., "Jungle East", "JUNGLE_EAST", "Jungle")

    Returns:
        Tuple of (base_location, variation_angle) where variation_angle is None if no suffix

    Examples:
        >>> parse_location_with_variation("JUNGLE_EAST")
        ('JUNGLE', 'east')
        >>> parse_location_with_variation("Jungle East")
        ('JUNGLE', 'east')
        >>> parse_location_with_variation("Dense Forest North")
        ('DENSE_FOREST', 'north')
        >>> parse_location_with_variation("Jungle")
        ('JUNGLE', None)
        >>> parse_location_with_variation("CAVE_ENTRANCE_WEST")
        ('CAVE_ENTRANCE', 'west')
    """
    if not location_str or not location_str.strip():
        return "", None

    # First normalize the location string to UPPERCASE_WITH_UNDERSCORES
    normalized = normalize_asset_name(location_str)

    # Check for directional suffixes (in order: _NORTH, _SOUTH, _EAST, _WEST)
    directional_suffixes = {
        '_NORTH': 'north',
        '_SOUTH': 'south',
        '_EAST': 'east',
        '_WEST': 'west'
    }

    for suffix, angle in directional_suffixes.items():
        if normalized.endswith(suffix):
            # Extract base location by removing the suffix
            base_location = normalized[:-len(suffix)]
            return base_location, angle

    # No directional suffix found - return normalized name with no variation
    return normalized, None
