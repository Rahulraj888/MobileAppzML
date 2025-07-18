"""
Utility functions for clustering module.
"""

def filter_valid_reports(raw_reports):
    """
    Filters out any entries that are not dicts or lack numeric latitude/longitude.

    Args:
        raw_reports (Iterable): Input sequence of reports.

    Returns:
        List[dict]: Only those reports with both 'latitude' and 'longitude' keys.
    """
    valid = []
    for r in raw_reports:
        if not isinstance(r, dict):
            continue
        lat = r.get('latitude')
        lon = r.get('longitude')
        # Ensure lat/lon exist and are convertible to float
        try:
            if lat is None or lon is None:
                continue
            float(lat)
            float(lon)
            valid.append(r)
        except (TypeError, ValueError):
            continue
    return valid
