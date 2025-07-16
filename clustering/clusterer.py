import logging

from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, mapping
import geojson

import config
from .utils import filter_valid_reports

# Configure module‐level logger
logger = logging.getLogger(__name__)


def run_dbscan(raw_reports):
    """
    Perform DBSCAN clustering on a list of reports.

    Each report should be a dict with numeric 'latitude' and 'longitude' keys.
    Returns a geojson.FeatureCollection:
      - Polygons for each DBSCAN cluster (ignoring noise).
      - If no clusters are found (or on error), falls back to returning
        each point as an individual Feature.

    Args:
        raw_reports (List[dict]): Incoming reports payload.

    Returns:
        geojson.FeatureCollection: Cluster polygons or point features.
    """
    try:
        # 1) Filter out any entries lacking valid coords
        valid = filter_valid_reports(raw_reports)

        # 2) Extract (longitude, latitude) pairs for scikit‐learn
        points = [(r['longitude'], r['latitude']) for r in valid]

        # If nothing to cluster, return empty collection
        if not points:
            return geojson.FeatureCollection([])

        # 3) Run DBSCAN with parameters from config.py
        clustering = DBSCAN(
            eps=config.DBSCAN_EPS,
            min_samples=config.DBSCAN_MIN_SAMPLES
        ).fit(points)
        labels = clustering.labels_

        features = []
        # Build one polygon per cluster label (excluding noise = -1)
        for label in set(labels):
            if label == -1:
                continue
            cluster_pts = [points[i] for i, lab in enumerate(labels) if lab == label]
            # Compute convex hull of cluster points
            hull = MultiPoint(cluster_pts).convex_hull
            poly_geo = mapping(hull)
            features.append(
                geojson.Feature(
                    geometry=poly_geo,
                    properties={'cluster_id': int(label), 'count': len(cluster_pts)}
                )
            )

        # 4) Fallback: if DBSCAN found no clusters, return each point
        if not features:
            for idx, (lon, lat) in enumerate(points):
                features.append(
                    geojson.Feature(
                        geometry=geojson.Point((lon, lat)),
                        properties={'cluster_id': idx, 'count': 1}
                    )
                )

        return geojson.FeatureCollection(features)

    except Exception as e:
        # Log unexpected errors and fallback to point features
        logger.error("DBSCAN clustering error: %s", e, exc_info=True)
        fallback = []
        try:
            for idx, r in enumerate(filter_valid_reports(raw_reports)):
                lon, lat = r['longitude'], r['latitude']
                fallback.append(
                    geojson.Feature(
                        geometry=geojson.Point((lon, lat)),
                        properties={'cluster_id': idx, 'count': 1}
                    )
                )
        except Exception as inner:
            logger.error("Fallback feature creation failed: %s", inner, exc_info=True)
            return geojson.FeatureCollection([])

        return geojson.FeatureCollection(fallback)
