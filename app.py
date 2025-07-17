import logging
import config
from flask import Flask, request, jsonify
from flask_cors import CORS
from clustering.clusterer import run_dbscan

# Configure Flask app and logging
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any domain
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/predict_hotspots', methods=['POST'])
def predict_hotspots():
    """
    HTTP POST /predict_hotspots
    ---------------------------
    Expects a JSON payload of the form:
      {
        "reports": [
          { "latitude": 43.65, "longitude": -79.38 },
          ...
        ]
      }

    Returns:
      - 200 + GeoJSON FeatureCollection on success
      - 400 if the request is malformed
      - 500 on internal error
    """
    try:
        # Force parsing as JSON, even if incorrect Content-Type
        payload = request.get_json(force=True)
    except Exception as e:
        logger.warning("Failed to parse JSON: %s", e)
        return jsonify({"msg": "Invalid JSON payload"}), 400

    # Validate that 'reports' is a list
    reports = payload.get('reports')
    if not isinstance(reports, list):
        logger.warning("Bad request: 'reports' missing or not a list")
        return jsonify({"msg": "'reports' field must be a list of objects"}), 400

    # Delegate to clustering logic
    try:
        feature_collection = run_dbscan(reports)
        # The result is already a GeoJSON-compatible dict
        return jsonify(feature_collection), 200

    except Exception as e:
        # Catch any unexpected errors from run_dbscan
        logger.error("Error in predict_hotspots: %s", e, exc_info=True)
        return jsonify({"msg": "Server error generating hotspots"}), 500


if __name__ == '__main__':
    # Run on configured port, defaulting to 5001
    app.run(host='0.0.0.0', port=getattr(config, 'FLASK_PORT', 5001), debug=False)
