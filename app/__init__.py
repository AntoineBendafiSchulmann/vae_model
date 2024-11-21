from flask import Flask
from flask_cors import CORS
import os

def create_app():

    app = Flask(__name__)
    CORS(app)

    app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "app/models/static/input_images")
    app.config["OUTPUT_FOLDER"] = os.path.join(os.getcwd(), "app/models/static/output_images")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limite de 16MB

    from app.models.utils.api import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix="/api")

    return app
