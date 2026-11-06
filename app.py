import os
from flask import Flask, render_template, redirect, url_for, Blueprint
from config import SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH
from database import init_db
from routes.auth_routes import auth_bp
from routes.predict_routes import predict_bp
from services.model_loader import load_models

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')


def create_app():
    app = Flask(__name__)

    app.secret_key                = SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    init_db()

    crop_model, health_model, device = load_models()
    app.config['CROP_MODEL']   = crop_model
    app.config['HEALTH_MODEL'] = health_model
    app.config['DEVICE']       = device

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)