import os
import secrets
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = secrets.token_hex(32)

DATABASE_PATH = os.path.join(BASE_DIR, 'tea_coffee.db')

CROP_MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'crop_model_best.pth')
HEALTH_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'health_model_best.pth')

UPLOAD_FOLDER      = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

CROP_CLASSES   = ['coffee', 'tea']
HEALTH_CLASSES = ['healthy', 'unhealthy']

CROP_CONFIDENCE_THRESHOLD = 60.0

SEVERITY_MILD_THRESHOLD     = 15.0
SEVERITY_MODERATE_THRESHOLD = 40.0

KB_WEIGHTS   = {'crop': 0.5, 'health': 0.3, 'color': 0.2}
KB_MIN_SCORE = 0.4

COLOR_RANGES = {
    'brown': [([0,  40, 30],  [18, 255, 170]),
              ([18, 40, 60],  [25, 180, 160])],
    'black': [([0,  0,  0],   [180, 120, 80])],
    'yellow':[([20, 80, 150], [35, 255, 255])],
    'white': [([0,  0,  160], [180, 50,  255])],
    'gray':  [([0,  0,  55],  [180, 55,  195])],
}

COLOR_THRESHOLDS = {
    'brown':  0.02,
    'black':  0.01,
    'yellow': 0.10,
    'white':  0.06,
    'gray':   0.04,
}

COLOR_CAP_RATIOS = {
    'gray':  0.25,
    'black': 0.20,
}

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL   = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')