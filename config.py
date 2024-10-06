import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, 'sessions')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Paths for StyleTTS2
STYLETTS2_DIR = os.path.join(BASE_DIR, 'StyleTTS2')
STYLETTS2_MODELS_DIR = os.path.join(STYLETTS2_DIR, 'Models', 'LibriTTS')
REFERENCE_VOICES_DIR = os.path.join(STYLETTS2_DIR, 'reference_voices')
STYLETTS2_CONFIG_PATH = os.path.join(STYLETTS2_MODELS_DIR, 'config.yml')
STYLETTS2_MODEL_WEIGHTS = os.path.join(STYLETTS2_MODELS_DIR, 'epoch_2nd_00100.pth')

# Paths for other models used by StyleTTS2 (adjust as necessary)
STYLETTS2_ASR_CONFIG = os.path.join(STYLETTS2_DIR, 'path', 'to', 'ASR_config')
STYLETTS2_ASR_PATH = os.path.join(STYLETTS2_DIR, 'path', 'to', 'ASR_model')
STYLETTS2_F0_PATH = os.path.join(STYLETTS2_DIR, 'path', 'to', 'F0_model')
STYLETTS2_PLBERT_DIR = os.path.join(STYLETTS2_DIR, 'Utils', 'PLBERT')
