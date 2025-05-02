import os

# Superset specific config
ROW_LIMIT = 5000
SUPERSET_WEBSERVER_PORT = 8088

# Flask App Builder configuration
# Your App secret key will be used for securely signing the session cookie
# and encrypting sensitive information on the database
# Make sure you are changing this key for your deployment with a strong key.
# You can generate a strong key using `openssl rand -base64 42`
SECRET_KEY = os.getenv('SUPERSET_SECRET_KEY', 'your_secret_key_here')

# The SQLAlchemy connection string to your database backend
# This connection defines the path to the database that stores your
# superset metadata (slices, connections, tables, dashboards, ...).
SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'postgresql://postgres:postgres@db:5432/supply_chain_analytics')

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_CELERY_DB = os.getenv('REDIS_CELERY_DB', 0)
REDIS_RESULTS_DB = os.getenv('REDIS_RESULTS_DB', 1)

# Cache configuration
CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_KEY_PREFIX': 'superset_',
    'CACHE_REDIS_HOST': REDIS_HOST,
    'CACHE_REDIS_PORT': REDIS_PORT,
    'CACHE_REDIS_DB': REDIS_RESULTS_DB,
}

# Custom configuration for our supply chain analytics
FEATURE_FLAGS = {
    'DASHBOARD_NATIVE_FILTERS': True,
    'DASHBOARD_CROSS_FILTERS': True,
    'DASHBOARD_NATIVE_FILTERS_SET': True,
    'ENABLE_TEMPLATE_PROCESSING': True,
    'ENABLE_TEMPLATE_REMOVE_FILTERS': True,
    'DASHBOARD_CACHE': True,
}

# Additional database configurations
SQLLAB_TIMEOUT = 300
SQLLAB_ASYNC_TIME_LIMIT_SEC = 300
RESULTS_BACKEND = CACHE_CONFIG

# Visualization configurations
VIZ_TYPE_BLACKLIST = []
ALERT_REPORTS_NOTIFICATION_DRY_RUN = True

# Default configurations for charts
CHART_CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_KEY_PREFIX': 'superset_chart_',
    'CACHE_REDIS_HOST': REDIS_HOST,
    'CACHE_REDIS_PORT': REDIS_PORT,
    'CACHE_REDIS_DB': REDIS_RESULTS_DB,
}

# Security configuration
AUTH_TYPE = 'db'
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "Admin"

# Email configuration (optional - for alerts)
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_STARTTLS = True
SMTP_SSL = False
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PORT = 587
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SMTP_MAIL_FROM = os.getenv('SMTP_MAIL_FROM', '')

# Dashboard refresh frequency
DASHBOARD_AUTO_REFRESH_INTERVALS = [
    5, 10, 30, 60, 300  # in seconds
]

# Custom CSS for branding (optional)
BRANDED_USER_ATTRIBUTE = {'custom_css': '/static/assets/custom/custom.css'} 