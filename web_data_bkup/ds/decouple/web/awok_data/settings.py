"""
Django settings for awok_data project.

For more information on this file, see
https://docs.djangoproject.com/en/1.6/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.6/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.6/howto/deployment/checklist/
NODE_OPTIONS = [

    {'name':'dropout', 'type':'boolean'},
    {'name':'reform', 'type':'boolean'},
    {'name':'fan_in','type':'number'},
    {'name':'fan_out','type':'number'},
    {'name':'weight_update_status','type':'boolean'},
    {'name':'weight_decay','type':'number'},
    {'name':'momentum','type':'number'},
    {'name':'alpha','type':'number'},
    {'name':'init_scaler','type':'text'},
    {'name':'a_func', 'type':'text'},
    {'name':'skip_grad', 'type':'boolean'},
    {'name':'n_channels', 'type':'number'},
    {'name':'n_filters', 'type':'number'},
    {'name':'filter_shape', 'type':'text'},
    {'name':'image_shape', 'type':'text'},
    {'name':'no_padding', 'type':'boolean'}

]

NODE_COLORS = { 
    "LossNode" : "#4A9097", 
    "FunctionNode": "#A99494",
    "FilterNode": "#8E5AB5",
    "DataNode": "#B93B3B",
    "PoolNode": "#8AA160",
    "ConvolutionNode": "#CBA966",
    "WeightNode": "#DF8675",
    "OnesNode": "#628BA9",
    "TargetNode":"#fff",
    "DotProductNode": "#5C8CEC"
    }

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'nu!a5k)gz9xhtf2_6axc00@ws6^ny^*rrt!-u7r=0f2^#pt*7+'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

TEMPLATE_DEBUG = True

ALLOWED_HOSTS = []

CURR_SESSION = 'empty'
# Application definition

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'data_tree',
    'ws4redis',
)

LOGIN_URL = '/login/'

WEBSOCKET_ACCEPT_ALL = True

WEBSOCKET_URL = '/ws/'

WS4REDIS_CONNECTION = {
    'host': '127.0.0.1',
    'port': 6379,
}

WS4REDIS_PREFIX = 'ws'

WS4REDIS_SUBSCRIBER = 'data_tree.redis_store.RedisSubscriber'

WSGI_APPLICATION = 'ws4redis.django_runserver.application'

TEMPLATE_CONTEXT_PROCESSORS = (
    'django.contrib.auth.context_processors.auth',
    'django.core.context_processors.static',
    'ws4redis.context_processors.default',
)

MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'awok_data.urls'

WSGI_APPLICATION = 'awok_data.wsgi.application'

# Database
# https://docs.djangoproject.com/en/1.6/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Internationalization
# https://docs.djangoproject.com/en/1.6/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

TEMPLATE_DIRS = (
# Put strings here, like "/home/html/django_templates" or
# "C:/www/django/templates".
# Always use forward slashes, even on Windows.
# Don't forget to use absolute paths, not relative paths.
os.path.join(BASE_DIR, '/data_tree/templates/'),
)

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.6/howto/static-files/

STATIC_URL = '/static/'

#DEBUG = True