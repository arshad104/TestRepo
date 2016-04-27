# import os
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_tree.settings')
# from django.conf import settings
# from django.core.wsgi import get_wsgi_application
# from ws4redis.uwsgi_runserver import uWSGIWebsocketServer

# _django_app = get_wsgi_application()
# _websocket_app = uWSGIWebsocketServer()

# def application(environ, start_response):
#     if environ.get('PATH_INFO').startswith(settings.WEBSOCKET_URL):
#         return _websocket_app(environ, start_response)
#     return _django_app(environ, start_response)

from ws4redis.publisher import RedisPublisher
from ws4redis.redis_store import RedisMessage

redis_publisher = RedisPublisher(facility='foobar', broadcast=True)
message = RedisMessage('Hello World')
# and somewhere else
redis_publisher.publish_message(message)