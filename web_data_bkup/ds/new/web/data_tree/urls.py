from django.conf.urls import patterns, url
from django.contrib.auth.decorators import login_required
from views import *
from sessions import *

urlpatterns = patterns('',
  url(r'^graph', GraphAPI.as_view()),
  url(r'^node', NodesAPI.as_view()),
  url(r'^link', LinkesAPI.as_view()),
  url(r'^attr', Attributes.as_view()),
  url(r'^model', ModelAPI.as_view()),
  url(r'^api', ProcessAPI.as_view()),
  url(r'^login', Login),
  url(r'^logout', Logout),
  url(r'^errors', output_console),
  url(r'^home', home)
)