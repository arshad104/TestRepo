

import simplejson
from django.views.generic import View
from django.http import \
    HttpResponse,\
    HttpResponseBadRequest,\
    HttpResponseNotFound,\
    HttpResponseServerError,\
    HttpResponseForbidden,\
    HttpResponseRedirect

import logging

log = logging.getLogger("django")

class ValidationError(Exception): pass
class NotFound(Exception): pass

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


class APIView(View):

    http_method_names = ['get', 'post', 'put', 'delete']
    mime_type = 'application/json; charset=utf-8'
        
    @method_decorator(csrf_exempt)
    def dispatch(self,request,*args,**kwargs):
        try:
            return super(APIView,self).dispatch(request,*args,**kwargs)
        except ValidationError,e:
            typ = 'BadRequest'
        except NotFound,e:
            typ = 'NotFound'
        except Exception,e:
            typ = 'ServerError'
            log.exception(
                "Exception in request: %s.\nREQUEST PARAMS: %s"%(e,request.REQUEST))

        error_object = {'error': '%s'%e}
        return self.render_response(error_object,status=typ)
    
    def render_response(self,json, status='OK', callback='', dump=True, indent = 4):

        st = dump and simplejson.dumps(json,indent=indent) or json

        if(callback): 
            response = callback + "(" + st + ');'
        else:
            response = st

        if(status == 'OK'): 
            http = HttpResponse
        if(status == 'BadRequest'): 
            http = HttpResponseBadRequest
        elif(status == 'NotFound'): 
            http = HttpResponseNotFound
        elif(status == 'ServerError'): 
            http = HttpResponseServerError 

        return http(response,mimetype=self.mime_type)

    def get_service(self, request, *args, **kwargs):
        uri = request.build_absolute_uri()
        req_domain = uri.split('/')[2]
        return req_domain.split(".")[0]