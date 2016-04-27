from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse, HttpResponseRedirect
from awok_data import settings
from django.contrib.auth.decorators import login_required
import simplejson as json
from base import *
from views import api_server

def Login(request):
    next = request.GET.get('next', '/home')
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return HttpResponseRedirect(next)
            else:
                return HttpResponse("Inactive user.")
        else:
            return HttpResponseRedirect('/login')

    return render(request, "login.html", {'redirect_to': next})

def Logout(request):
    logout(request)
    return HttpResponseRedirect('./login')

# class Login(APIView):

#     http_method_names = ['get','post']

#     def get(self,request,*args,**kwargs):
#         return self.render_response({'authenticated': request.user.is_authenticated()})

#     def success(self):
#         return self.render_response({'success':True})

#     def post(self,request,*args,**kwargs):

#         if request.user.is_authenticated(): 
#             return self.render_response({'success': True})
        
#         params = json.loads(request.body)

#         if params.get("username"):
#             username = params.get("username")
#             password = params.get("password")
#             user = authenticate(username = username, password = password)
#             if user:
#                 login(request,user)
#                 return self.success()
#             else:
#                 raise NotFound("Email and Password do not match")
#         else:
#             raise NotFound("Please include email and password.")

# class Logout(APIView):

#     http_method_names = ['post']

#     @csrf_exempt
#     def post(self,request,*args,**kwargs):
#         response =  logout(request, *args, **kwargs)
#         return self.render_response({'success':True})