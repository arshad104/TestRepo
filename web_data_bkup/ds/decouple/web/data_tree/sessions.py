from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse, HttpResponseRedirect

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