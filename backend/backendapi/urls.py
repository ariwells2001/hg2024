from django.urls import path, include
from django.contrib.auth.views import LoginView, LogoutView
from django.conf.urls.static import static
from django.conf import settings

from . import views
#from .forms import AuthenticationForm
   #############################################################################################
   ############################ URLs for Backend API ###########################################
   #############################################################################################
urlpatterns = [
   # path(r'',views.index,name='index'),
    path(r'backend/pattern/',views.patternAPI.as_view(),name="pattern"),
    path(r'backend/random/',views.randomAPI.as_view(),name="random"),

]
   #############################################################################################
   ############################## End of URLs for Backend API ##################################
   #############################################################################################
