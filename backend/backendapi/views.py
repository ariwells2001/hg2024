# From Original Django Framework
from warnings import resetwarnings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

# django packages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
from django.db import IntegrityError
from django.core import serializers
from django.contrib.auth.models import User

# rest_framework packages
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import permissions, generics
from rest_framework.authentication import TokenAuthentication
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view
from rest_framework.decorators import authentication_classes, permission_classes

from apscheduler.schedulers.background import BackgroundScheduler
import time 
import requests
import json
import hashlib
import random

#from .forms import SignupForm
#from .authentication import myToken, reToken, getKeyVar, Timing
from .models import patternTable
from .serializers import patternSerializer

###########################################################################################
############################### Lastest Backend API #######################################
###########################################################################################
class randomAPI(generics.ListAPIView):
    serializer_class = patternSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        user = self.request.user
        print(user)
        print("Authorization key of the request headers is {}".format(self.request.META.get('HTTP_AUTHORIZATION')))
        rows =int(self.request.META.get('HTTP_DN'))
        rn = random.randint(0,3900)
        print(patternTable.objects.filter(user=user).order_by('id')[rn-1:rn])
        return patternTable.objects.filter(user=user).order_by('id')[rn-1:rn]

class patternAPI(generics.ListAPIView):
    serializer_class = patternSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        user = self.request.user
        print("Authorization key of the request headers is {}".format(self.request.META.get('HTTP_AUTHORIZATION')))
        rows =int(self.request.META.get('HTTP_DN'))
        print(patternTable.objects.filter(user=user).order_by('-id')[0:rows])
        print(rows)
        return patternTable.objects.filter(user=user).order_by('-id')[0:rows]

