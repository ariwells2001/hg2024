from rest_framework import serializers
from .models import patternTable

class patternReadonlySerializer(serializers.ModelSerializer):
    class Meta:
        model = patternTable
        fields = '__all__'

class patternValueSerializer(serializers.ModelSerializer):
    class Meta:
        model = patternTable
        fields = ['account', 'timestamp','co2','temperature', 'humidity', 'door', 'motion','fp2','occupancy']

class patternSerializer(serializers.ModelSerializer):
    class Meta:
        model = patternTable
        fields = '__all__'