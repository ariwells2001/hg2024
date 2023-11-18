from django.db import models
from django.contrib.auth.models import User

class patternTable(models.Model):
    account = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    co2 = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    door = models.FloatField()
    motion = models.FloatField()
    fp2 = models.FloatField()
    occupancy = models.FloatField()
    user = models.ForeignKey(User,on_delete=models.CASCADE)

    class Meta:
        db_table = 'patternTable'
    def __str__(self):
        return self.account