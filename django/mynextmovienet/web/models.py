from django.db import models

# Create your models here.
class MovieInfo(models.Model):
    ml_movieId = models.IntegerField()
    title = models.CharField(max_length=500)
    ml_genres = models.CharField(max_length=500)
    tmdb_id = models.IntegerField()
    tmdb_date = models.IntegerField()
    tmdb_tagline = models.CharField(max_length=500)
    tmdb_description = models.CharField(max_length=500)
    tmdb_minute = models.IntegerField()
    tmdb_rating = models.FloatField()
    link = models.CharField(max_length=500)

    def __str__(self):
        return self.ml_movieId

class MoviePicks(models.Model):
    user = models.CharField(max_length=500)
    picks = models.CharField(max_length=5000)

    def __str__(self):
        return self.picks
    