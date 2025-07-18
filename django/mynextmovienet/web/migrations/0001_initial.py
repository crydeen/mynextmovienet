# Generated by Django 4.0.6 on 2024-11-11 02:10

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MovieInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ml_movieId', models.IntegerField()),
                ('title', models.CharField(max_length=500)),
                ('ml_genres', models.CharField(max_length=500)),
                ('tmdb_id', models.IntegerField()),
                ('tmdb_date', models.IntegerField()),
                ('tmdb_tagline', models.CharField(max_length=500)),
                ('tmdb_description', models.CharField(max_length=500)),
                ('tmdb_minute', models.IntegerField()),
                ('tmdb_rating', models.FloatField()),
                ('link', models.CharField(max_length=500)),
            ],
        ),
        migrations.CreateModel(
            name='MoviePicks',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.CharField(max_length=500)),
                ('picks', models.CharField(max_length=5000)),
            ],
        ),
    ]
