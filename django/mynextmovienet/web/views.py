import requests
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views.generic import CreateView

from .model import load_model, recommend_movies_for_new_user, movie_info_lookup, get_popular_movies, train

model, movie_id_to_index, index_to_movie_id = load_model(True)

# Registration
class SignUpView(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"

# Redirect to Home Page (Logged out)
def redirect_home(request):
    response = redirect('/app/popular-movies/')
    return response

# Create your views here.
def information_page(request):
    return render(request, 'recommender/information_page.html')

def get_recommendations(request):
    # Get preferred movie IDs from the request
    preferred_movie_ids = request.GET.getlist('movie_ids', [])

    # Ensure movie IDs are integers
    preferred_movie_ids = [int(movie_id) for movie_id in preferred_movie_ids]

    # Get recommendations
    recommended_movie_ids = recommend_movies_for_new_user(model, preferred_movie_ids, movie_id_to_index, index_to_movie_id, True)

    # Return recommendations as a JSON response
    return JsonResponse({'recommended_movie_ids': recommended_movie_ids})

def show_popular_movies(request):

    popular_movies = [
        {'id': 1, 'name': 'Toy Story'},
        {'id': 48, 'name': 'Pocahontas'},
        {'id': 158, 'name': 'Casper'},
        {'id': 239, 'name': 'A Goofy Movie'},
        {'id': 3114, 'name': 'Toy Story 2'},
    ]

    popular_movies = get_popular_movies()

    return render(request, 'recommender/popular_movies.html', {'popular_movies': popular_movies})

@csrf_exempt
def get_recommendations_page(request):
    if request.method == 'POST':
        selected_movies = request.POST.get('selected_movies')
        selected_movie_ids = [int(id) for id in selected_movies.split(',')]

        response = requests.get(
            'http://127.0.0.1:8000/app/recommendations/',
            params={'movie_ids':selected_movie_ids}
        )
        print(response)
        recommendations = response.json().get('recommended_movie_ids', [])

        recommendations_with_info = []
        for recommendation in recommendations:
            recommendations_with_info.append(movie_info_lookup(recommendation))

        return render(request, 'recommender/recommendations.html', {'recommendations':recommendations_with_info})
    return HttpResponse(status=405)

def train_model(request):
    if request.method == "POST":
        try:
            model.train_model_function()  # Assuming this function trains the model
            return JsonResponse({"message": "Model training started successfully!"})
        except Exception as e:
            return JsonResponse({"message": f"An error occurred: {str(e)}"})