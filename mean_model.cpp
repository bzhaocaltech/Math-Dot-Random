#include "mean_model.hpp"
#include <stdlib.h>
#include <stdio.h>

/* Constructor for the mean model. Takes the number of movies and number
 * of users which is by default set to the defines in model.hpp */
Mean_Model::Mean_Model(int num_of_movies, int num_of_users) {
  movie_means = (float*) malloc(sizeof(float) * num_of_movies);
  user_means = (float*) malloc(sizeof(float) * num_of_users);
  this->num_of_movies = num_of_movies;
  this->num_of_users = num_of_users;
}

/* Given a list of x values in the form of (user, movie, time) predicts the
 * rating. Predicted rating of user i and movie j is
 * (user_means[i] + movie_means[i]) / 2 */
vector<float>* Mean_Model::predict(vector<int*>* x) {
  vector<float>* predictions = new vector<float>();
  for (unsigned int i = 0; i < x->size(); i++) {
    float user_rating = this->user_means[x->at(i)[0]];
    float movie_rating = this->movie_means[x->at(i)[1]];
    predictions->push_back(user_rating + movie_rating / (float) 2);
  }
  return predictions;
}

/* Fits the model given a set of data in the form of (user, movie, time,
 * rating) by filling out movie_means and user_means */
void Mean_Model::fit(vector<int*>* x) {
  int* movie_count = (int*) malloc(sizeof(int) * this->num_of_movies);
  int* user_count = (int*) malloc(sizeof(int) * this->num_of_users);
  fprintf(stderr, "Fitting model");
  // Find aggreggate of ratings
  for (unsigned int i = 0; i < x->size(); i++) {
    int user = x->at(i)[0];
    int movie = x->at(i)[1];
    float rating = (float) x->at(i)[3];
    movie_count[movie]++;
    user_count[user]++;
    this->movie_means[movie] += rating;
    this->user_means[user] += rating;
    if (i % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  // Divide to get the average rating
  for (int i = 0; i < this->num_of_movies; i++) {
    this->movie_means[i] /= movie_count[i];
  }
  for (int i = 0; i < this->num_of_users; i++) {
    this->user_means[i] /= user_count[i];
  }
  fprintf(stderr, "\n");
}

Mean_Model::~Mean_Model() {
  free(movie_means);
  free(user_means);
}
