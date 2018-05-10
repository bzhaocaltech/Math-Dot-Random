#include "mean_model.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

/* Constructor for the mean model. Takes the number of movies and number
 * of users which is by default set to the defines in model.hpp */
Mean_Model::Mean_Model(int num_of_movies, int num_of_users) {
  movie_means = (float*) malloc(sizeof(float) * num_of_movies);
  user_means = (float*) malloc(sizeof(float) * num_of_users);
  this->num_of_movies = num_of_movies;
  this->num_of_users = num_of_users;
}

/* Constructs a mean model from a serialized file */
Mean_Model::Mean_Model(string file) {
  fprintf(stderr, "Constructing mean model from %s\n", file.c_str());
  ifstream ifs(file);
  boost::archive::binary_iarchive ia(ifs);

  // Unserialize num_of_movies and num_of_users
  ia & num_of_movies;
  ia & num_of_users;

  // Unserialize movie_means
  movie_means = (float*) malloc(sizeof(float) * num_of_movies);
  for (int i = 0; i < num_of_movies; i++) {
    ia & movie_means[i];
  }

  // Unserialize user_means
  user_means = (float*) malloc(sizeof(float) * num_of_users);
  for (int i = 0; i < num_of_users; i++) {
    ia & user_means[i];
  }
}

/* Serializes the mean_model into a given file */
void Mean_Model::serialize(string file) {
  ofstream ofs(file);
  boost::archive::binary_oarchive oa(ofs);

  // Serialize parameters of model
  oa & num_of_movies;
  oa & num_of_users;

  // Serialize movie_means
  for (int i = 0; i < num_of_movies; i++) {
    oa & movie_means[i];
  }

  // Unserialize user_means
  for (int i = 0; i < num_of_users; i++) {
    oa & user_means[i];
  }
}


/* Given a list of x values in the form of (user, movie, time) predicts the
 * rating. Predicted rating of user i and movie j is
 * (user_means[i] + movie_means[i]) / 2 */
vector<float>* Mean_Model::predict(struct dataset* dataset) {
  vector<float>* predictions = new vector<float>();
  for (int i = 0; i < dataset->size; i++) {
    int user = dataset->data[i].user;
    int movie = dataset->data[i].movie;
    float user_rating = this->user_means[user];
    float movie_rating = this->movie_means[movie];
    predictions->push_back((user_rating + movie_rating) / (float) 2);
  }
  return predictions;
}

/* Fits the model given a set of data in the form of (user, movie, time,
 * rating) by filling out movie_means and user_means */
void Mean_Model::fit(struct dataset* dataset) {
  fprintf(stderr, "Fitting model");
  int* movie_count = (int*) malloc(sizeof(int) * this->num_of_movies);
  int* user_count = (int*) malloc(sizeof(int) * this->num_of_users);
  // Find aggreggate of ratings
  for (int i = 0; i < dataset->size; i++) {
    int user = dataset->data[i].user;
    int movie = dataset->data[i].movie;
    float rating = (float) dataset->data[i].rating;
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
    this->movie_means[i] /= (float) movie_count[i];
  }
  for (int i = 0; i < this->num_of_users; i++) {
    this->user_means[i] /= (float) user_count[i];
  }
  free(movie_count);
  free(user_count);

  fprintf(stderr, "\n");
}

Mean_Model::~Mean_Model() {
  free(this->movie_means);
  free(this->user_means);
}
