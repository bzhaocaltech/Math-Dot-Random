#include "better_blend.hpp"

better_blend::better_blend(float eta, float user_reg, float movie_reg, int num_models,
  float* avg_constants, vector<float>** model_probe, vector<float>** model_qual,
  int num_movies, int num_users) {
  this->eta = eta;
  this->user_reg = user_reg;
  this->movie_reg = movie_reg;
  this->num_models = num_models;
  this->avg_constants = avg_constants;
  this->model_probe = model_probe;
  this->model_qual = model_qual;
  this->num_movies = num_movies;
  this->num_users = num_users;
  this->movie_constants = new Matrix<float>(num_movies, num_models);
  for (int i = 0; i < num_movies; i++) {
    for (int j = 0; j < num_models; j++) {
      movie_constants->set_val(i, j, avg_constants[j]);
    }
  }
  this->user_constants = new Matrix<float>(num_users, num_models);
  for (int i = 0; i < num_users; i++) {
    for (int j = 0; j < num_models; j++) {
      user_constants->set_val(i, j, avg_constants[j]);
    }
  }
  this->user_biases = new Vector(num_users);
  this->movie_biases = new Vector(num_movies);
  for (int i = 0; i < num_users; i++) {
    user_biases->update_element(i, 0);
  }
  for (int i = 0; i < num_movies; i++) {
    movie_biases->update_element(i, 0);
  }
  fprintf(stderr, "Blending with %d models user_reg %f, movie_reg %f, and eta %f\n",
  num_models, user_reg, movie_reg, eta);
}

vector<float>* better_blend::predict(struct dataset* qual) {
  vector<float>* predictions = new vector<float>();
  for (int i = 0; i < qual->size; i++) {
    struct data data = qual->data[i];
    float* user_c = user_constants->row(data.user);
    float* movie_c = movie_constants->row(data.movie);
    float user_prediction = 0;
    float movie_prediction = 0;
    float movie_bias = movie_biases->at(data.movie);
    float user_bias = user_biases->at(data.user);
    for (int j = 0; j < num_models; j++) {
      user_prediction += user_c[j] * model_qual[j]->at(i);
      movie_prediction += movie_c[j] * model_qual[j]->at(i);
    }
    float prediction = (user_prediction + movie_prediction) / 2;
    prediction += movie_bias + user_bias;
    if (prediction > 5) {
      prediction = 5;
    }
    if (prediction < 1) {
      prediction = 1;
    }
    predictions->push_back(prediction);
  }
  return predictions;
}

void better_blend::fit(struct dataset* probe, int epochs) {
  for (int n = 0; n < epochs; n++) {
    fprintf(stderr, "Running epoch %d\n", n);
    for (int i = 0; i < probe->size; i++) {
      float* model_ratings = new float[num_models];
      for (int j = 0; j < num_models; j++) {
        model_ratings[j] = model_probe[j]->at(i);
      }
      grad_one(probe->data[i], model_ratings);
      delete model_ratings;
    }
  }
}

void better_blend::grad_one(struct data data, float* model_ratings) {
  // Find the current error
  float user_prediction = 0;
  float movie_prediction = 0;
  float* user_c = user_constants->row(data.user);
  float* movie_c = movie_constants->row(data.movie);
  float movie_bias = movie_biases->at(data.movie);
  float user_bias = user_biases->at(data.user);
  for (int i = 0; i < num_models; i++) {
    user_prediction += user_c[i] * model_ratings[i];
    movie_prediction += movie_c[i] * model_ratings[i];
  }
  float prediction = (user_prediction + movie_prediction) / 2;
  prediction += movie_bias + user_bias;
  float error = data.rating - prediction;

  // Find the gradients
  float* new_user_c = new float[num_models];
  float* new_movie_c = new float[num_models];
  for (int i = 0; i < num_models; i++) {
    // For users
    float reg_term = user_reg * (user_c[i] - avg_constants[i]);
    float err_term = -model_ratings[i] * error;
    float grad = eta * (reg_term + err_term);
    new_user_c[i] = user_c[i] - grad;

    // For movies
    reg_term = movie_reg * (movie_c[i] - avg_constants[i]);
    err_term = -model_ratings[i] * error;
    grad = eta * (reg_term + err_term);
    new_movie_c[i] = movie_c[i] - grad;

  }
  // For biases
  float reg_term = user_reg * user_bias;
  float err_term = -error;
  float grad = eta * (reg_term + err_term);
  float new_user_bias = user_bias - grad;

  reg_term = movie_reg * movie_bias;
  err_term = -error;
  grad = eta * (reg_term + err_term);
  float new_movie_bias = movie_bias - grad;

  // user_constants->update_row(data.user, new_user_c);
  // movie_constants->update_row(data.movie, new_movie_c);
  user_biases->update_element(data.user, new_user_bias);
  movie_biases->update_element(data.movie, new_movie_bias);
}
