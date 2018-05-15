/* A basic matrix factorization model */

#include "svdpp.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <fstream>
#include <math.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

/* Constructor for SVD */
SVDPP::SVDPP(int latent_factors, float eta, float reg, int num_users, int num_movies)
: SVD::SVD(latent_factors, eta, reg, num_users, num_movies) {
  fprintf(stderr, "Adding SVD++ features \n");
  this->N = new vector<int>*[num_users];
  this->y = new Matrix(num_movies, latent_factors);
}

/* Given a list of x values in the form of (user, movie, time) predicts
 * the rating */
std::vector<float>* SVDPP::predict(struct dataset* dataset) {
  vector<float>* predictions = new vector<float>();
  for (int i = 0; i < dataset->size; i++) {
      struct data data = dataset->data[i];
      float p = this->predict_one(data);
      predictions->push_back(p);
  }
  return predictions;
}

/* Predicts a single point */
float SVDPP::predict_one(struct data data) {
  int user = data.user;
  int movie = data.movie;
  // User and movie vectors
  float* Ui = this->U->row(user);
  float* Vj = this->V->row(movie);
  // Calculate the feedback factor
  float* f_factor = this->get_f_factor(user);
  // Add the f_factor to the user vector to get the adjusted user
  // vector
  float* adjusted_Ui = vec_add(Ui, f_factor, this->latent_factors);
  delete f_factor;
  // Get the prediction
  float p = dot_prod(adjusted_Ui, Vj, this->latent_factors) + this->a->at(user)
            + this->b->at(movie) + this->mu;
  delete adjusted_Ui;
  return p;
}

/* Returns the error when trying to predict a data point */
float SVDPP::get_err(struct data d) {
  float p = this->predict_one(d);
  return ((float) d.rating) - p;
}

/* Returns the neighbor factor |N(u)|^2 * sum_{j \in N(u)} y_j for a given
 * user */
float* SVDPP::get_f_factor(int user) {
  // Calculate factor |N(u)|^(-0.5)
  vector<int>* Ni = this->N[user];
  float n = pow((float) Ni->size(), -0.5);
  // Calculate sum of all implicit feedback movie vectors that the user has
  // watched
  float* sum_n = new float[this->latent_factors];
  for (int j = 0; j < this->latent_factors; j++) {
    sum_n[j] = 0;
  }
  for (unsigned int j = 0; j < Ni->size(); j++) {
    int movie_watched = Ni->at(j);
    float* yj = this->y->row(movie_watched);
    for (int i = 0; i < this->latent_factors; i++) {
      sum_n[i] += yj[i];
    }
  }
  // Multiply n * sum_n to get |N(u)|^(-0.5) * sum_(j \in N) y(j)
  float* f_factor = scalar_vec_prod(n, sum_n, this->latent_factors);
  delete sum_n;
  return f_factor;
}

void SVDPP::grad_U(struct data d, float e) {
  int user = d.user;
  int movie = d.movie;
  // User and movie vectors
  float* Ui = this->U->row(user);
  float* Vj = this->V->row(movie);
  // Calculate the regularization term
  float* reg_term = scalar_vec_prod(this->reg, Ui, this->latent_factors);
  // Calculate the error term
  float* err_term = scalar_vec_prod(e, Vj, this->latent_factors);
  // Gradient is difference
  float* grad = vec_sub(reg_term, err_term, this->latent_factors);
  delete reg_term;
  delete err_term;
  // Multiply by eta
  float* eta_grad = scalar_vec_prod(this->eta, grad, this->latent_factors);
  delete grad;
  // Descend down the gradient
  float* new_u = vec_sub(Ui, eta_grad, this->latent_factors);
  delete eta_grad;
  this->U->update_row(user, new_u);
}

void SVDPP::grad_V(struct data d, float e) {
  int user = d.user;
  int movie = d.movie;
  // User and movie vectors
  float* Ui = this->U->row(user);
  float* Vj = this->V->row(movie);
  // Calculate the regularization term
  float* reg_term = scalar_vec_prod(this->reg, Vj, this->latent_factors);
  // Calculate the error term
  float* f_factor = this->get_f_factor(user);
  float* adjusted_Ui = vec_add(Ui, f_factor, this->latent_factors);
  float* err_term = scalar_vec_prod(e, adjusted_Ui, this->latent_factors);
  delete adjusted_Ui;
  delete f_factor;
  // Gradient is difference
  float* grad = vec_sub(reg_term, err_term, this->latent_factors);
  delete reg_term;
  delete err_term;
  // Multiply by eta
  float* eta_grad = scalar_vec_prod(this->eta, grad, this->latent_factors);
  delete grad;
  // Descend down the gradient
  float* new_v = vec_sub(Vj, eta_grad, this->latent_factors);
  delete eta_grad;
  this->V->update_row(movie, new_v);
}

void SVDPP::grad_a(struct data d, float e) {
  int user = d.user;
  float reg_term = this->a->at(user) * this->reg;
  float eta_grad = this->eta * (reg_term - e);
  float value = this->a->at(user) - eta_grad;
  this->a->update_element(user, value);
}

void SVDPP::grad_b(struct data d, float e) {
  int movie = d.movie;
  float reg_term = this->b->at(movie) * this->reg;
  float eta_grad = this->eta * (reg_term - e);
  float value = this->b->at(movie) - eta_grad;
  this->b->update_element(movie, value);
}

// Takes in a movie that the user has watched (movie_watched does not
// necessarily = d.movie)
void SVDPP::grad_y(struct data d, int movie_watched, float e) {
  int user = d.user;
  int movie = d.movie;
  float* yj = this->y->row(movie_watched);
  float* Vj = this->V->row(movie);
  // Find regularization term
  float* reg_term = scalar_vec_prod(this->reg, yj, this->latent_factors);
  // Find error term
  float n = pow((float) this->N[user]->size(), -0.5);
  float* temp = scalar_vec_prod(n, Vj, this->latent_factors);
  float* err_term = scalar_vec_prod(e, temp, this->latent_factors);
  delete temp;
  // Gradient is difference
  float* grad = vec_sub(reg_term, err_term, this->latent_factors);
  delete reg_term;
  delete err_term;
  // Multiply by eta
  float* eta_grad = scalar_vec_prod(this->eta, grad, this->latent_factors);
  delete grad;
  // Descend down the gradient
  float* new_y = vec_sub(yj, eta_grad, this->latent_factors);
  delete eta_grad;
  this->y->update_row(movie_watched, new_y);
}

/* Run a gradient on part of the dataset */
void SVDPP::grad_part(struct dataset* ds, bool track_progress) {
  int dot_break = ds->size / 30;
  for (int n = 0; n < ds->size; n++) {
    struct data data = ds->data[n];
    float error = this->get_err(data);
    this->grad_U(data, error);
    this->grad_V(data, error);
    this->grad_a(data, error);
    this->grad_b(data, error);
    int user = data.user;
    for (unsigned int u = 0; u < this->N[user]->size(); u++) {
      int movie_watched = this->N[user]->at(u);
      // TODO: The if statement is only here to reduce runtime
      if (movie_watched == data.movie) {
      this->grad_y(data, movie_watched, error);
      }
    }
    if ((n % dot_break == 0) && track_progress) {
      fprintf(stderr, ".");
    }
  }
}

void SVDPP::fit(struct dataset* dataset, int epochs, int num_threads) {
  fprintf(stderr, "Fitting the data of size %i\n", dataset->size);
  fprintf(stderr, "Using %i threads\n", num_threads);

  // Calculate the global bias
  this->mu = 0;
  fprintf(stderr, "Calculating the global bias\n");
  for (int i = 0; i < dataset->size; i++) {
    int rating = dataset->data[i].rating;
    this->mu += (double) rating;
  }
  this->mu /= (double) dataset->size;
  fprintf(stderr, "Global bias was %f\n", this->mu);

  // Initializing N
  fprintf(stderr, "Initializing N \n");
  for (int i = 0; i < this->num_users; i++) {
    this->N[i] = new std::vector<int>();
  }
  for (int i = 0; i < dataset->size; i++) {
    int user = dataset->data[i].user;
    int movie = dataset->data[i].movie;
    this->N[user]->push_back(movie);
  }
  fprintf(stderr, "N initialized \n");

  fprintf(stderr, "Running %i epochs\n", epochs);
  // Initialize U, V, a, b randomly
  fprintf(stderr, "Randomly initializing matrices\n");
  for (int i = 0; i < this->num_users; i++) {
    for (int j = 0; j < this->latent_factors; j++) {
      float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
      this->U->set_val(i, j, random);
    }
  }
  for (int i = 0; i < this->num_movies; i++) {
    for (int j = 0; j < this->latent_factors; j++) {
      float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
      this->V->set_val(i, j, random);
    }
  }
  for (int i = 0; i < this->num_users; i++) {
    float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
    this->a->update_element(i, random);
  }
  for (int j = 0; j < this->num_movies; j++) {
    float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
    this->b->update_element(j, random);
  }
  for (int i = 0; i < this->num_movies; i++) {
    for (int j = 0; j < this->latent_factors; j++) {
      float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
      this->y->set_val(i, j, random);
    }
  }
  fprintf(stderr, "Matrices randomly initialized\n");

  // Split the dataset
  struct dataset** threaded_dataset = split_dataset(dataset, num_threads);

  for (int curr_epoch = 0; curr_epoch < epochs; curr_epoch++) {
    fprintf(stderr, "Running epoch %i", curr_epoch + 1);

    // Create the threads
    std::thread threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
      if (i == 0) {
        threads[i] = std::thread(&SVDPP::grad_part, this, threaded_dataset[i], true);
      }
      else {
        threads[i] = std::thread(&SVDPP::grad_part, this, threaded_dataset[i], false);
      }
    }

    // Join the threads back together
    for (int i = 0; i < num_threads; i++) {
      threads[i].join();
    }

    fprintf(stderr, "\n");
  }
}

SVDPP::~SVDPP() {
  for (int i = 0; i < num_users; i++) {
    delete this->N[i];
  }
  delete this->N;
  delete this->y;
}
