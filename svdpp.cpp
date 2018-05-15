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
  float p = 0;
  for (int i = 0; i < this->latent_factors; i++) {
    float adjusted_Ui = Ui[i] - f_factor[i];
    p += adjusted_Ui * Vj[i];
  }
  delete f_factor;
  p += b->at(movie) + a->at(user) + this->mu;
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
  if (std::isinf(n)) {
    fprintf(stderr, "n in get f_factor is inf \n");
    exit(1);
  }
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
      if (std::isinf(yj[i])) {
        fprintf(stderr, "yji was %f\n", yj[i]);
        exit(1);
      }
    }
  }
  // Multiply n * sum_n to get |N(u)|^(-0.5) * sum_(j \in N) y(j)
  float* f_factor = scalar_vec_prod(n, sum_n, this->latent_factors);
  delete sum_n;
  for (int i = 0; i < this->latent_factors; i++) {
    if (std::isnan(f_factor[i]) || std::isinf(f_factor[i])) {
      fprintf(stderr, "is nan in getting f_factor\n");
      fprintf(stderr, "value was %f\n", f_factor[i]);
      fprintf(stderr, "n was %f\n", n);
      exit(1);
    }
  }
  return f_factor;
}

void SVDPP::grad_U(struct data d, float e) {
  int user = d.user;
  int movie = d.movie;
  // User and movie vectors
  float* Ui = this->U->row(user);
  float* Vj = this->V->row(movie);
  float* new_u = new float[this->latent_factors];
  for (int i = 0; i < this->latent_factors; i++) {
    float reg_term = this->reg * Ui[i];
    float err_term = e * Vj[i];
    float grad = eta * (reg_term - err_term);
    new_u[i] = Ui[i] - grad;
  }
  for (int i = 0; i < this->latent_factors; i++) {
    if (std::isnan(new_u[i]) || std::isinf(new_u[i])) {
      fprintf(stderr, "is nan in getting grad_u\n");
      exit(1);
    }
  }
  this->U->update_row(user, new_u);
}

void SVDPP::grad_V(struct data d, float e) {
  int user = d.user;
  int movie = d.movie;
  // User and movie vectors
  float* Ui = this->U->row(user);
  float* Vj = this->V->row(movie);
  float* f_factor = this->get_f_factor(user);
  float* new_v = new float[this->latent_factors];
  for (int i = 0; i < this->latent_factors; i++) {
    float reg_term = this->reg * Vj[i];
    float err_term = e * (Ui[i] + f_factor[i]);
    float grad = eta * (reg_term - err_term);
    new_v[i] = Vj[i] - grad;
  }
  delete f_factor;
  for (int i = 0; i < this->latent_factors; i++) {
    if (std::isnan(new_v[i]) || std::isinf(new_v[i])) {
      fprintf(stderr, "is nan in getting grad_v\n");
      exit(1);
    }
  }
  this->V->update_row(movie, new_v);
}

void SVDPP::grad_a(struct data d, float e) {
  int user = d.user;
  float reg_term = this->a->at(user) * this->reg;
  float eta_grad = this->eta * (reg_term - e);
  float value = this->a->at(user) - eta_grad;
  if (std::isnan(value) || std::isinf(value)) {
    fprintf(stderr, "is nan in getting grad_a\n");
    exit(1);
  }
  this->a->update_element(user, value);
}

void SVDPP::grad_b(struct data d, float e) {
  int movie = d.movie;
  float reg_term = this->b->at(movie) * this->reg;
  float eta_grad = this->eta * (reg_term - e);
  float value = this->b->at(movie) - eta_grad;
  if (std::isnan(value) || std::isinf(value)) {
    fprintf(stderr, "is nan in getting grad_b\n");
    exit(1);
  }
  this->b->update_element(movie, value);
}

// Takes in a movie that the user has watched (movie_watched does not
// necessarily = d.movie)
void SVDPP::grad_y(struct data d, int movie_watched, float e) {
  int user = d.user;
  int movie = d.movie;
  float* yj = this->y->row(movie_watched);
  float* Vj = this->V->row(movie);
  float error_const = pow((float) this->N[user]->size(), -0.5) * e;
  if (std::isnan(error_const)) {
    fprintf(stderr, "is nan in getting error_const\n");
    exit(1);
  }
  float* new_y = new float[this->latent_factors];
  for (int i = 0; i < this->latent_factors; i++) {
    float reg_term = this->reg * yj[i];
    float err_term = error_const * Vj[i];
    float grad = eta * (reg_term - err_term);
    if (std::isnan(reg_term) || std::isinf(reg_term)) {
      fprintf(stderr, "is nan in getting reg\n");
      exit(1);
    }
    if (std::isnan(err_term) || std::isinf(reg_term)) {
      fprintf(stderr, "is nan in getting err\n");
      exit(1);
    }
    if (std::isnan(grad) || std::isinf(grad)) {
      fprintf(stderr, "is nan in getting grad\n");
      fprintf(stderr, "reg term was %f\n", reg_term);
      fprintf(stderr, "err term was %f\n", err_term);
      exit(1);
    }
    new_y[i] = yj[i] - grad;
  }
  for (int i = 0; i < this->latent_factors; i++) {
    if (std::isnan(new_y[i]) || std::isinf(new_y[i])) {
      fprintf(stderr, "is nan in getting grad_y\n");
      exit(1);
    }
  }
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
      this->grad_y(data, movie_watched, error);
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
  fprintf(stderr, "Running %i epochs\n", epochs);

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
