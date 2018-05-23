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
  this->y = new Matrix<float>(num_movies, latent_factors);
}

/* Given a list of x values in the form of (user, movie, time) predicts
 * the rating */
std::vector<float>* SVDPP::predict(struct dataset* dataset) {
  /* Loop through the dataset to update N */
  for (int i = 0; i < dataset->size; i++) {
    struct data data = dataset->data[i];
    int movie = data.movie;
    int user = data.user;
    bool found = false;
    for (unsigned int i = 0; i < this->N[user]->size(); i++) {
      if (movie == this->N[user]->at(i)) {
        found = true;
      }
    }
    if (!found) {
      this->N[user]->push_back(movie);
    }
  }
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
    float adjusted_Ui = Ui[i] + f_factor[i];
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

/* Run a gradient on part of the dataset */
void SVDPP::grad_part(struct dataset* ds, bool track_progress) {
  int dot_break = ds->size / 30;
  for (int n = 0; n < ds->size; n++) {
    struct data data = ds->data[n];
    int user = data.user;
    int movie = data.movie;
    float* Ui = this->U->row(user);
    float* Vj = this->V->row(movie);
    float* f_factor = this->get_f_factor(user);
    // Calculate the current prediction
    float curr_p = 0;
    for (int i = 0; i < this->latent_factors; i++) {
      float adjusted_Ui = Ui[i] + f_factor[i];
      curr_p += adjusted_Ui * Vj[i];
    }
    curr_p += b->at(movie) + a->at(user) + this->mu;
    // Calculate the error
    float error = ((float) data.rating) - curr_p;
    // Update U and V
    float* new_u = new float[this->latent_factors];
    float* new_v = new float[this->latent_factors];
    for (int i = 0; i < this->latent_factors; i++) {
      // Update V
      float V_reg_term = this->reg * Vj[i];
      float V_err_term = error * (Ui[i] + f_factor[i]);
      float V_grad = eta * (V_reg_term - V_err_term);
      new_v[i] = Vj[i] - V_grad;
      // Update U
      float U_reg_term = this->reg * Ui[i];
      float U_err_term = error * Vj[i];
      float U_grad = eta * (U_reg_term - U_err_term);
      new_u[i] = Ui[i] - U_grad;
    }
    delete f_factor;
    this->U->update_row(user, new_u);
    this->V->update_row(movie, new_v);
    // Update a
    float a_reg_term = this->a->at(user) * this->reg;
    float a_eta_grad = this->eta * (a_reg_term - error);
    float a_value = this->a->at(user) - a_eta_grad;
    this->a->update_element(user, a_value);
    // Update b
    float b_reg_term = this->b->at(movie) * this->reg;
    float b_eta_grad = this->eta * (b_reg_term - error);
    float b_value = this->b->at(movie) - b_eta_grad;
    this->b->update_element(movie, b_value);
    // Loop through to update the y vectors
    float error_const = pow((float) this->N[user]->size(), -0.5) * error;
    for (unsigned int u = 0; u < this->N[user]->size(); u++) {
      int movie_watched = this->N[user]->at(u);
      float* yj = this->y->row(movie_watched);
      float* new_y = new float[this->latent_factors];
      for (int i = 0; i < this->latent_factors; i++) {
        float reg_term = this->reg * yj[i];
        float err_term = error_const * Vj[i];
        float grad = eta * (reg_term - err_term);
        new_y[i] = yj[i] - grad;
      }
      this->y->update_row(movie_watched, new_y);
    }
    // Track progress
    if ((n % dot_break == 0) && track_progress) {
      fprintf(stderr, ".");
    }
  }
}

void SVDPP::fit(struct dataset* dataset, int epochs, struct dataset* validation_set, int num_threads) {
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

  float original_eta = this->eta;
  float prev_RMSE = 5;
  int failed_epochs = 0;
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

    // Reduce the learning rate
    this->eta *= 0.9;

    fprintf(stderr, "\n");

    // Get error on validation set
    if (validation_set != NULL) {
        float score = this->score(validation_set);
        fprintf(stderr, "RMSE on validation set was: %f \n", score);
        if (curr_epoch == 0) {
          prev_RMSE = score;
        }
        // Early stopping
        else if (prev_RMSE < score + 0.0005) {
          failed_epochs++;
        }
        else {
          failed_epochs = 0;
        }
        // If we have three consecutive failures to decrease RMSE, stop
        if (failed_epochs == 3) {
          fprintf(stderr, "RMSE did not decrease for three consecutive epochs. Stopping early\n");
          break;
        }
        prev_RMSE = score;
    }
  }
  this->eta = original_eta;
}

SVDPP::~SVDPP() {
  for (int i = 0; i < num_users; i++) {
    delete this->N[i];
  }
  delete this->N;
  delete this->y;
}

template class Matrix<int>;
template class Matrix<float>;
