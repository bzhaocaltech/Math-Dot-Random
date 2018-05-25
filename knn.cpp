#include "knn.hpp"
#include <math.h>
#include <thread>

/* Constructor for KNN
 * n_size is the size of the neighborhood */
KNN::KNN(int n_size, int alpha, float e, float min_pearson, int num_threads, int num_users, int num_movies) {
  this->n_size = n_size;
  this->num_users = num_users;
  this->num_movies = num_movies;
  this->num_threads = num_threads;
  this->alpha = alpha;
  this->e = e;
  this->min_pearson = min_pearson;
  corr = new Matrix<float>(num_movies, num_movies);
  sorted_corr = new vector<int>*[num_movies];
  user_index = new int[num_users];
  movie_index = new int[num_movies];
  fprintf(stderr, "Creating KNN using Pearson correlation with n_size = %i\n", n_size);
  fprintf(stderr, "Using %d threads\n", num_threads);
  fprintf(stderr, "Using parameter alpha = %d, exp = %f, and min_pearson = %f\n", alpha, e, min_pearson);
}

/* Returns the neighborhood size */
int KNN::get_n_size() {
  return n_size;
}

/* Set the neighborhood size */
void KNN::set_n_size(int n_size) {
  this->n_size = n_size;
  fprintf(stderr, "Setting neighborhood size to %i\n", n_size);
}

/* Returns e */
float KNN::get_e() {
  return e;
}

/* Set e */
void KNN::set_e(float e) {
  this->e = e;
  fprintf(stderr, "Setting e to %f\n", e);
}

/* Quicksort indices based on corr to create sorted_corr */
vector<int>* KNN::quicksort_corr(vector<int>* vec, float* values) {
  // Base case
  if (vec->size() == 1 || vec->size() == 0) {
    return vec;
  }
  int pivot = vec->at(0);
  float pivot_value = values[pivot];
  // Split the dataset into low and high vec
  vector<int>* low = new vector<int>();
  vector<int>* high = new vector<int>();
  for (unsigned int i = 1; i < vec->size(); i++) {
    int index = vec->at(i);
    float value = values[index];
    if (value <= pivot_value) {
      low->push_back(index);
    }
    else {
      high->push_back(index);
    }
  }
  delete vec;
  // Recursive calls
  vector<int>* sorted_low = quicksort_corr(low, values);
  vector<int>* sorted_high = quicksort_corr(high, values);
  // Combine the two sorted lists
  vector<int>* sorted_all = new vector<int>();
  for (unsigned int i = 0; i < sorted_low->size(); i++) {
    sorted_all->push_back(sorted_low->at(i));
  }
  sorted_all->push_back(pivot);
  for (unsigned int i = 0; i < sorted_high->size(); i++) {
    sorted_all->push_back(sorted_high->at(i));
  }
  delete sorted_low;
  delete sorted_high;
  return sorted_all;
}

/* Predict a single datapoint */
float KNN::predict_one(struct data data) {
  float* correlation_values = corr->row(data.movie);
  vector<int>* movie_correlations = sorted_corr[data.movie];
  // Indices we need to search between to see if a user has rated a movie
  int user_start_index = user_index[data.user];
  int user_end_index;
  if (data.user != (unsigned int) num_users - 1) {
    user_end_index = user_index[data.user + 1];
  }
  else {
    user_end_index = training_set->size;
  }
  // Get the nearest neighbors
  int neighbors_found = 0;
  float numer = 0;
  float denom = 0;
  for (int i = movie_correlations->size() - 1; i >= 0; i--) {
    int movie_neighbor = movie_correlations->at(i);
    // Find user's rating of movie_neighbor (if exists)
    int movie_neighbor_rating = -1;
    for (int j = user_start_index; j < user_end_index; j++) {
      if (movie_neighbor == training_set->data[j].movie) {
        movie_neighbor_rating = training_set->data[j].rating;
        break;
      }
    }
    // Update values if the user has watched that movie
    if (movie_neighbor_rating != -1) {
      neighbors_found++;
      float adjusted_corr = pow(correlation_values[movie_neighbor], e);
      numer += adjusted_corr * (float) movie_neighbor_rating;
      denom += adjusted_corr;
      // If we found enough neighbors, stop
      if (neighbors_found == n_size) {
        break;
      }
    }
  }
  if (denom == 0) {
    return 3;
  }
  return (numer / (denom));
}

/* Given a list of x values in the form of (user, movie, time) predicts
 * the rating */
std::vector<float>* KNN::predict(struct dataset* dataset) {
  fprintf(stderr, "Predicting data of size %i", dataset->size);
  vector<float>* predictions = new vector<float>();
  float* arr_predictions = new float[dataset->size];
  // Divide the dataset
  int* thread_ends = new int[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_ends[i] = (i + 1) * (dataset->size / num_threads);
  }
  // Create the threads
  std::thread threads[num_threads];
  for (int i = 0; i < num_threads; i++) {
    if (i == 0) {
      threads[i] = std::thread(&KNN::predict_part, this, 0, thread_ends[i], dataset, arr_predictions, true);
    }
    else {
      if (i == num_threads - 1) {
        threads[i] = std::thread(&KNN::predict_part, this, thread_ends[i - 1], dataset->size, dataset, arr_predictions, false);
      }
      else {
        threads[i] = std::thread(&KNN::predict_part, this, thread_ends[i - 1], thread_ends[i], dataset, arr_predictions, false);
      }
    }
  }
  // Join the threads back together
  for (int i = 0; i < num_threads; i++) {
      threads[i].join();
  }
  fprintf(stderr, "\n");
  // Convert arr_predictions into a vector
  for (int i = 0; i < dataset->size; i++) {
    predictions->push_back(arr_predictions[i]);
  }
  delete arr_predictions;
  delete thread_ends;
  return predictions;
}

void KNN::predict_part(int start, int end, struct dataset* dataset, float* predictions, bool track_progress) {
  int print_dot = (end - start) / 30;
  for (int i = start; i < end; i++) {
    if (track_progress && (i % print_dot == 0)) {
      fprintf(stderr, ".");
    }
    struct data data = dataset->data[i];
    float p = this->predict_one(data);
    predictions[i] = p;
  }
}

/* Fit on a portion of the data */
void KNN::fit_part(int start, int end, struct dataset* mu_train, struct dataset* um_train, bool track_progress) {
  int print_dot = (end - start) / 30;
  for (int movie = start; movie < end; movie++) {
    if (track_progress && ((movie - start) % print_dot == 0)) {
      fprintf(stderr, ".");
    }
    // Set up the pearson intermediates
    struct pearson* intermediates = new struct pearson[num_movies];
    for (int j = 0; j < num_movies; j++) {
      initialize_pearson(&intermediates[j]);
    }
    // Set up starting and ending indices to search through
    int movie_start_index = movie_index[movie];
    int movie_end_index;
    if (movie != num_movies - 1) {
      movie_end_index = movie_index[movie + 1];
    }
    else {
      movie_end_index = mu_train->size;
    }

    // For every user that watched movie i:
    for (int i = movie_start_index; i < movie_end_index; i++) {
      struct data movie_data = mu_train->data[i];
      int user = movie_data.user;
      // Set up indices for the user
      int user_start_index = user_index[user];
      int user_end_index;
      if (user != num_users - 1) {
        user_end_index = user_index[user + 1];
      }
      else {
        user_end_index = um_train->size;
      }
      // For every other movie the user rated, update pearson for (movie, other_movie)
      for (int l = user_start_index; l < user_end_index; l++) {
        struct data u_data = um_train->data[l];
        update_pearson(&intermediates[u_data.movie], movie_data.rating, u_data.rating);
      }
    }

    // Calculate correlations and update corr
    float* movie_correlations = new float[num_movies];
    // Also calculate sorted corr
    vector<int>* to_sort = new vector<int>();
    for (int i = 0; i < num_movies; i++) {
      movie_correlations[i] = this->calculate_corr(&intermediates[i]);
      // Ignore correlations that are below are equal to 0
      if (movie_correlations[i] > 0) {
        to_sort->push_back(i);
      }
    }
    sorted_corr[movie] = quicksort_corr(to_sort, movie_correlations);
    corr->update_row(movie, movie_correlations);
    delete intermediates;
  }
}

/* Given two datasets, one sorted by users and the other sorted by movies,
 * calculates the correlation between all movies */
void KNN::fit(struct dataset* um_train, struct dataset* mu_train) {
  fprintf(stderr, "Fitting the data of size %i\n", um_train->size);

  // Save um train
  this->training_set = um_train;

  // Initialize movie and user index
  fprintf(stderr, "Initializing movie and user index\n");
  unsigned int curr_index = 0;
  movie_index[curr_index] = 0;
  for (int i = 0; i < mu_train->size; i++) {
    struct data data = mu_train->data[i];
    if (data.movie != curr_index) {
      curr_index++;
      movie_index[curr_index] = i;
    }
  }
  curr_index = 0;
  user_index[curr_index] = 0;
  for (int i = 0; i < um_train->size; i++) {
    struct data data = um_train->data[i];
    if (data.user != curr_index) {
      curr_index++;
      user_index[curr_index] = i;
    }
  }
  fprintf(stderr, "Movie and user index initialized\n");

  // Calculate the pearson coefficient for a single movie
  fprintf(stderr, "Now calculating correlations");
  // Divide the dataset
  int* thread_ends = new int[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_ends[i] = (i + 1) * (num_movies / num_threads);
  }
  std::thread threads[num_threads];
  // Create the threads
  for (int i = 0; i < num_threads; i++) {
    if (i == 0) {
      threads[i] = std::thread(&KNN::fit_part, this, 0, thread_ends[i], mu_train, um_train, true);
    }
    else {
      if (i == num_threads - 1) {
        threads[i] = std::thread(&KNN::fit_part, this, thread_ends[i - 1], num_movies, mu_train, um_train, false);
      }
      else {
        threads[i] = std::thread(&KNN::fit_part, this, thread_ends[i - 1], thread_ends[i], mu_train, um_train, false);
      }
    }
  }
  // Join the threads back together
  for (int i = 0; i < num_threads; i++) {
      threads[i].join();
  }
  delete thread_ends;
  fprintf(stderr, "\n");
  fprintf(stderr, "Correlations calculated. Model has been fitted\n");
}

/* Calculate the correlation based on the given struct (in this case a
 * pearson struct) */
float KNN::calculate_corr(struct pearson* p) {
  float numer = p->cnt * p->xy - (p->x * p->y);
  float denom_1 = pow((float) (p->cnt * p->xx - p->x * p->x), 0.5);
  float denom_2 = pow((float) (p->cnt * p->yy - p->y * p->y), 0.5);
  float pearson;
  if (denom_1 * denom_2 == 0) {
    pearson = 0;
  }
  else {
    pearson = (numer / (denom_1 * denom_2));
  }
  // Ignore values below a certain number
  if (pearson < min_pearson) {
    return 0;
  }
  // Penalize sparsity
  float sparse_pearson = pearson * (float) p->cnt / ((float) p->cnt + this->alpha);
  return sparse_pearson;
}

KNN::~KNN() {
  delete corr;
  delete sorted_corr;
  delete user_index;
}

// Initialize a pearson struct
void initialize_pearson(struct pearson* p) {
  p->x = 0;
  p->y = 0;
  p->xy = 0;
  p->xx = 0;
  p->yy = 0;
  p->cnt = 0;
}

// Update a pearson struct with two ratings
void update_pearson(struct pearson* p, int x_rating, int y_rating) {
  p->x += x_rating;
  p->y += y_rating;
  p->xy += x_rating * y_rating;
  p->xx += x_rating * x_rating;
  p->yy += y_rating * y_rating;
  p->cnt++;
}
