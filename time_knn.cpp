#include "time_knn.hpp"
#include <math.h>
#include <thread>

TIME_KNN::TIME_KNN(int n_size, int alpha, float e, float min_pearson, float tau, float delta, float gamma,
int num_threads, int num_users, int num_movies)
: KNN::KNN(n_size, alpha, e, min_pearson, num_threads, num_users, num_movies){
  this->tau = tau;
  this->delta = delta;
  this->gamma = gamma;
  fprintf(stderr, "Adding tau factor = %f for time \n", tau);
}

/* Returns e */
float TIME_KNN::get_tau() {
  return tau;
}

/* Set e */
void TIME_KNN::set_tau(float tau) {
  this->tau = tau;
  fprintf(stderr, "Setting tau to %f\n", tau);
}

/* Predict a single datapoint */
float TIME_KNN::predict_one(struct data data) {
  float* correlation_values = corr->row(data.movie);
  // Indices we need to search between to see if a user has rated a movie
  int user_start_index = user_index[data.user];
  int user_end_index;
  if (data.user != (unsigned int) num_users - 1) {
    user_end_index = user_index[data.user + 1];
  }
  else {
    user_end_index = training_set->size;
  }
  // Get all the other movies that the user has watched
  vector<int>* to_sort = new vector<int>();
  // Also save the ratings and grab adjusted correlation values
  int* movie_ratings = new int[num_movies];
  float* adjusted_corr = new float[num_movies];
  for (int i = user_start_index; i < user_end_index; i++) {
    struct data curr_point = training_set->data[i];
    int movie = curr_point.movie;
    movie_ratings[movie] = curr_point.rating;
    float deltat = fabs(data.date - curr_point.date);
    float time_factor = exp(-deltat / tau);
    float raw_corr = delta * correlation_values[movie] * time_factor + gamma;
    // Simgoid function
    adjusted_corr[movie] = (float) 1 / ((float) 1 + exp(-raw_corr));
    to_sort->push_back(movie);
  }
  // Sort the other movies that the user has watched
  vector<int>* sorted_correlations = quicksort_corr(to_sort, adjusted_corr);
  // Get as many neighbors as possible
  int num_neighbors;
  if (n_size < (int) sorted_correlations->size()) {
    num_neighbors = n_size;
  }
  else {
    num_neighbors = sorted_correlations->size();
  }
  // Get prediction
  float numer = 0;
  float denom = 0;
  for (int i = 0; i < num_neighbors; i++) {
    int neighbor = sorted_correlations->at(i);
    float z_rating = ((float) movie_ratings[neighbor] - movie_means[neighbor]) / movie_std[neighbor];
    numer += (float) z_rating * pow(adjusted_corr[neighbor], e);
    denom += pow(adjusted_corr[neighbor], e);
  }
  delete sorted_correlations;
  delete movie_ratings;
  delete adjusted_corr;
  if (denom == 0) {
    return 3;
  }
  float value = numer / denom;
  // Convert from z value back to regular rating
  float rating = (value * movie_std[data.movie]) + movie_means[data.movie];
  return rating;
}

/* Given a list of x values in the form of (user, movie, time) predicts
 * the rating */
std::vector<float>* TIME_KNN::predict(struct dataset* dataset) {
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
      threads[i] = std::thread(&TIME_KNN::predict_part, this, 0, thread_ends[i], dataset, arr_predictions, true);
    }
    else {
      if (i == num_threads - 1) {
        threads[i] = std::thread(&TIME_KNN::predict_part, this, thread_ends[i - 1], dataset->size, dataset, arr_predictions, false);
      }
      else {
        threads[i] = std::thread(&TIME_KNN::predict_part, this, thread_ends[i - 1], thread_ends[i], dataset, arr_predictions, false);
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

void TIME_KNN::predict_part(int start, int end, struct dataset* dataset, float* predictions, bool track_progress) {
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

/* Fit on part of the dataset */
void TIME_KNN::fit_part(int start, int end, struct dataset* mu_train, struct dataset* um_train, bool track_progress) {
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
        // Convert to z scores
        float curr_movie_z = ((float) movie_data.rating - movie_means[movie]) / movie_std[movie];
        float o_movie_z = ((float) u_data.rating - movie_means[u_data.movie]) / movie_std[u_data.movie];
        update_pearson(&intermediates[u_data.movie], curr_movie_z, o_movie_z);
      }
    }

    // Calculate correlations and update corr
    float* movie_correlations = new float[num_movies];
    for (int i = 0; i < num_movies; i++) {
      float p = this->calculate_pearson(&intermediates[i]);
      movie_correlations[i] = this->calculate_corr(&intermediates[i], p);
    }
    corr->update_row(movie, movie_correlations);
    delete intermediates;
  }
}

/* Given two datasets, one sorted by users and the other sorted by movies,
 * calculates the correlation between all movies */
void TIME_KNN::fit(struct dataset* um_train, struct dataset* mu_train) {
  fprintf(stderr, "Fitting the data of size %i\n", um_train->size);

  // Save um train
  this->training_set = um_train;

  // Initialize movie and user index
  fprintf(stderr, "Initializing movie and user index\n");
  unsigned int curr_index = 0;
  movie_index[curr_index] = 0;
  int curr_count = 0;
  for (int i = 0; i < mu_train->size; i++) {
    struct data data = mu_train->data[i];
    if (data.movie != curr_index) {
      movie_means[curr_index] /= (float) curr_count;
      curr_count = 0;
      curr_index++;
      movie_index[curr_index] = i;
    }
    curr_count++;
    movie_means[data.movie] += data.rating;
  }
  // For the last movie
  movie_means[curr_index] /= (float) curr_count;
  curr_index = 0;
  user_index[curr_index] = 0;
  for (int i = 0; i < um_train->size; i++) {
    struct data data = um_train->data[i];
    if (data.user != curr_index) {
      curr_index++;
      user_index[curr_index] = i;
    }
  }
  // Go back and compute movie standard deviation
  fprintf(stderr, "Calculating movie standard deviations\n");
  curr_index = 0;
  curr_count = 0;
  float inner_sum = 0;
  for (int i = 0; i < mu_train->size; i++) {
    struct data data = mu_train->data[i];
    if (curr_index != data.movie) {
      movie_std[curr_index] = pow((inner_sum / (float) curr_count), 0.5);
      curr_index++;
      curr_count = 0;
      inner_sum = 0;
    }
    inner_sum += pow(data.rating - movie_means[curr_index], 2);
    curr_count++;
  }
  // For the last movie
  movie_std[curr_index] = pow((inner_sum / (float) curr_count), 0.5);
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
      threads[i] = std::thread(&TIME_KNN::fit_part, this, 0, thread_ends[i], mu_train, um_train, true);
    }
    else {
      if (i == num_threads - 1) {
        threads[i] = std::thread(&TIME_KNN::fit_part, this, thread_ends[i - 1], num_movies, mu_train, um_train, false);
      }
      else {
        threads[i] = std::thread(&TIME_KNN::fit_part, this, thread_ends[i - 1], thread_ends[i], mu_train, um_train, false);
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
