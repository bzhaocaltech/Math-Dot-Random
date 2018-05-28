#include "time_svdpp.hpp"
#include <cmath>
#include <thread>

TIME_SVDPP::TIME_SVDPP(int latent_factors, float eta, float reg, float beta,
                       int num_bins, int num_dates,
                       int num_users, int num_movies)
: SVDPP::SVDPP(latent_factors, eta, reg, num_users, num_movies) {
  fprintf(stderr, "Adding TIME_SVDPP features\n");
  this->beta = beta;
  // Set up the date bins for all the movies
  this->num_bins = num_bins;
  this->date_to_bin = new int[num_dates];
  int bin_break = num_dates / num_bins;
  for (int i = 0; i < num_dates; i++) {
    this->date_to_bin[i] = floor(i / bin_break);
  }
  this->bin_values = new Matrix<float>(num_movies, num_bins);
  // Set up the alpha values
  this->alpha_u = new Vector(num_users);
  this->alpha_u_k = new Matrix<float>(num_users, latent_factors);
  // Set up the day by day biases for users and their factors
  this->a_date = new Vector*[num_users];
  this->U_date = new Matrix<float>*[num_users];
  this->mean_user_time = new float[num_users];
  // Set up the index to date to index maps
  this->date_to_index = new map<int, int>*[num_users];
  for (int i = 0; i < num_users; i++) {
    this->date_to_index[i] = new map<int, int>();
  }
}

/* Predicts a single point */
float TIME_SVDPP::predict_one(struct data d) {
  // Get adjusted movie bias
  float adjusted_b = bin_values->get_val(d.movie, date_to_bin[d.date]) + b->at(d.movie);
  // Get adjusted user bias
  float devt = get_devt(d.user, d.date);
  float date_dev_term = alpha_u->at(d.user) * devt;
  // Get the day by day term only if we've seen that date before
  float day_by_day_term = 0;
  if (date_to_index[d.user]->find(d.date) != date_to_index[d.user]->end()) {
    day_by_day_term = a_date[d.user]->at(date_to_index[d.user]->at(d.date));
  }
  float adjusted_a = a->at(d.user) + date_dev_term + day_by_day_term;
  // Get adjusted user factors
  float* f_factor = get_f_factor(d.user);
  float* Ui = U->row(d.user);
  float* alpha_vector = alpha_u_k->row(d.user);
  // Get the day by day term only if we've seen that date before
  float* factor_day_by_day_term = new float[latent_factors];
  for (int i = 0; i < latent_factors; i++) {
    factor_day_by_day_term[i] = 0;
  }
  if (date_to_index[d.user]->find(d.date) != date_to_index[d.user]->end()) {
    delete factor_day_by_day_term;
    factor_day_by_day_term = U_date[d.user]->row(date_to_index[d.user]->at(d.date));
  }
  float* adjusted_Ui = new float[latent_factors];
  for (int i = 0; i < latent_factors; i++) {
    adjusted_Ui[i] = f_factor[i] + Ui[i] + alpha_vector[i] * devt +
                     factor_day_by_day_term[i];
  }
  delete f_factor;
  // Take the inner product of adjusted_u with v
  float* Vj = V->row(d.movie);
  float p = 0;
  for (int i = 0; i < latent_factors; i++) {
    p += adjusted_Ui[i] * Vj[i];
  }
  p += mu + adjusted_b + adjusted_a;
  return p;
}

/* Returns dev_u(t) */
float TIME_SVDPP::get_devt(int user, int date) {
  float deltat = (float) date - mean_user_time[user];
  float value;
  if (deltat > 0) {
    value = pow(fabs(deltat), beta);
  }
  else if (deltat < 0) {
    value =  (float) -1 * pow(fabs(deltat), beta);
  }
  else {
    return 0;
  }
  return value / (float) 3;
}

void TIME_SVDPP::grad_part(struct dataset* ds, bool track_progress) {
  int dot_break = ds->size / 30;
  for (int n = 0; n < ds->size; n++) {
    // Track progress
    if ((n % dot_break == 0) && track_progress) {
      fprintf(stderr, ".");
    }

    // Get stuff from current data point
    struct data data = ds->data[n];
    int movie = data.movie;
    int user = data.user;
    int date = data.date;
    int rating = data.rating;

    // Grab current factors
    int date_index = date_to_index[user]->at(date);
    float movie_bias = b->at(movie);
    float movie_bin_bias = bin_values->get_val(movie, date_to_bin[date]);
    float devt = get_devt(user, date);
    float user_bias = a->at(user);
    float a_alpha = alpha_u->at(user);
    float day_by_day_term = a_date[user]->at(date_index);
    float* f_factor = get_f_factor(user);
    float* Ui = U->row(user);
    float* Vj = V->row(movie);
    float* factor_day_by_day_term = U_date[user]->row(date_index);
    float* alpha_vector = alpha_u_k->row(user);

    // Calculate current prediction
    float adjusted_b = movie_bin_bias + movie_bias;
    float adjusted_a = user_bias + a_alpha * devt + day_by_day_term;
    float prediction = 0;
    for (int i = 0; i < latent_factors; i++) {
      float adjusted_Ui = Ui[i] + factor_day_by_day_term[i] + alpha_vector[i] * devt + f_factor[i];
      prediction += Vj[i] * adjusted_Ui;
    }
    prediction += mu + adjusted_b + adjusted_a;
    // Use prediction to calculate the error
    float error = (float) rating - prediction;

    // Find the gradient with respect to the movie bias
    float grad_movie_bias = reg * movie_bias - error;
    float new_movie_bias = movie_bias - eta * grad_movie_bias;

    // Find the gradient with respect to the movie bin bias
    float grad_movie_bin_bias = reg * movie_bin_bias - error;
    float new_movie_bin_bias = movie_bin_bias - eta * grad_movie_bin_bias;

    // Find the gradient with respect to the user bias
    float grad_user_bias = reg * user_bias - error;
    float new_user_bias = user_bias - eta * grad_user_bias;

    // Find the gradient with respect to a_alpha
    float grad_a_alpha = reg * a_alpha - devt * error;
    float new_alpha_a = a_alpha - eta * grad_a_alpha;

    // Find the gradent with respect ot the day by day term
    float grad_day_by_day_term = reg * day_by_day_term - error;
    float new_day_by_day_term = day_by_day_term - eta * grad_day_by_day_term;

    // For loop for all the vector stuff
    float* new_Ui = new float[latent_factors];
    float* new_factor_day_by_day_term = new float[latent_factors];
    float* new_alpha_vector = new float[latent_factors];
    float* new_Vj = new float[latent_factors];
    for (int i = 0; i < latent_factors; i++) {
      // Find the gradient with respect to Ui
      float grad_Ui = reg * Ui[i] - Vj[i] * error;
      new_Ui[i] = Ui[i] - eta * grad_Ui;
      // Find the gradient with resepct to factor_day_by_day_term
      float grad_factor_day_by_day_term = reg * factor_day_by_day_term[i] - Vj[i] * error;
      new_factor_day_by_day_term[i] = factor_day_by_day_term[i] - eta * grad_factor_day_by_day_term;
      // Find the gradient with respect to alpha vector
      float grad_alpha_vector = reg * alpha_vector[i] - Vj[i] * devt * error;
      new_alpha_vector[i] = alpha_vector[i] - eta * grad_alpha_vector;
      // Find the gradient with respect to Vj
      float adjusted_Ui = Ui[i] + factor_day_by_day_term[i] + devt * alpha_vector[i] + f_factor[i];
      float grad_Vj = reg * Vj[i] - adjusted_Ui * error;
      new_Vj[i] = Vj[i] - eta * grad_Vj;
    }

    // Find the gradient with respect to all the y's
    float error_const = pow((float) this->N[user]->size(), -0.5) * error;
    for (unsigned int u = 0; u < this->N[user]->size(); u++) {
      int movie_watched = this->N[user]->at(u);
      float* yj = this->y->row(movie_watched);
      float* new_y = new float[this->latent_factors];
      for (int i = 0; i < this->latent_factors; i++) {
        float grad_y = this->reg * yj[i] - error_const * Vj[i];
        new_y[i] = yj[i] - eta * grad_y;
      }
      this->y->update_row(movie_watched, new_y);
    }

    // Update everything
    b->update_element(movie, new_movie_bias);
    bin_values->set_val(movie, date_to_bin[date], new_movie_bin_bias);
    a->update_element(user, new_user_bias);
    alpha_u->update_element(user, new_alpha_a);
    a_date[user]->update_element(date_index, new_day_by_day_term);
    U_date[user]->update_row(date_index, new_factor_day_by_day_term);
    alpha_u_k->update_row(user, new_alpha_vector);
    U->update_row(user, new_Ui);
    V->update_row(movie, new_Vj);

    delete f_factor;
  }
}

void TIME_SVDPP::fit(struct dataset* dataset, int epochs,
struct dataset* validation_set, int num_threads) {
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
    float random = 0;
    this->b->update_element(j, random);
  }
  // Initialize SVDPP factors randomly
  for (int i = 0; i < this->num_movies; i++) {
    for (int j = 0; j < this->latent_factors; j++) {
      float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
      this->y->set_val(i, j, random);
    }
  }
  // Initialize TIME_SVDPP factors randomly
  for (int i = 0; i < this->num_movies; i++) {
    for (int j = 0; j < this->latent_factors; j++) {
      float random = 0;
      this->bin_values->set_val(i, j, random);
    }
  }
  for (int i = 0; i < this->num_users; i++) {
    for (int j = 0; j < this->latent_factors; j++) {
      float random = (((float) rand()) / (float) RAND_MAX) / (float) 3;
      this->alpha_u_k->set_val(i, j, random);
    }
  }
  for (int i = 0; i < this->num_users; i++) {
    float random = (((float) rand()) / (float) RAND_MAX) / (float) 3;
    this->alpha_u->update_element(i, random);
  }
  fprintf(stderr, "Matrices randomly initialized\n");

  fprintf(stderr, "Initializing date variables\n");
  // Initializiing date variables
  fprintf(stderr, "Calculating date means\n");
  unsigned int curr_index = 0;
  int curr_count = 0;
  mean_user_time[0] = 0;
  for (int i = 0; i < dataset->size; i++) {
    struct data data = dataset->data[i];
    if (data.user != curr_index) {
      mean_user_time[curr_index] /= (float) curr_count;
      curr_index++;
      mean_user_time[curr_index] = 0;
      curr_count = 0;
    }
    mean_user_time[curr_index] += data.date;
    curr_count++;
  }
  mean_user_time[curr_index] /= curr_count;
  // Find unique dates
  fprintf(stderr, "Finding unique dates for each user\n");
  curr_index = 0;
  for (int i = 0; i < dataset->size; i++) {
    struct data data = dataset->data[i];
    int user = data.user;
    int date = data.date;
    // Check to see if date already exists in map. If not, insert it
    if (date_to_index[user]->find(date) == date_to_index[user]->end()) {
      date_to_index[user]->insert(pair<int, int>(date, date_to_index[user]->size()));
    }
  }
  // Now use the unique dates to initialize a_date and U_date
  for (int i = 0; i < num_users; i++) {
    a_date[i] = new Vector(date_to_index[i]->size());
    for (unsigned int j = 0; j < date_to_index[i]->size(); j++) {
      float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
      a_date[i]->update_element(j, random);
    }
    U_date[i] = new Matrix<float>(date_to_index[i]->size(), latent_factors);
    for (unsigned int j = 0; j < date_to_index[i]->size(); j++) {
      for (int k = 0; k < latent_factors; k++) {
        float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
        U_date[i]->set_val(j, k, random);
      }
    }
  }
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
        threads[i] = std::thread(&TIME_SVDPP::grad_part, this, threaded_dataset[i], true);
      }
      else {
        threads[i] = std::thread(&TIME_SVDPP::grad_part, this, threaded_dataset[i], false);
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
        else if (prev_RMSE < score + 0.0003) {
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

/* Destructor for TIME_SVDPP */
TIME_SVDPP::~TIME_SVDPP() {
  delete date_to_bin;
  delete bin_values;
  delete alpha_u;
  delete alpha_u_k;
  for (int i = 0; i < num_users; i++) {
    delete a_date[i];
  }
  delete a_date;
  for (int i = 0; i < num_users; i++) {
    delete U_date[i];
  }
  delete U_date;
  delete mean_user_time;
  for (int i = 0; i < num_users; i++) {
    delete date_to_index[i];
  }
  delete date_to_index;
}
