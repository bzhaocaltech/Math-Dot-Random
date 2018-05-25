/* KNN algorithm using Pearson correlation coefficient.
 * Movie similarities are calculated, user similarities are ignored */

#include "model.hpp"
#include "matrix.hpp"

class KNN : public Model {
  protected:
    /* The size of neighborhoods we are considering */
    int n_size;
    /* Roughly correlates to a regularization term. How much we consider sparsity */
    float alpha;
    /* The exponent that we take in weighting values */
    float e;

    /* Num of users and movies */
    int num_users;
    int num_movies;
    int num_threads;

    /* 2D array that contains all correlations. corr[i][j] holds similarity
     * between movies i and j */
    Matrix<float>* corr;
    /* The values of corr sorted such that corr[i][j] represents the movie with
     * the jth highest correlation with movie i */
    // NOTE: We ignore correlations below 0.5
    vector<int>** sorted_corr;
    /* We need to keep a copy of the um_training data */
    struct dataset* training_set;

    /* user_index[j] represents the first index where user i appears in um_train
     * movie_index[i] represents the first index where movie i appears in mu_train */
    int* user_index;
    int* movie_index;

    /* Quicksort indices based on corr to create sorted_corr */
    vector<int>* quicksort_corr(vector<int>* vec, float* values);

    /* Predicts the rating of a single datapoint */
    float predict_one(struct data data);

    /* Predicts the rating of part of the dataset. Meant to be used to enable multithreading */
    void predict_part(int start, int end, struct dataset* dataset, float* predictions, bool track_progress = false);

    /* Fit on part of the dataset */
    void fit_part(int start, int end, struct dataset* mu_train, struct dataset* um_train, bool track_progress = false);

    /* Calculate the correlation based on the given struct (in this case a
     * pearson struct) */
    float calculate_corr(struct pearson* p);
  public:
    /* Constructor for KNN
     * n_size is the size of the neighborhood. alpha is a term representing how
     * much we punish sparsity */
    KNN(int n_size, float alpha, float e, int num_threads = 8, int num_users = NUM_USERS, int num_movies = NUM_MOVIES);

    /* Returns the neighborhood size */
    int get_n_size();

    /* Set the neighborhood size */
    void set_n_size(int n_size);

    /* Returns e */
    float get_e();

    /* Set e */
    void set_e(float e);

    /* Predicts the ratings for a given dataset */
    std::vector<float>* predict(struct dataset* dataset);

    /* Given two datasets, one sorted by users and the other sorted by movies,
     * calculates the correlation between all movies.
     * NOTE: um_train must be sorted by users and mu_train must be sorted by
     * movie */
    void fit(struct dataset* um_train, struct dataset* mu_train);

    /* Destructor for SVD */
    ~KNN();
};

// Contains intermediate values to calculate Pearson correlation
struct pearson {
  float x; // sum of ratings for movie_x
  float y; // sum of ratings for movie y
  float xy; // sum of product of ratings for movies X and Y
  float xx; // sum of square of ratings for movie X
  float yy; // sum of square of ratings for movie y
  unsigned int cnt; // number of viewers who rated both movies
};

// Initialize a pearson struct
void initialize_pearson(struct pearson* p);

// Update a pearson struct with two ratings
void update_pearson(struct pearson* p, int x_rating, int y_rating);

// Calculate the pearson correlation coefficient given the pearson struct
float calculate_pearson(struct pearson* p);
