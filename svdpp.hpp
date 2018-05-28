/* SVD Plus Plus */

#include "svd.hpp"

class SVDPP : public SVD {
  protected:
    /* N[u] contains all the movies rated by one user */
    std::vector<int>** N;

    /* Matrix of implicit feedbacks. y->row(i) represents the implicit feedback
     * vector associated with movie j */
    Matrix* y;

    /* Returns the error when trying to predict a data point */
    float get_err(struct data d);

    /* Predicts a single point */
    float predict_one(struct data d);

    /* Returns the implicit feedback factor |N(u)|^-0.5 * sum_{j \in N(u)} y_j for
     * a given user */
    float* get_f_factor(int user);

    /* Run a gradient on part of the dataset */
    void grad_part(struct dataset* ds, bool track_progress);

  public:
    /* Constructor for SVDPP */
    SVDPP(int latent_factors, float eta, float reg, int num_users = NUM_USERS,
          int num_movies = NUM_MOVIES);

    /* Given a list of x values in the form of (user, movie, time) predicts
     * the rating */
    std::vector<float>* predict(struct dataset* dataset);

    /* Given a list of x values in the form of (user, movie, time, rating)
     * fits the model */
    void fit(struct dataset* dataset, int epochs, float early_stopping, struct dataset* validation_set = NULL, int num_threads = 16);

    /* Destructor for SVD */
    ~SVDPP();
};
