/* Introduces a time factor to the typical KNN algorithm with pearson correlation */

#include "knn.hpp"

class TIME_KNN : public KNN {
  protected:
    /* Time factor. Represents how much we punish differences in time between
     * datapoints */
    float tau;
    // Two other random floats that need to be tuned
    float delta;
    float gamma;

    /* Predict a single datapoint */
    float predict_one(struct data data);

    void predict_part(int start, int end, struct dataset* dataset, float* predictions, bool track_progress);

    void fit_part(int start, int end, struct dataset* mu_train, struct dataset* um_train, bool track_progress = false);
  public:
    /* Constructor */
    TIME_KNN(int n_size, int alpha, float e, float min_pearson, float tau, float delta, float gamma,
             int num_threads = 8, int num_users = NUM_USERS, int num_movies = NUM_MOVIES);

    /* Returns e */
    float get_tau();

    /* Set e */
    void set_tau(float tau);

    /* Given a list of x values in the form of (user, movie, time) predicts
     * the rating */
    std::vector<float>* predict(struct dataset* dataset);

    void fit(struct dataset* um_train, struct dataset* mu_train);
};
