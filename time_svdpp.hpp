/* Time SVD Plus Plus */

#include "svdpp.hpp"
#include <map>

class TIME_SVDPP : public SVDPP {
  protected:
    /* The number of bins we seperate the dates into for the movie */
    int num_bins;

    /* Maps dates to bins */
    int* date_to_bin;

    /* Values of each bin */
    Matrix<float>* bin_values;

    /* Variable to calculate dev_u(t) */
    float beta;

    /* Mean user time */
    float* mean_user_time;

    /* What we multiply be dev_u(t) to get the user bias */
    Vector* alpha_u;

    /* Day by day user bias */
    Vector** a_date;

    /* What we multiply by dev_u(t) to get the user factor bias */
    Matrix<float>* alpha_u_k;

    /* Day by day user factor bias */
    Matrix<float>** U_date;

    /* Helps to map date with entry in a_time. I.e. U_index_to_time[user][date]
     * returns the index i so that the bias for the date and user is just
     * that U_time[user]->row(i) is calculating the bias for */
    std::map<int, int>** date_to_index;

    /* Predicts a single point */
    float predict_one(struct data d);

    /* Returns dev_u(t) */
    float get_devt(int user, int date);

    void grad_part(struct dataset* ds, bool track_progress);
  public:
    /* Constructor for SVDPP */
    TIME_SVDPP(int latent_factors, float eta, float reg, float beta = 0.4,
               int num_bins = 30, int num_dates = NUM_DATES,
               int num_users = NUM_USERS, int num_movies = NUM_MOVIES);

    /* Fit the model or something I dunno */
    void fit(struct dataset* dataset, int epochs, struct dataset* validation_set = NULL, int num_threads = 16);

    /* Destructor for TIME_SVDPP */
    ~TIME_SVDPP();
};
