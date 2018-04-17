/* A basic matrix factorization model */

#include "model.hpp"

class SVD : public Model {
  private:
    /* The two matrices of latent factors
     * U is a num_users * latent_factors
     * V is a num_movies * latent_factors */
    int** U;
    int** V;
    // Num of latent factors
    int latent_factors;
    // Num of users and movies
    int num_users;
    int num_movies;
    // Learning rate
    int eta;

    float regularized_error(vector<int*>* data);
  public:
    /* Constructor for SVD */
    SVD(int num_of_movies = NUM_MOVIES, int num_of_users = NUM_USERS,
        int latent_factors, int eta);

    /* Given a list of x values in the form of (user, movie, time) predicts the
     * rating */
    vector<float>* predict(vector<int*>* x);

    /* Given a list of x values in the form of (user, movie, time, rating) fits
     * the model */
    void fit(vector<int*>* x);

    /* Destructor for SVD */
    ~SVD();
}
