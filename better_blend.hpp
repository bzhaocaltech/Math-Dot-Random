/* Another blending model */

#include "matrix.hpp"
#include "vector.hpp"
#include "model.hpp"

class better_blend {
  private:
    // Each row represents the movie's constants that we take for linear regression
    Matrix<float>* movie_constants;
    // Ditto for users
    Matrix<float>* user_constants;
    // User biases
    Vector* user_biases;
    // Movie biases
    Vector* movie_biases;
    // The thing we regularize against
    float* avg_constants;
    // The regularization strength for users and movies
    float user_reg;
    float movie_reg;
    // The learning rate
    float eta;
    // Num of stuff
    int num_movies;
    int num_users;
    int num_models;
    // The results from qual and probe of each model
    vector<float>** model_probe;
    vector<float>** model_qual;
  public:
    better_blend(float eta, float user_reg, float movie_reg, int num_models,
      float* avg_constants, vector<float>** model_probe, vector<float>** model_qual,
      int num_movies = NUM_MOVIES, int num_users = NUM_USERS);

    // Predicts qual
    std::vector<float>* predict(struct dataset* qual);

    // Fits agaisnt the actually results from probe
    void fit(struct dataset* probe, int epochs);

    // Fit against single point
    void grad_one(struct data data, float* model_ratings);
};
