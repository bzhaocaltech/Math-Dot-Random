/* A simple model which finds the average rating for each movie and user. The
 * predictions of this model are simply the average of the average. */

#include "model.hpp"

class Mean_Model: public Model {
  private:
    /* Array that contains a list of movie means and user means.
     * movie_means[i] corresponds to the average rating of movie i */
    float* movie_means;
    float* user_means;
    /* The total number of movies and users in the dataset */
    int num_of_movies, num_of_users;
  public:
    /* Constructor for the mean model. Takes the number of movies and number
     * of users which is by default set to the defines in model.hpp */
    Mean_Model(int num_of_movies = NUM_MOVIES, int num_of_users = NUM_USERS);

    /* Given a list of x values in the form of (user, movie, time) predicts the
     * rating. Predicted rating of user i and movie j is
     * (user_means[i] + movie_means[i]) / 2 */
    vector<float>* predict(vector<int*>* x);

    /* Fits the model given a set of data in the form of (user, movie, time,
     * rating) by filling out movie_means and user_means */
    void fit(vector<int*>* x);

    /* Destructor for mean model */
    ~Mean_Model();
};
