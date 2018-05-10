/* A simple model which finds the average rating for each movie and user. The
 * predictions of this model are simply the average of the average. */

#include "model.hpp"
#include <string>

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

    /* Constructs a mean model from a serialized file */
    Mean_Model(string file);

    /* Serializes the mean_model into a given file */
    void serialize(string file);

    /* Given dataset returns vector containing predicted rating.
     * Predicted rating of user i and movie j is
     * (user_means[i] + movie_means[i]) / 2 */
    vector<float>* predict(struct dataset* dataset);

    /* Fits the model given a dataset filling out movie_means and user_means */
    void fit(struct dataset* dataset);

    /* Destructor for mean model */
    ~Mean_Model();
};
