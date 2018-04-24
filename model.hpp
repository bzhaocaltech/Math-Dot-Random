/* The model class from which all other models inherit from */
#include <vector>

using namespace std;

#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define NUM_DATES 2243

class Model {
  public:
    /* Given a list of x values in the form of (user, movie, time) predicts the
     * rating */
    virtual vector<float>* predict(vector<int*>* x) = 0;

    /* Uses an some measure to return the error incurred by a predicted rating.
     * Uses squared error by default unless overloaded by a child class. */
    float error(float predicted_rating, int actual_rating);

    /* Returns the mean error for a set. Lower score is better. The elements
     * of the vector are in the form of (user, movie, time, rating) */
    float score(vector<int*>* x);

    /* Fits the model given a set of data in the form of (user, movie, time,
     * rating) */
    // virtual void fit(vector<int*>* x) = 0;
};
