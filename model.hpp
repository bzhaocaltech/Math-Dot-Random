/* The model class from which all other models inherit from */
#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>
#include "data.hpp"

using namespace std;

#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define NUM_DATES 2243

class Model {
  public:
    /* Given dataset returns vector containing predicted rating */
    virtual vector<float>* predict(struct dataset* x) = 0;

    /* Serializes the model into a given file */
    virtual void serialize(string file) = 0;

    /* Uses an some measure to return the error incurred by a predicted rating.
     * Uses squared error by default unless overloaded by a child class. */
    float error(float predicted_rating, int actual_rating);

    /* Returns the mean error for a set. Lower score is better. */
    float score(struct dataset* dataset);
};

/* Function for splits the dataset into equal sized pieces */
struct dataset** split_dataset(struct dataset* dataset, int num_splits);

#endif
