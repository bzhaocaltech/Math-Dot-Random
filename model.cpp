#include "model.hpp"
#include <math.h>

using namespace std;

/* Returns the mean error for a set. Lower score is better. The elements
 * of the vector are in the form of (user, movie, time, rating) */
float Model::score(vector<int*>* x) {
  float error = 0;
  vector<float>* predictions = this->predict(x);
  for (unsigned int i = 0; i < predictions->size(); i++) {
    // Add the error of this single prediction to the total error
    error += this->error(predictions->at(i), x->at(i)[3]);
  }
  float mse = error / (float) predictions->size();
  return pow(mse, 0.5);
}

/* Uses an some measure to return the error incurred by a predicted rating.
 * Uses squared error by default unless overloaded by a child class. */
float Model::error(float predicted_rating, int actual_rating) {
  return pow((predicted_rating - (float) actual_rating), 2);
}
