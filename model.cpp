#include "model.hpp"

using namespace std;

/* Returns the combined error for a set. Lower score is better. The elements
 * of the vector are in the form of (user, movie, time, rating) */
float Model::score(vector<int*>* x) {
  float error = 0;
  vector<float>* predictions = this->predict(x);
  for (unsigned int i = 0; i < predictions->size(); i++) {
    // Add the error of this single prediction to the total error
    error += this->error(predictions->at(i), x->at(i)[3]);
  }
  return error;
}
