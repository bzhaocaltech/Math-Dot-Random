#include "model.hpp"
#include <math.h>
#include <stdlib.h>

using namespace std;

/* Returns the mean error for a set. Lower score is better. The elements
 * of the vector are in the form of (user, movie, time, rating) */
 float Model::score(struct dataset* dataset) {
  float error = 0;
  vector<float>* predictions = this->predict(dataset);
  for (unsigned int i = 0; i < predictions->size(); i++) {
    // Add the error of this single prediction to the total error
    error += this->error(predictions->at(i), dataset->data[i].rating);
  }
  float mse = error / (float) predictions->size();
  free(predictions);
  return pow(mse, 0.5);
}

/* Uses an some measure to return the error incurred by a predicted rating.
 * Uses squared error by default unless overloaded by a child class. */
float Model::error(float predicted_rating, int actual_rating) {
  return pow((predicted_rating - (float) actual_rating), 2);
}

struct dataset** split_dataset(struct dataset* dataset, int num_splits) {
  int curr_index = 0;
  struct dataset** to_return = new struct dataset*[num_splits];
  for (int i = 0; i < num_splits; i++) {
    int new_index = dataset->size * ((float) (i + 1) / (float) num_splits);
    if (i == num_splits - 1) {
      new_index = dataset->size;
    }
    struct dataset* split_dataset = new struct dataset();
    split_dataset->size = new_index - curr_index;
    split_dataset->data = &(dataset->data[curr_index]);
    to_return[i] = split_dataset;
    curr_index = new_index;
  }
  return to_return;
}
