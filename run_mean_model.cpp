/* Run a mean model */

#include "mean_model.hpp"
#include "serialize.hpp"
#include "output.hpp"

int main() {
  struct dataset* x_train = unserialize("data/mu_train.ser");
  struct dataset* x_valid = unserialize("data/mu_valid.ser");

  Mean_Model* model = new Mean_Model();
  model->fit(x_train);
  float score = model->score(x_valid);

  printf("Out of sample MSE is %f\n", score);

  score = model->score(x_train);

  printf("In sample MSE is %f\n", score);

  delete x_train->data;
  delete x_train;
  delete x_valid->data;
  delete x_valid;

  struct dataset* x_test = unserialize("data/mu_qual.ser");
  vector<float>* predictions = model->predict(x_test);

  output(*predictions, "results.dta");

  delete x_test->data;
  delete x_test;
  delete predictions;

  // Serialize the mean model
  model->serialize("models/mean_model.ser");

  return 0;
}
