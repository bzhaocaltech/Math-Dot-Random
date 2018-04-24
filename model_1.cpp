/* Run a mean model */

#include "mean_model.hpp"
#include "serialize.hpp"
#include "output.hpp"

int main() {
  vector<int*>* x_train = unserialize("data/mu_train.ser", 4);
  vector<int*>* x_valid = unserialize("data/mu_valid.ser", 4);

  Mean_Model* model = new Mean_Model();
  model->fit(x_train);
  float score = model->score(x_valid);

  printf("Out of sample MSE is %f\n", score);

  score = model->score(x_train);

  printf("In sample MSE is %f\n", score);

  free(x_train);
  free(x_valid);

  vector<int*>* x_test = unserialize("data/mu_qual.ser", 3);
  vector<float>* predictions = model->predict(x_test);

  output(*predictions, "results.dta");

  free(x_test);
  free(predictions);

  return 0;
}
