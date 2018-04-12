/* Run a mean model */

#include "mean_model.hpp"
#include "serialize.hpp"

int main() {
  vector<int*>* x_train = unserialize("data/mu_train.ser", 4);
  vector<int*>* x_valid = unserialize("data/mu_valid.ser", 4);

  Mean_Model* model = new Mean_Model();
  model->fit(x_train);
  float score = model->score(x_valid);

  printf("Out of sample MSE is %f\n", score);

  score = model->score(x_train);

  printf("In sample MSE is %f\n", score);

  return 0;
}
