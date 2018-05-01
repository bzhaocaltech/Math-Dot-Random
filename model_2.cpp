/* Runs a simple SVD */

#include "svd.hpp"
#include "serialize.hpp"
#include "output.hpp"

int main() {
  struct dataset* x_train = unserialize("data/mu_train.ser");
  struct dataset* x_valid = unserialize("data/mu_valid.ser");

  SVD* svd = new SVD(2, 0.005, 0.01);
  svd->fit(x_train, 2);
  float score = svd->score(x_valid);
  printf("Out of sample MSE is %f\n", score);
  score = svd->score(x_train);
  printf("In sample MSE is %f\n", score);

  delete x_train->data;
  delete x_train;
  delete x_valid->data;
  delete x_valid;

  struct dataset* x_test = unserialize("data/mu_qual.ser");
  std::vector<float>* predictions = svd->predict(x_test);

  output(*predictions, "results.dta");

  delete x_test->data;
  delete x_test;
  delete predictions;

  return 0;
}
