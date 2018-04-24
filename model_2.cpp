/* Runs a simple SVD */

#include "svd.hpp"
#include "serialize.hpp"
#include "output.hpp"

int main() {
  std::vector<int*>* x_train = unserialize("data/mu_train.ser", 4);
  std::vector<int*>* x_valid = unserialize("data/mu_valid.ser", 4);

  SVD* svd = new SVD(5, 0.03, 0.01);
  svd->fit(x_train, 3);
}
