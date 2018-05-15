/* Runs a simple SVD */
/* Results stored to models/svd_[#latent_factors]_[eta]_[reg]_[#epochs] */

#include "svd.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("USAGE: ./run_svd #latent_factors eta regularization #epochs\n");
    exit(1);
  }
  // Get command line arguments
  int latent_factors = atoi(argv[1]);
  float eta = atof(argv[2]);
  float reg = atof(argv[3]);
  int epochs = atoi(argv[4]);
  string file_name = string("models/svd_") + argv[1] + "_" + argv[2] + "_" + argv[3]
  + "_" + argv[4] + ".ser";

  struct dataset* x_train = unserialize("data/mu_train.ser");
  struct dataset* x_valid = unserialize("data/mu_valid.ser");

  SVD* svd = new SVD(latent_factors, eta, reg);
  svd->fit(x_train, epochs);
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

  svd->serialize(file_name);

  return 0;
}