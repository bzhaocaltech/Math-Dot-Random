/* Runs a SVDPP */
/* TODO: Serialize model */

#include "svdpp.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("USAGE: ./run_svdpp #latent_factors eta regularization #epochs\n");
    exit(1);
  }
  // Get command line arguments
  int latent_factors = atoi(argv[1]);
  float eta = atof(argv[2]);
  float reg = atof(argv[3]);
  int epochs = atoi(argv[4]);
  // string file_name = string("models/svd_") + argv[1] + "_" + argv[2] + "_" + argv[3]
  // + "_" + argv[4] + ".ser";

  struct dataset* x_train = unserialize("data/um_train.ser");
  struct dataset* x_valid = unserialize("data/um_valid.ser");

  SVDPP* svdpp = new SVDPP(latent_factors, eta, reg);
  svdpp->fit(x_train, epochs);
  float score = svdpp->score(x_valid);
  printf("Out of sample MSE is %f\n", score);
  score = svdpp->score(x_train);
  printf("In sample MSE is %f\n", score);

  delete x_train->data;
  delete x_train;
  delete x_valid->data;
  delete x_valid;

  struct dataset* x_test = unserialize("data/um_qual.ser");
  std::vector<float>* predictions = svdpp->predict(x_test);

  output(*predictions, "results.dta");

  delete x_test->data;
  delete x_test;
  delete predictions;

  return 0;
}
