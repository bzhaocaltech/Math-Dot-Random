/* Runs a simple SVD */
/* Results stored to models/svd_[#latent_factors]_[eta]_[reg]_[#epochs] */

#include "svd.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("USAGE: ./run_svd #latent_factors eta regularization #epochs early_stopping\n");
    exit(1);
  }
  // Get command line arguments
  int latent_factors = atoi(argv[1]);
  float eta = atof(argv[2]);
  float reg = atof(argv[3]);
  int epochs = atoi(argv[4]);
  float early_stopping = atof(argv[5]);
  string file_name = string("models/svd_") + argv[1] + "_" + argv[2] + "_" + argv[3]
  + "_" + argv[4] + "_";

  struct dataset* x_train = unserialize("data/um_train.ser");
  struct dataset* x_valid = unserialize("data/um_valid.ser");
  struct dataset* x_probe = unserialize("data/um_probe.ser");

  SVD* svd = new SVD(latent_factors, eta, reg);
  svd->fit(x_train, epochs, early_stopping, x_probe);
  float score = svd->score(x_train);
  printf("In sample RMSE is %f\n", score);
  score = svd->score(x_valid);
  printf("Out of sample RMSE is %f\n", score);
  score = svd->score(x_probe);
  printf("Probe RMSE is %f\n", score);

  delete x_train->data;
  delete x_train;
  delete x_valid->data;
  delete x_valid;

  struct dataset* x_test = unserialize("data/um_qual.ser");
  std::vector<float>* predictions = svd->predict(x_test);
  std::vector<float>* blend = svd->predict(x_probe);

  output(*predictions, file_name + "results.dta");
  output(*blend, file_name + "blend.dta");

  delete x_test->data;
  delete x_test;
  delete predictions;
  delete blend;

  // svd->serialize(file_name);

  return 0;
}
