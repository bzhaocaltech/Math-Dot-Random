/* This file loads the training and test sets */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "serialize.hpp"
using namespace std;

struct data parse_line(string line);

int main() {
  fstream mu_dta;
  fstream mu_idx;
  fstream um_dta;
  fstream um_idx;

  // Open all files
  mu_dta.open("mu/all.dta", ios::in | ios::binary);
  mu_idx.open("mu/all.idx", ios::in | ios::binary);
  um_dta.open("um/all.dta", ios::in | ios::binary);
  um_idx.open("um/all.idx", ios::in | ios::binary);
  assert(mu_dta.is_open() && mu_idx.is_open() && um_dta.is_open()
    && um_idx.is_open());

  // Used for parsing
  string line;
  string index_string;

  // Get the mu training data
  // Initialize datasets
  struct dataset* mu_train = new struct dataset();
  struct dataset* mu_valid = new struct dataset();
  struct dataset* mu_hidden = new struct dataset();
  struct dataset* mu_probe = new struct dataset();
  struct dataset* mu_qual = new struct dataset();
  mu_train->size = SIZE_TRAIN;
  mu_train->data = new data[SIZE_TRAIN];
  mu_valid->size = SIZE_VALID;
  mu_valid->data = new data[SIZE_VALID];
  mu_hidden->size = SIZE_HIDDEN;
  mu_hidden->data = new data[SIZE_HIDDEN];
  mu_probe->size = SIZE_PROBE;
  mu_probe->data = new data[SIZE_PROBE];
  mu_qual->size = SIZE_QUAL;
  mu_qual->data = new data[SIZE_QUAL];

  fprintf(stderr, "Loading mu files...");
  int num_lines = 0;
  int train_counter = 0;
  int valid_counter = 0;
  int hidden_counter = 0;
  int probe_counter = 0;
  int qual_counter = 0;
  while (getline(mu_dta, line) && getline(mu_idx, index_string)) {
    int index = atoi(index_string.c_str());
    struct data parsed_line = parse_line(line);
    // Part of base
    if (index == 1) {
      mu_train->data[train_counter] = parsed_line;
      train_counter++;
    }
    // Part of the validation set
    if (index == 2) {
      mu_valid->data[valid_counter] = parsed_line;
      valid_counter++;
    }
    // Part of the hidden set
    if (index == 3) {
      mu_hidden->data[hidden_counter] = parsed_line;
      hidden_counter++;
    }
    // Part of probe
    if (index == 4) {
      mu_probe->data[probe_counter] = parsed_line;
      probe_counter++;
    }
    // Part of qual
    if (index == 5) {
      mu_qual->data[qual_counter] = parsed_line;
      qual_counter++;
    }
    num_lines++;
    if (num_lines % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");

  // Serialize the mu files
  fprintf(stderr, "Serializing mu files...");
  serialize(mu_train, "data/mu_train.ser");
  serialize(mu_valid, "data/mu_valid.ser");
  serialize(mu_hidden, "data/mu_hidden.ser");
  serialize(mu_probe, "data/mu_probe.ser");
  serialize(mu_qual, "data/mu_qual.ser");
  fprintf(stderr, "\n");

  struct dataset* dataset = unserialize("data/mu_train.ser");
  for (int i = 0; i < 10; i++) {
    fprintf(stderr, "User: %i, Movie: %i, Date: %i, Rating: %i \n",
    dataset->data[i].user, dataset->data[i].movie, dataset->data[i].date,
    dataset->data[i].rating);
  }

  // Get the um training data
  // Initialize datasets
  struct dataset* um_train = new struct dataset();
  struct dataset* um_valid = new struct dataset();
  struct dataset* um_hidden = new struct dataset();
  struct dataset* um_probe = new struct dataset();
  struct dataset* um_qual = new struct dataset();
  um_train->size = SIZE_TRAIN;
  um_train->data = new data[SIZE_TRAIN];
  um_valid->size = SIZE_VALID;
  um_valid->data = new data[SIZE_VALID];
  um_hidden->size = SIZE_HIDDEN;
  um_hidden->data = new data[SIZE_HIDDEN];
  um_probe->size = SIZE_PROBE;
  um_probe->data = new data[SIZE_PROBE];
  um_qual->size = SIZE_QUAL;
  um_qual->data = new data[SIZE_QUAL];

  fprintf(stderr, "Loading um files...");
  num_lines = 0;
  train_counter = 0;
  valid_counter = 0;
  hidden_counter = 0;
  probe_counter = 0;
  qual_counter = 0;
  while (getline(um_dta, line) && getline(um_idx, index_string)) {
    int index = atoi(index_string.c_str());
    struct data parsed_line = parse_line(line);
    // Part of base
    if (index == 1) {
      um_train->data[train_counter] = parsed_line;
      train_counter++;
    }
    // Part of the validation set
    if (index == 2) {
      um_valid->data[valid_counter] = parsed_line;
      valid_counter++;
    }
    // Part of the hidden set
    if (index == 3) {
      um_hidden->data[hidden_counter] = parsed_line;
      hidden_counter++;
    }
    // Part of probe
    if (index == 4) {
      um_probe->data[probe_counter] = parsed_line;
      probe_counter++;
    }
    // Part of qual
    if (index == 5) {
      um_qual->data[qual_counter] = parsed_line;
      qual_counter++;
    }
    num_lines++;
    if (num_lines % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");

  // Serialize the um files
  fprintf(stderr, "Serializing um files...");
  serialize(um_train, "data/um_train.ser");
  serialize(um_valid, "data/um_valid.ser");
  serialize(um_hidden, "data/um_hidden.ser");
  serialize(um_probe, "data/um_probe.ser");
  serialize(um_qual, "data/um_qual.ser");
  fprintf(stderr, "\n");

  // Close files
  mu_dta.close();
  mu_idx.close();
  um_dta.close();
  um_idx.close();

  printf("Loaded data \n");
  return 0;
}

/* Parses a line in the .dta files into a integer array */
struct data parse_line(string line) {
  struct data data;
  data.user = atoi(line.substr(0, line.find(' ')).c_str()) - 1;
  line = line.substr(line.find(' ') + 1);

  data.movie = atoi(line.substr(0, line.find(' ')).c_str()) - 1;
  line = line.substr(line.find(' ') + 1);

  data.date = atoi(line.substr(0, line.find(' ')).c_str()) - 1;
  line = line.substr(line.find(' ') + 1);

  data.rating = atoi(line.c_str());
  return data;
}
