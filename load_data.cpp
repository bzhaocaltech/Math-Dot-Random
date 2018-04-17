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

int* parse_line(string line);

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
  vector<int*>* mu_train = new vector<int*>();
  vector<int*>* mu_valid = new vector<int*>();;
  vector<int*>* mu_hidden = new vector<int*>();;
  vector<int*>* mu_probe = new vector<int*>();;
  vector<int*>* mu_qual = new vector<int*>();;
  fprintf(stderr, "Loading mu files...");
  int num_lines = 0;
  while (getline(mu_dta, line) && getline(mu_idx, index_string)) {
    int index = atoi(index_string.c_str());
    int* parsed_line = parse_line(line);
    // Part of base
    if (index == 1) {
      mu_train->push_back(parsed_line);
    }
    // Part of the validation set
    if (index == 2) {
      mu_valid->push_back(parsed_line);
    }
    // Part of the hidden set
    if (index == 3) {
      mu_hidden->push_back(parsed_line);
    }
    // Part of probe
    if (index == 4) {
      mu_probe->push_back(parsed_line);
    }
    // Part of qual
    if (index == 5) {
      mu_qual->push_back(parsed_line);
    }
    num_lines++;
    if (num_lines % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");

  // Serialize the mu files
  fprintf(stderr, "Serializing mu files...");
  serialize(mu_train, "data/mu_train.ser", 4);
  serialize(mu_valid, "data/mu_valid.ser", 4);
  serialize(mu_hidden, "data/mu_hidden.ser", 4);
  serialize(mu_probe, "data/mu_probe.ser", 3);
  serialize(mu_qual, "data/mu_qual.ser", 3);
  fprintf(stderr, "\n");

  // Get the um training data
  vector<int*>* um_train = new vector<int*>();
  vector<int*>* um_valid = new vector<int*>();;
  vector<int*>* um_hidden = new vector<int*>();;
  vector<int*>* um_probe = new vector<int*>();;
  vector<int*>* um_qual = new vector<int*>();;
  fprintf(stderr, "Loading um files...");
  num_lines = 0;
  while (getline(um_dta, line) && getline(um_idx, index_string)) {
    int index = atoi(index_string.c_str());
    int* parsed_line = parse_line(line);
    // Part of base
    if (index == 1) {
      um_train->push_back(parsed_line);
    }
    // Part of the validation set
    if (index == 2) {
      um_valid->push_back(parsed_line);
    }
    // Part of the hidden set
    if (index == 3) {
      um_hidden->push_back(parsed_line);
    }
    // Part of probe
    if (index == 4) {
      um_probe->push_back(parsed_line);
    }
    // Part of qual
    if (index == 5) {
      um_qual->push_back(parsed_line);
    }
    num_lines++;
    if (num_lines % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");

  // Serialize the um files
  fprintf(stderr, "Serializing um files...");
  serialize(um_train, "data/um_train.ser", 4);
  serialize(um_valid, "data/um_valid.ser", 4);
  serialize(um_hidden, "data/um_hidden.ser", 4);
  serialize(um_probe, "data/um_probe.ser", 3);
  serialize(um_qual, "data/um_qual.ser", 3);
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
int* parse_line(string line) {
  int* int_array = (int*) malloc(4 * sizeof(int));
  for (int i = 0; i < 3; i++) {
    int_array[i] = atoi(line.substr(0, line.find(' ')).c_str()) - 1;
    line = line.substr(line.find(' ') + 1);
  }
  int_array[3] = atoi(line.c_str());
  return int_array;
}
