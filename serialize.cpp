#include "serialize.hpp"
#include <fstream>
#include <stdio.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;

/* Serializes a vector of int* each of length len to the given file.
 * The bool free determines whether or not the vector is freed at the end of
 * serialization */
void serialize(std::vector<int*>* vec, string file, int len, bool to_free) {
  ofstream ofs(file);
  boost::archive::text_oarchive oa(ofs);

  // First serialize the size of the vector
  oa & vec->size();
  int num_lines = 0;

  // Then loop through and serialize each int
  for (unsigned int i = 0; i < vec->size(); i++) {
    int* arr = (*vec)[i];
    for (int j = 0; j < len; j++) {
      oa & arr[j];
    }
    if (to_free) {
      free(arr);
    }
    num_lines++;
    if (num_lines % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  if (to_free) {
    delete(vec);
  }
}

/* Unserializes data into a vector of int* where each int* is of length len */
std::vector<int*>* unserialize(string file, int len) {
  fprintf(stderr, "Unserializing %s", file.c_str());
  ifstream ifs(file);
  boost::archive::text_iarchive ia(ifs);

  // Grab size of vector (equivalent to the number of rows)
  int size = 0;
  ia & size;

  // Unserialize each int
  std::vector<int*>* vec = new vector<int*>();
  for (int i = 0; i < size; i++) {
    int* arr = (int*) malloc(sizeof(int) * len);
    for (int j = 0; j < len; j++) {
      ia >> *(arr + j);
    }
    vec->push_back(arr);
    if (i % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");
  return vec;
}
