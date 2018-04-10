#include "serialize.hpp"
#include <fstream>
#include <stdio.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

/* Serializes a vector of vectors to the given file. Optionally frees the vec
 * after serialization */
void serialize(vector<vector<int>*>* vec, string file, bool to_free) {
  ofstream ofs(file);
  boost::archive::text_oarchive oa(ofs);

  // First serialize the size of the vector
  oa & vec->size();
  int num_lines = 0;

  // Then loop through and serialize each int
  for (unsigned int i = 0; i < vec->size(); i++) {
    vector<int>* arr = (*vec)[i];
    oa & *arr;
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
vector<vector<int>*>* unserialize(string file) {
  ifstream ifs(file);
  boost::archive::text_iarchive ia(ifs);

  // Grab size of vector (equivalent to the number of rows)
  int size = 0;
  ia & size;

  // Unserialize each vector
  vector<vector<int>*>* vec = new vector<vector<int>*>();
  for (int i = 0; i < size; i++) {
    vector<int>* arr = new vector<int>();
    ia & *arr;
    vec->push_back(arr);
  }

  return vec;
}
