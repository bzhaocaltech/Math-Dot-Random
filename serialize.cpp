#include "serialize.hpp"
#include <fstream>
#include <stdio.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;

/* Serializes a vector of int* each of length len to the given file.
 * The bool free determines whether or not the vector is freed at the end of
 * serialization */
void serialize(struct dataset* s, string file, bool to_free) {
  ofstream ofs(file);
  boost::archive::text_oarchive oa(ofs);

  // First serialize the size of the dataset
  oa & s->size;
  int num_lines = 0;

  // Then loop through and serialize each value of data
  for (int i = 0; i < s->size; i++) {
    struct data d = s->data[i];

    // Serialize the different members of a data struct
    oa & d.user;
    oa & d.movie;
    oa & d.date;
    oa & d.rating;

    num_lines++;
    if (num_lines % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  if (to_free) {
    delete s->data;
    delete s;
  }
}

/* Unserializes a serialized file into a dataset* */
struct dataset* unserialize(string file) {
  fprintf(stderr, "Unserializing %s", file.c_str());
  ifstream ifs(file);
  boost::archive::text_iarchive ia(ifs);

  // Grab size of dataset
  struct dataset* dataset = new struct dataset();
  int size = 0;
  ia & size;
  dataset->size = size;
  dataset->data = new data[size];

  // Unserialize each data member
  for (int i = 0; i < size; i++) {
    struct data d = dataset->data[i];
    ia & d.user;
    ia & d.movie;
    ia & d.date;
    ia & d.rating;

    if (i % 3000000 == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");
  return dataset;
}
