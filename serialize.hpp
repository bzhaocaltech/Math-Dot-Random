/* Contains helper functions for serializing and unserializing the data */
#include <stdlib.h>
#include <vector>
#include <string>

using namespace std;

/* Serializes a vector of int* each of length len to the given file */
void serialize(std::vector<int*>* vec, string file, int len, bool to_free = true);

/* Unserializes data into a vector of int* where each int* is of length len */
std::vector<int*>* unserialize(string file, int len);
