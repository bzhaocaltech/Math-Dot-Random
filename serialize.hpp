/* Contains helper functions for serializing and unserializing the data */
#include <stdlib.h>
#include <vector>
#include <string>

using namespace std;

/* Serializes a vector of vectors to the given file. Optionally frees the vec
 * after serialization */
void serialize(vector<vector<int>*>* vec, string file, bool to_free = true);

/* Unserializes data into a vector of int* where each int* is of length len */
vector<vector<int>*>* unserialize(string file);
