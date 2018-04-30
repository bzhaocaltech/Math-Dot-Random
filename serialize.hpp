/* Contains helper functions for serializing and unserializing the data */
#include <stdlib.h>
#include <vector>
#include <string>
#include "data.hpp"
#include <cstdint>

using namespace std;

/* Serializes a dataset to the given file
 * The bool free determines whether or not the vector is freed at the end of
 * serialization */
void serialize(struct dataset* s, string file, bool to_free = true);

/* Unserializes a serialized file into a dataset* */
struct dataset* unserialize(string file);
