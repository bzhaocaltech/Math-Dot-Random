// Takes file names and converts to vector<float>**
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>
#include <stdio.h>
#include "input.hpp"

using namespace std;

std::vector<float>** input(int num_inputs, char* inputs[]) {
  std::vector<float>** to_return = new std::vector<float>*[num_inputs];
  for (int i = 0; i < num_inputs; i++) {
    to_return[i] = new std::vector<float>();
  }

  for (int i = 0; i < num_inputs; i++) {
    fstream input_file;
    fprintf(stderr, "Opening file %s\n", inputs[i]);
    input_file.open(inputs[i], ios::in | ios::binary);
    assert(input_file.is_open());

    string line;
    while(getline(input_file, line)) {
      to_return[i]->push_back(atof(line.c_str()));
    }
    fprintf(stderr, "File %s loaded \n", inputs[i]);
  }

  return to_return;
}
