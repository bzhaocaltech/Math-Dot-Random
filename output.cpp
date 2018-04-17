#include "output.hpp"
#include <iostream>
#include <fstream>

/* Outputs a vector<float> of predictions to a given file */
void output(vector<float> predictions, string file) {
    ofstream file_out;
    file_out.open(file);
    for (vector<float>::iterator iter = predictions.begin();
                                            iter != predictions.end(); ++iter) {

        file_out << *iter << "\n";
    }
    file_out.close();
}
