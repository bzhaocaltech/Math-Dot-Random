// Thread-safe vector implementation.

#include "vector.hpp"
#include <algorithm>

Vector::Vector(int length) {
    this->vector = new float[length];
    this->vector_locks = new std::mutex*[length];
    for (int i = 0; i < length; i++) {
      this->vector_locks[i] = new std::mutex();
    }
    this->length = length;
}

Vector::Vector(int length, float* vector) {
    this->vector = vector;
    this->vector_locks = new std::mutex*[length];
    for (int i = 0; i < length; i++) {
      this->vector_locks[i] = new std::mutex();
    }
    this->length = length;
}

float* Vector::get_vector() {
    return this->vector;
}

void Vector::update_element(int idx, float val) {
    if (idx > length) {
      fprintf(stderr, "Out of bounds for update for vector \n");
      exit(1);
    }
    this->vector_locks[idx]->lock();
    this->vector[idx] = val;
    this->vector_locks[idx]->unlock();
}

float Vector::at(int idx) {
    if (idx > length) {
      fprintf(stderr, "Out of bounds for access for vector \n");
      exit(1);
    }
    return this->vector[idx];
}

Vector::~Vector() {
    delete this->vector;
    for (int i = 0; i < length; i++) {
        delete this->vector_locks[i];
    }
    delete this->vector_locks;
}
