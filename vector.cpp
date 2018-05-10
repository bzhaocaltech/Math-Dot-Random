// Thread-safe vector implementation.

#include "vector.hpp"
#include <algorithm>

Vector::Vector(int length) {
    this->vector = new float[length];
    this->vector_locks = new std::vector<std::mutex*>(length);
    std::generate (this->vector_locks->begin(), this->vector_locks->end(),
        [] () {
            return new std::mutex();
        }
    );
}

void Vector::update_element(int idx, float val) {
    this->vector_locks->at(idx)->lock();
    this->vector[idx] = val;
    this->vector_locks->at(idx)->unlock();
}

float Vector::at(int idx) {
    return this->vector[idx];
}

Vector::~Vector() {
    delete this->vector;
    delete this->vector_locks;
}
