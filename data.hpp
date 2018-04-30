#ifndef DATA_HPP
#define DATA_HPP

/* This files defines the basic data structs we will use while doing stuff */
#include <cstdint>

// Size of datasets
#define SIZE_TRAIN 94362233
#define SIZE_VALID 1965045
#define SIZE_HIDDEN 1964391
#define SIZE_PROBE 1374739
#define SIZE_QUAL 2749898

/* Basic data type representing a simple data point with user, movie, data,
 * and rating */
struct data {
  uint32_t user;
  uint16_t movie;
  uint16_t date;
  uint8_t rating;
};

/* Wrapper for a data array. Crucially, contains the size of the array */
struct dataset {
  struct data* data;
  int size;
};

#endif
