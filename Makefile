CXX = g++
CXXFLAGS = -std=c++14 -Wall -g -DNDEBUG -DBOOST_UBLAS_NDEBUG
BOOSTROOT = -L/usr/local/lib -static
BOOSTSERIALIZE = -lboost_serialization
.PHONY: clean
EXECUTABLES = load_data model_1 model_2
LOAD_DATA_DEP = load_data.o serialize.o
MODEL_1_DEP = mean_model.o serialize.o model_1.o model.o output.o
MODEL_2_DEP = matrix.o svd.o model_2.o model.o serialize.o output.o

all: $(EXECUTABLES)

load_data: $(LOAD_DATA_DEP)
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) $(LOAD_DATA_DEP) -o load_data $(BOOSTSERIALIZE)

serialize.o: serialize.cpp serialize.hpp data.hpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp

load_data.o: load_data.cpp
	$(CXX) $(CXXFLAGS) -c load_data.cpp

model.o: model.hpp model.cpp
	$(CXX) $(CXXFLAGS) -c model.cpp

output.o: output.hpp output.cpp
	$(CXX) $(CXXFLAGS) -c output.cpp

mean_model.o: model.o mean_model.cpp mean_model.hpp
	$(CXX) $(CXXFLAGS) -c mean_model.cpp

svd.o: model.o svd.cpp svd.hpp
	$(CXX) $(CXXFLAGS) -c svd.cpp

matrix.o: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp

model_1: $(MODEL_1_DEP)
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) $(MODEL_1_DEP) -o model_1 $(BOOSTSERIALIZE)

model_2: $(MODEL_2_DEP)
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) $(MODEL_2_DEP) -o model_2 $(BOOSTSERIALIZE)

clean:
	$(RM) $(EXECUTABLES) *.o results.dta
