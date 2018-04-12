CXX = g++
CXXFLAGS = -std=c++14 -Wall -g
BOOSTROOT = -L/usr/local/lib
BOOSTSERIALIZE = -lboost_serialization
.PHONY: clean
EXECUTABLES = load_data model_1

all: $(EXECUTABLES)

load_data: load_data.o
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) -static load_data.o serialize.o -o load_data $(BOOSTSERIALIZE)

serialize.o: serialize.cpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp

load_data.o: load_data.cpp serialize.o
	$(CXX) $(CXXFLAGS) -c load_data.cpp

model.o: model.hpp model.cpp
	$(CXX) $(CXXFLAGS) -c model.cpp

output.o: output.hpp output.cpp
	$(CXX) $(CXXFLAGS) -c output.cpp

mean_model.o: model.o mean_model.cpp mean_model.hpp
	$(CXX) $(CXXFLAGS) -c mean_model.cpp

model_1: mean_model.o serialize.o model_1.o model.o
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) -static model_1.o serialize.o mean_model.o model.o -o model_1 $(BOOSTSERIALIZE)

clean:
	$(RM) $(EXECUTABLES) *.o
