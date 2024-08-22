# Makefile for compiling C_sim with math library and OpenMP support

# Compiler for C++
CXX = g++


# Compiler flags
CXXFLAGS = -Wall -Wextra -O2 -fopenmp -std=c++17 -I$(XILINX_HLS)/include

# Target executable
TARGET = c_sim

# Source files
SRCS = c_sim.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lm

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean

