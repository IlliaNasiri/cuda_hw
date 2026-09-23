NVCC := nvcc
ARCH ?= 89
FLAGS := -std=c++17 -arch=sm_$(ARCH)

TARGETS := hw2 q1 q2 q3

all: $(TARGETS)

hw2: main.cu utils.h
	$(NVCC) $(FLAGS) -rdc=true main.cu -o $@

q1: q1.cpp utils.h
	$(NVCC) $(FLAGS) -x cu q1.cpp -o $@

q2: q2.cpp utils.h
	$(NVCC) $(FLAGS) -x cu q2.cpp -o $@

q3: q3.cpp utils.h
	$(NVCC) $(FLAGS) -x cu q3.cpp -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean