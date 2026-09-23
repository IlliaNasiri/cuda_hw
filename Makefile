
TARGETS := q1 q2 q3

all: $(TARGETS)

q1: q1.cpp utils.h
	nvcc -arch=native -x cu q1.cpp -o $@

q2: q2.cpp utils.h
	nvcc -arch=native -x cu q2.cpp -o $@

q3: q3.cpp utils.h
	nvcc -arch=native -x cu q3.cpp -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean