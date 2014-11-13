CC=g++

SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

# Only enable -msse4.2 on CPUs supporting the POPCNT instruction
CFLAGS = -Wall `pkg-config opencv --cflags` -I./include -O3 -fopenmp -msse4.2
LDFLAGS = `pkg-config opencv --libs` -lgomp

all: $(EXECUTABLE)

debug: CFLAGS += -g -O0 -Wextra
debug: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES) $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@ 

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
