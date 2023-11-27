CC = gcc
CFLAGS = -O3 -march=native -lm -fopenmp -fPIC
SHARED_FLAGS = -shared

all: graph.so hdb.so

graph.so: graph.c
	$(CC) $(SHARED_FLAGS) -o $@ $< $(CFLAGS)

hdb.so: hdb_c.c
	$(CC) $(SHARED_FLAGS) -o $@ $< $(CFLAGS)

clean:
	rm -f graph.so hdb.so
