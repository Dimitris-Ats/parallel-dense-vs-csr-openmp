# Makefile for parallel-dense-vs-csr-openmp
CC = gcc
CFLAGS = -O2 -Wall -fopenmp
SRC = src/main.c
TARGET = main


all: $(TARGET)


$(TARGET): $(SRC)
  $(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm


clean:
	rm -f $(TARGET) *.o


.PHONY: all clean
