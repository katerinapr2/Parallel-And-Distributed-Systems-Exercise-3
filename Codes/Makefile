SHELL := /bin/bash

CC = gcc -O3
NVCC = nvcc 
RM = rm -f

EXECUTABLES = v0 v1 v2

all: $(EXECUTABLES)

v0: v0.c
	$(CC) $< -o ./$@ -lm

v1: v1.cu
	$(NVCC) $< -o ./$@ 
	
v2: v2.cu
	$(NVCC) $< -o ./$@ 

clean:
	$(RM) *.o *~ $(EXECUTABLES)

default:
	all
