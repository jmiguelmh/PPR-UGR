all: bbmpi

bbmpi: bbmpi.o libbb.o
	mpicxx -O3 bbmpi.o libbb.o -o bbmpi

bbmpi.o: bbmpi.cc
	mpicxx -O3 -c bbmpi.cc

libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc 


clean:
	/bin/rm -f *.o bbmpi



