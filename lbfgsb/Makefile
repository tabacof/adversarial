FFLAGS = -fPIC -O2 -fbounds-check -g -Wno-uninitialized

#wrapper.o:
#	gcc -fPIC -c wrapper.c -lm
	
#liblbfgsb.so : lbfgsb.o timer.o linpack.o wrapper.o
#	gfortran -shared lbfgsb.o linpack.o timer.o wrapper.o -o liblbfgsb.so -lblas

liblbfgsb.so : lbfgsb.o timer.o linpack.o
	gfortran -shared lbfgsb.o linpack.o timer.o -o liblbfgsb.so -lblas
	
#example: liblbfgsb.so 
#	gcc -L. -Wl,-rpath=/home/tabacof/Documents/Lbfgsb.3.0/ example.c -o example -llbfgsb

linpack.o: linpack.f
	gfortran $(FFLAGS) -c linpack.f
	
timer.o: timer.f
	gfortran $(FFLAGS) -c timer.f
	
lbfgsb.o: lbfgsb.f
	gfortran $(FFLAGS) -c lbfgsb.f

#all: example

lib: liblbfgsb.so

clean: 
	rm *.o *.so example
