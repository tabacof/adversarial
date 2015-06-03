INCLUDE = -I.
LDFLAGS :=
LIBOPTS = -shared
CFLAGS = -c -fpic -Wall
CC = gcc


.PHONY : all
all : libParamBank.so

ParamBank.o :
	$(CC) $(CFLAGS) $(INCLUDE) ParamBank.c

libParamBank.so : ParamBank.o
	$(CC) $< $(LIBOPTS) -o $@ $(LDFLAGS)

.PHONY : clean
clean :
	rm -f *.o libParamBank.so
