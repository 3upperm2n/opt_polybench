CLROOT = /opt/AMDAPP
CFLAG = -std=c99 -Wall

INCLUDE = -I$(CLROOT)/include
LIBPATH = -L$(CLROOT)/lib/x86_64 

LIB=-lOpenCL -lm

all:
	gcc -O3 $(CFLAG) ${INCLUDE} ${LIBPATH} ${LIB} ${CFILES} -o ${EXECUTABLE}

clean:
	rm -f *~ *.exe
