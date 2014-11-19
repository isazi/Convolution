
include		../Makefile.inc

all: clean Convolution
 
Convolution: Convolution.cpp
	$(CC) -o $(PROJ_BASE)/bin/ConvolutionTest Convolution.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/ConvolutionTest
