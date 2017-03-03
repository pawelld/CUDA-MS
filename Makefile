LIB_PATH=$(HOME)/local/lib
INCLUDE_PATH=$(HOME)/local/include

BASE_CXXFLAGS:= -g -std=c++11 -Wall -Wno-deprecated -D_GNU_SOURCE -I$(INCLUDE_PATH) -m64 -fopenmp

# CUSTOM_FLAGS=-DNO_CUDA

#flags for different builds
BASE_CFLAGS:= -g -Wall -Wno-unknown-pragmas -std=gnu99 -D_GNU_SOURCE $(CUSTOM_FLAGS) -I$(INCLUDE_PATH) -I$(CUDA_HOME)/include/ -m64  -fopenmp
BASE_LDFLAGS:= -g -L$(LIB_PATH) -fopenmp
BASE_LDLIBS:= #-lcurand

CC=$(GCC)
NVCC=nvcc

NVFLAGS=-m64 -g -G  -lineinfo  $(CUSTOM_FLAGS)  -I$(INCLUDE_PATH) -I$(CUDA_HOME)/include/ -Xcompiler "-Wall -g" -Xptxas -v,-warn-lmem-usage,-warn-spills  --maxrregcount 32

BASE_CFLAGS+= -march=native
NVFLAGS+=  -std=c++11 -arch=sm_30
GCC=gcc
GPP=g++
LIBTOOL=libtool

DEBUG_CFLAGS:=  -DSINGLE #-finstrument-functions
DEBUG_LDFLAGS:=
DEBUG_LDLIBS:=
FINAL_CFLAGS:=-ffast-math  -O3 #-mcpu=pentium3
FINAL_LDFLAGS:=
FINAL_LDLIBS:=

CXXFLAGS=$(BASE_CXXFLAGS) $(FINAL_CFLAGS)

CFLAGS=$(BASE_CFLAGS) $(FINAL_CFLAGS) #-pg
LDFLAGS=$(BASE_LDFLAGS) $(FINAL_LDFLAGS)
LDLIBS=$(BASE_LDLIBS) $(FINAL_LDLIBS) #-pg

# CFLAGS=$(BASE_CFLAGS) $(DEBUG_CFLAGS)
# LDFLAGS=$(BASE_LDFLAGS) $(DEBUG_LDFLAGS)
# CXXFLAGS=$(BASE_CXXFLAGS) $(DEBUG_CFLAGS)


MOTZKIN_SOURCES= motzkin.c motzkin_cpu.c
MOTZKIN_CUDA_SOURCES=


ifeq (,$(findstring DNO_CUDA,$(CUSTOM_FLAGS)))
	MOTZKIN_CUDA_SOURCES+= motzkin_cuda.cu
endif

COMMON_SOURCES=parsers.c

ARRAYS_SOURCES=bitops.c random.c arrays.c

SOURCES= $(COMMON_SOURCES) $(ARRAYS_SOURCES) $(MOTZKIN_SOURCES)

CUDA_SOURCES= $(MOTZKIN_CUDA_SOURCES)

%.d: %.c
	$(GCC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o \1.lo $@ :,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.d: %.cu
	$(NVCC) -M $(filter-out -g, $(NVFLAGS))  $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o \1.lo $@ :,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.lo: %.c
	$(LIBTOOL) --mode=compile --tag=CC $(GCC) -c $(CFLAGS) $<

%.lo: %.cc
	$(LIBTOOL) --mode=compile --tag=CXX $(GPP) -c $(CXXFLAGS) $<

%.cc: %.cu
	$(NVCC) -cuda $(NVFLAGS) -Xptxas -v  -o $@ $< # 2> $(@:.cc=.info)



all: libcudamole.la find_cliques

dep: $(SOURCES:.c=.d) $(SOURCES_CXX:.c=.d)



libcudamole.la: $(MOTZKIN_SOURCES:.c=.lo)  $(MOTZKIN_CUDA_SOURCES:.cu=.lo)
	$(LIBTOOL) --mode=link --tag=CXX $(GPP) $(LDFLAGS) -o $@ $^ $(LDLIBS) -lopts -rpath $(LIB_PATH)

find_cliques: find_cliques.o  $(COMMON_SOURCES:.c=.o)  $(ARRAYS_SOURCES:.c=.o) $(MOTZKIN_SOURCES:.c=.o)  $(MOTZKIN_CUDA_SOURCES:.cu=.o)
	$(GCC)  $(LDFLAGS) -o $@ $^   -lstdc++ -L$(CUDA_HOME)/lib64 -lcudart -lcurand

clean:
	rm -fr test opt-check *.o *.d .libs *.lo *.la *.opari.inc tags $(CUDA_SOURCES:.cu=.cc) $(CUDA_SOURCES:.cu=.info) find_cliques

install-dev:
	install -p bitops.h $(INCLUDE_PATH)/bitops.h
	install -p cudams.h $(INCLUDE_PATH)/cudams.h

install: install-dev all
	$(LIBTOOL) --mode=install install libcudamole.la $(LIB_PATH)/libcudamole.la


include $(SOURCES:.c=.d)
include $(CUDA_SOURCES:.cu=.d)


