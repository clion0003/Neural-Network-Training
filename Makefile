GPU=0
DEBUG=0

ARCH= --gpu-architecture=compute_52 --gpu-code=compute_52

VPATH=./src/
EXEC=memristor
OBJDIR=./obj/

CC=c++
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -std=c++11 -I./
CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda-7.5/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda-7.5/lib64 -lcuda -lcudart -lcublas -lcurand
endif

OBJ=main.o activation.o fault.o fullyconnected_layer.o convolutional_layer.o maxpooling_layer.o output_layer.o util.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=fullyconnected_layer_kernel.o output_layer_kernel.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

