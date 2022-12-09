# From https://x.momo86.net/?p=29
CXX=g++
CXXFLAGS=-std=c++11 -I./include -O3 -g -G -Xcompiler -Wall -lm

NVCC=nvcc
ARCH=sm_75
NVCCFLAGS=-I./include -arch=$(ARCH) -std=c++11 -O3 -g -G -Xcompiler -Wall --compiler-bindir=$(CXX) -lm

SRCDIR:=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR:=src
OBJS:=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

BIN:=bin
TARGET:=knn
TEST_TARGET:=test

OBJS_KNN:=$(filter-out $(SRCDIR)/$(TEST_TARGET).o, $(OBJS))
OBJS_TEST:=$(filter-out $(SRCDIR)/$(TARGET).o, $(OBJS))

all: test knn

knn: CXXFLAGS += -DNDEBUG
knn: NVCCFLAGS += -DNDEBUG
knn: dir $(BIN)/$(TARGET)

test: dir $(BIN)/$(TEST_TARGET)

dir: ${BIN}
  
${BIN}:
	mkdir -p $(BIN)

$(BIN)/$(TARGET): $(OBJS_KNN)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(BIN)/$(TEST_TARGET): $(OBJS_TEST)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(BIN)/$(TARGET)
	rm -rf $(BIN)/$(TEST_TARGET)