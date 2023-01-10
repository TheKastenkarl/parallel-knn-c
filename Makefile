# From https://x.momo86.net/?p=29
CXX=g++
#cxx flags for compiling
CXXFLAGS=-std=c++11 -I./include -O2 -g -G -Xcompiler -Wall -lm # add -pg flag for gprof profiling

NVCC=nvcc
# specify required architecture. E.g. for google colab Tesla T4: sm_75
ARCH=sm_75
# nvcc flags for compiling. -lnvToolsExt is added for linking the nvidia library nvtx3
NVCCFLAGS=-I./include -arch=$(ARCH) -std=c++11 -O2 -g -G -Xcompiler -Wall --compiler-bindir=$(CXX) -lm -lnvToolsExt # add -pg flag for gprof profiling

# specify directories and relevant files
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

# "make clean" should remove all outputs of "make all"
clean:
	rm -rf $(OBJS)
	rm -rf $(BIN)/$(TARGET)
	rm -rf $(BIN)/$(TEST_TARGET)