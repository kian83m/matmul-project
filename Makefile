# ---
# Platform-dependent configuration
#
# If you have multiple platform-dependent configuration options that you want
# to play with, you can put them in an appropriately-named Makefile.in.
# For example, the default setup has a Makefile.in.icc and Makefile.in.gcc.

PLATFORM=cuda

include Makefile.in.$(PLATFORM)
DRIVERS=$(addprefix matmul-,$(BUILDS))
TIMINGS=$(addsuffix .csv,$(addprefix timing-,$(BUILDS)))

.PHONY:	all
all:	$(DRIVERS)

# ---
# Rules to build the drivers

matmul-%: $(OBJS) dgemm_%.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-f2c: $(OBJS) dgemm_f2c.o dgemm_f2c_desc.o fdgemm.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) -lifcore -lifport

matmul-blas: $(OBJS) dgemm_blas.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBBLAS)

matmul-mkl: $(OBJS) dgemm_mkl.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBMKL)

matmul-veclib: $(OBJS) dgemm_veclib.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) -framework Accelerate

matmul-mine: $(OBJS) dgemm_mine.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) 

matmul-basic: $(OBJS) dgemm_basic.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-gpu: $(OBJS) dgemm_gpu.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-gpu_basic: $(OBJS) dgemm_gpu_basic.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-cublas: $(OBJS) dgemm_cublas.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBcuBLAS)

# --
# Rules to build object files

# matmul.o: matmul.cpp
# 	$(CC) -c $(CFLAGS) $(CPPFLAGS) $<
matmul.o: matmul.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $< 

%.o: %.c
	$(CC) -c $(CFLAGS) $(OPTFLAGS) $(CPPFLAGS) $<

%.o: %.f
	$(FC) -c $(FFLAGS) $(OPTFLAGS) $<

dgemm_blas.o: dgemm_blas.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCBLAS) $< 

dgemm_mkl.o: dgemm_blas.c
	$(CC) -o $@ -c $(CFLAGS) $(CPPFLAGS) $(INCMKL) $< 

dgemm_veclib.o: dgemm_blas.c
	clang -o $@ -c $(CFLAGS) $(CPPFLAGS) -DOSX_ACCELERATE $< 


dgemm_gpu.o: dgemm_gpu.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $< 

dgemm_cublas.o: dgemm_cublas.cpp
	$(NVCC) -o $@ -c $(NVCCFLAGS) $<

dgemm_gpu_basic.o: dgemm_gpu_basic.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $< 
# ---
# Rules for building timing CSV outputs

.PHONY: run
run:    $(TIMINGS)

timing-%.csv: matmul-%
	./matmul-$*

# ---
#  Rules for plotting

.PHONY: plot
plot:
	$(PYTHON) plotter.py $(BUILDS)

# ---

.PHONY: run-matmul-mine
run-matmul-mine: matmul-mine
	./matmul-mine
	$(PYTHON) plotter.py mine

.PHONY:	clean realclean 
clean:
	rm -f matmul-* *.o

realclean: clean
	rm -f *~ timing-*.csv timing.pdf dump_*.txt 
