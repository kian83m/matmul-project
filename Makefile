# ---
# Platform-dependent configuration
#
# If you have multiple platform-dependent configuration options that you want
# to play with, you can put them in an appropriately-named Makefile.in.
# For example, the default setup has a Makefile.in.icc and Makefile.in.gcc.

PLATFORM=cuda

include Makefile.in.$(PLATFORM)
DRIVERS=$(addprefix batched_,$(BUILDS))
TIMINGS=$(addsuffix .csv,$(addprefix timing-,$(BUILDS)))

.PHONY:	all
all:	$(DRIVERS)

# ---
# Rules to build the drivers

batched_%: $(OBJS) batched_%.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

batched_basic: $(OBJS) batched_basic.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

batched_cuBlas-basic: $(OBJS) batched_cuBlas-basic.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBcuBLAS)

batched_cuBlas-advance: $(OBJS) batched_cuBlas-advance.o
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBcuBLAS)


# --
# Rules to build object files

matmul.o: matmul.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $< 

batched_basic.o: batched_basic.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $<



%.o: %.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $<

%.o: %.cpp
	$(NVCC) -o $@ -c $(NVCCFLAGS) $<

# ---
# Rules for building timing CSV outputs

.PHONY: run
run:    $(TIMINGS)

timing-%.csv: batched_%
	./batched_$* $@

# ---
#  Rules for plotting

.PHONY: plot
plot:
	$(PYTHON) plotter.py $(BUILDS)

# ---

.PHONY:	clean realclean 
clean:
	rm -f matmul-* *.o

realclean: clean
	rm -f *~ timing-*.csv timing_*.csv dump_*.txt $(DRIVERS)
