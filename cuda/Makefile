# CUDA Plummer math-library build. GPU here is sm_75 (GTX 1650); CUDA 11.8 at /usr/local/cuda.
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_75
PYTHON    ?= python3
NVCCFLAGS := -O2 -arch=$(ARCH) -std=c++14 --expt-relaxed-constexpr

testPlummer: testPlummer.cu plummer.cuh
	$(NVCC) $(NVCCFLAGS) testPlummer.cu -o testPlummer

testPlummerPair: testPlummerPair.cu plummer.cuh
	$(NVCC) $(NVCCFLAGS) testPlummerPair.cu -o testPlummerPair

testPlummerGrad: testPlummerGrad.cu plummer.cuh
	$(NVCC) $(NVCCFLAGS) testPlummerGrad.cu -o testPlummerGrad

testSharpPacking: testSharpPacking.cu sharpKernels.cu sharpKernels.cuh sharp.cuh
	$(NVCC) $(NVCCFLAGS) testSharpPacking.cu sharpKernels.cu -o testSharpPacking

# Exact-arc device math. Checked by tests/roundedContactDeviceCheck.py, which feeds it the hard root
# cases on stdin and compares every output against roundedContact.py -- a kernel that agrees with
# itself proves nothing.
testRoundedContact: testRoundedContact.cu roundedContact.cuh
	$(NVCC) $(NVCCFLAGS) testRoundedContact.cu -o testRoundedContact

# Shared library behind cudaOverlap.py: every energy term that runs on the GPU.
LIBSRC := plummerDriver.cu selfRepulsion.cu springs.cu sharpDriver.cu sharpKernels.cu measure.cu \
          container.cu neighbors.cu softDepthKernels.cu polyContactKernels.cu \
          roundedContactKernels.cu
libplummer.so: $(LIBSRC) plummer.cuh sharp.cuh sharpKernels.cuh softDepth.cuh polyContact.cuh \
               roundedContact.cuh
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared $(LIBSRC) -o libplummer.so

# Regenerate reference vectors from energies.py, then build+run the validations.
.PHONY: check vectors clean
vectors:
	$(PYTHON) genVectors.py

check: testPlummer testPlummerPair testPlummerGrad testSharpPacking
	./testPlummer
	./testPlummerPair
	./testPlummerGrad
	./testSharpPacking

clean:
	rm -f testPlummer testPlummerPair testPlummerGrad testSharpPacking
