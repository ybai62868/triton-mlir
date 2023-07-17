make clean && cmake ..
make -j128 && cp libtriton.so ../python/triton/_C
rm -rf ~/.triton
python gemm.py
