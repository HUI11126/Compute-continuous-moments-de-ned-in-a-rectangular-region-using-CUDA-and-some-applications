cd build
rm -r *
# cmake ..
cmake -D CMAKE_BUILD_TYPE=RELEASE .. 
make
cd ..
# ./build/edge_extraction
./build/reconstruction