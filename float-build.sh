mkdir -p Build-VTKm-Float
mkdir -p Installs-Float
cd Build-VTKm-Float
echo "Build and install the VTK-m codebase"
#-DVTKm_USE_DOUBLE_PRECISION=ON off
cmake -DCMAKE_INSTALL_PREFIX=../Installs-Float -DVTKm_ENABLE_OPENMP=ON -DCMAKE_CXX_FLAGS="-fopenmp" -DCMAKE_BUILD_TYPE=Release ../VTK-m
make -j20 install
echo "Build and install the main contour-tree-augmented code used by flexible-isosurfaces (the contour-visualiser)"
cd ../VTK-m/examples/contour_tree_augmented/
mkdir -p Build-Float
cd Build-Float
cmake -DVTKm_ENABLE_OPENMP=ON -DCMAKE_CXX_FLAGS="-fopenmp" -DVTKm_DIR=../../../Installs-Float/lib/cmake/vtkm-2.1/ -DCMAKE_INSTALL_PREFIX=../../../Installs-Float ../
make -j20 install
echo "Build the flexible isosurfaces executable"
cd ../../../../
mkdir -p Flexible-Isosurfaces-Regular
cd Flexible-Isosurfaces-Regular
cmake -DVTKm_ENABLE_OPENMP=ON -DCMAKE_CXX_FLAGS="-fopenmp" -DVTKm_DIR=../Installs-Float/lib/cmake/vtkm-2.1/ -DCMAKE_INSTALL_PREFIX=../Installs-Float ../VTK-m/examples/contour-visualiser
make -j20

