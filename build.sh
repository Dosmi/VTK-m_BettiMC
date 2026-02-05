mkdir -p Build-VTKm
mkdir -p Installs
cd Build-VTKm
echo "Build and install the VTK-m codebase"
cmake -DCMAKE_INSTALL_PREFIX=../Installs -DVTKm_USE_DOUBLE_PRECISION=ON  -DVTKm_ENABLE_OPENMP=ON -DCMAKE_CXX_FLAGS="-fopenmp" -DCMAKE_BUILD_TYPE=Release ../VTKm-main
make -j20 install
echo "Build and install the main contour-tree-augmented code used by flexible-isosurfaces (the contour-visualiser)"
cd ../VTKm-main/examples/contour_tree_augmented/
mkdir -p Build
cd Build
cmake -DVTKm_USE_DOUBLE_PRECISION=ON  -DVTKm_ENABLE_OPENMP=ON -DCMAKE_CXX_FLAGS="-fopenmp" -DVTKm_DIR=../../../Installs/lib/cmake/vtkm-2.1/ -DCMAKE_INSTALL_PREFIX=../../../Installs ../
make -j20 install
echo "Build the flexible isosurfaces executable"
cd ../../../../
mkdir -p Flexible-Isosurfaces-Irregular
cd Flexible-Isosurfaces-Irregular
cmake -DVTKm_USE_DOUBLE_PRECISION=ON  -DVTKm_ENABLE_OPENMP=ON -DCMAKE_CXX_FLAGS="-fopenmp" -DVTKm_DIR=../Installs/lib/cmake/vtkm-2.1/ -DCMAKE_INSTALL_PREFIX=../Installs ../VTKm-main/examples/contour-visualiser
make -j20

