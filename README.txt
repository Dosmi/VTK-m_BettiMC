OMP_NUM_THREADS=1 ./ContourVisualiser -f ../Data/out_hh_24_coarsened_ascii.txt -o output -t 10 --decompositionType volume


Run the contour tree augmented from:
/home/sc17dd/modules/HCTC2024/VTK-m_BettiMC/VTK-m/examples/contour_tree_augmented/Build
Marching cubes setting is --mc (just the flag is enough)
./ContourTree_Augmented --vtkm-device=Any --printCT --mc --vtkm-log-level=INFO ../../../../Data/5b-unique.txt

dot command:
dot -Tsvg ContourTreeGraph-56M-original-fullCT.gv > 5b-marching-cubes-regular.svg
