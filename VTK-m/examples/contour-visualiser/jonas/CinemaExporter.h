
#include <iostream>

#include <vtkm/cont/DataSet.h>

namespace CinemaExporter {

  // This method adds a global field to the dataset with a given underlying data type, name, and data
  template<typename T>
  void AddGlobalField(
    vtkm::cont::DataSet& dataSet,
    std::string name,
    std::vector<T> vector
  ){
    dataSet.AddField(
      vtkm::cont::Field(
        name,
        vtkm::cont::Field::Association::WholeDataSet,
        vtkm::cont::make_ArrayHandle<T>(vector,vtkm::CopyFlag::On)
      )
    );
  }

  // This method returns a filename for a dataset by concatenating all 1-component field data values
  std::string GetFileName(
    vtkm::cont::DataSet& dataSet
  );

  // renders and writes images of the dataset
  void RenderAndWrite(
    int contourId,
    vtkm::cont::DataSet& dataSet,
    vtkm::Bounds& bounds,
    const std::string outputDirectory,
    int resX,
    int resY,
    int elevation0, int elevation1, int elevationStep,
    int azimuth0, int azimuth1, int azimuthStep
  );
}
