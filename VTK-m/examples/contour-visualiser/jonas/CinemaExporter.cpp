
#include "CinemaExporter.h"
#include <iostream>


#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/rendering/ScalarRenderer.h>

#include "VTKDataSetWriter2.h"

std::string CinemaExporter::GetFileName(
  vtkm::cont::DataSet& dataSet
){
  std::string name = "";
  for (vtkm::Id f = 0; f < dataSet.GetNumberOfFields(); f++)
  {
    auto field = dataSet.GetField(f);
    //    if (!field.IsFieldGlobal()) old, deprecated name
    if (!field.IsGlobalField())
      continue;

    auto data = field.GetData();

    vtkm::IdComponent nComponents = data.GetNumberOfComponentsFlat();

    if(nComponents==1){
      if(data.IsBaseComponentType<vtkm::Float32>()){
        auto componentArray = data.ExtractArrayFromComponents<vtkm::Float32>();
        auto portal = componentArray.ReadPortal();
        name += std::to_string(portal.Get(0));
      } else {
        auto componentArray = data.ExtractArrayFromComponents<vtkm::Int32>();
        auto portal = componentArray.ReadPortal();
        name += std::to_string(portal.Get(0));
      }
      name += "_";
    }
  }

  if(name.length()<1)
    name="FILE.";
  else
    name[name.length()-1] = '.';

  // name = name.substr(0, name.length()-1);
  name += "vtk";

  return name;
}

void CinemaExporter::RenderAndWrite(
  int contourId,
  vtkm::cont::DataSet& dataSet,
  vtkm::Bounds& bounds,
  const std::string outputDirectory,
  int resX, int resY,
  int elevation0, int elevation1, int elevationStep,
  int azimuth0, int azimuth1, int azimuthStep
){
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);

  vtkm::rendering::ScalarRenderer renderer;
  renderer.SetInput(dataSet);
  renderer.SetDefaultValue(1.0);
  renderer.SetWidth(resX);
  renderer.SetHeight(resY);

  for(int elevation=elevation0; elevation<=elevation1; elevation+=elevationStep){
    camera.Elevation((vtkm::Float32)elevation);
    for(int azimuth=azimuth0; azimuth<azimuth1; azimuth+=azimuthStep){
      camera.Azimuth((vtkm::Float32)azimuth);

      // render
      std::cout<<"Rendering "<<elevation<<" "<<azimuth<<std::endl;
      vtkm::rendering::ScalarRenderer::Result res = renderer.Render(camera);

      vtkm::cont::DataSet result = res.ToDataSet();

      AddGlobalField<vtkm::Int32>(result, "ContourId", {contourId});

      AddGlobalField<vtkm::Int32>(result, "CamAzimuth", {azimuth});
      AddGlobalField<vtkm::Int32>(result, "CamElevation", {elevation});

      AddGlobalField<vtkm::Vec3f_32>(result, "CamPosition", {camera.GetPosition()});
      AddGlobalField<vtkm::Vec3f_32>(result, "CamUp", {camera.GetViewUp()});
      AddGlobalField<vtkm::Vec3f_32>(result, "CamDirection", {(camera.GetLookAt()-camera.GetPosition())});

      auto outPath = GetFileName(result);
      std::cout<<"Writing "<<outPath<<std::endl;
      VTKDataSetWriter2 writer(outputDirectory + outPath);
      writer.WriteDataSet(result);
    }
  }
}
