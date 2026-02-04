//============================================================================
// Copyright (c) 2019, The Regents of the University of California, through
// Lawrence Berkeley National Laboratory (subject to receipt of any required approvals
// from the U.S. Dept. of Energy).  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence Berkeley National
//     Laboratory, U.S. Dept. of Energy nor the names of its contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//============================================================================

#include "./ct.h"
#include "./filter.h"
#include "./mc.h"
#include "./triangle.h"
#include "./interface.h"

#include "../libraries/CLI11.hpp"

#include "./jonas/CinemaExporter.h"

#include <vtkm/cont/Initialize.h>
#include <vtkm/cont/PartitionedDataSet.h>

#include <vtkm/io/VTKDataSetWriter.h>
// in VTK 2.2, both VTKDataSetWriter and VTKDataSetReader are under vtkm/io
//#include <vtkm/io/reader/VTKDataSetReader.h> old version
#include <vtkm/io/VTKDataSetReader.h>
// same with BOVDataSetReader:
//#include <vtkm/io/reader/BOVDataSetReader.h>
#include <vtkm/io/BOVDataSetReader.h>

//#include <vtkm/cont/DataSetFieldAdd.h> deprecated since VTK 1.6 https://gitlab.kitware.com/vtk/vtk-m/-/releases/v1.6.0
#include <vtkm/cont/DataSetBuilderUniform.h>

//#define DEBUG_PRINT

using namespace std;
using namespace vtkm;

cont::DataSet read2DUniformGridData(std::string fileName, std::string fieldName)
{
    bool debug = false;

    // Open File
    fstream file;
    file.open(fileName, ios::in);

    if (!file.is_open())
    {
        return cont::DataSetBuilderUniform::Create(Id2(0, 0));
    }

    // Read Dimensions (swapped to get the correct output)
    int xdim, ydim;
    file >> xdim >> ydim;

    const Id2 pointDimensions(ydim, xdim);

    // Read Data Values
    cont::ArrayHandle<Float64, VTKM_DEFAULT_STORAGE_TAG> fieldVals;
    fieldVals.Allocate(xdim * ydim);
    auto fieldValsPortal = fieldVals.WritePortal();

    for (Id i = 0; i < xdim * ydim; i++)
    {
        Float64 fieldValue;
        file >> fieldValue;
        fieldValsPortal.Set(i, fieldValue);
    }
    
    file.close();

    // Create 3D Uniform Grid
    cont::DataSet inputData = cont::DataSetBuilderUniform::Create(pointDimensions);

    // Add field to data (DataSetFieldAdd.h deprecated since VTK 1.6 https://gitlab.kitware.com/vtk/vtk-m/-/releases/v1.6.0)
    //    cont::DataSetFieldAdd::AddPointField(inputData, fieldName, fieldVals);
    // now can simply do:
    inputData.AddPointField(fieldName, fieldVals);

    if (true == debug)
    {
        // Print some useful debug information
//        using PortalConstType = typename vtkm::cont::ArrayHandle<vtkm::Range>::PortalConstControl;
        auto readPortal = inputData.GetPointField(fieldName).GetRange().ReadPortal();
        cout << "The range of the field is " << readPortal.Get(0) << endl;

        cout << "Dimensions of the data set are " << pointDimensions[0] << " " << pointDimensions[1]  << endl;

        Vec<Range, 3> coordinateRange = inputData.GetCoordinateSystem().GetRange();

        printf("The range of the X dimensions is [%.2f, %.2f]\n", coordinateRange[0].Min, coordinateRange[0].Max);
        printf("The range of the Y dimensions is [%.2f, %.2f]\n", coordinateRange[1].Min, coordinateRange[1].Max);
    }

    return inputData;
}

int main(int argc, char* argv[])
{
    // Initialising VTKm (for things like log level, setting calling thread, etc.)
    auto opts = cont::InitializeOptions::DefaultAnyDevice;
    cont::InitializeResult config = cont::Initialize(argc, argv, opts);


#ifdef DEBUG_PRINT
    cout << "This is a debug print.";
#endif

    // Application Input
    CLI::App cliApp("Contour Visualiser 1000");

    string fileName;
    cliApp.add_option("--file, -f", fileName, "Input data fileName. Has to be either .txt of .vti.")->required();

    string fieldName = "var";
    cliApp.add_option("--isovalue, -i", fieldName, "The name of the isovalue field in the input data file. For txt files, the name does not matter.");

    string decompositionType = "height";
    cliApp.add_option("--decompositionType", decompositionType, "What type of branch decomposition to use - height (parallel), volume (parallel) or persistence (serial).")->required();

    //bool volumeBD = false;
    //cliApp.add_flag("--volume, -l", volumeBD, "Use volume branch decomposition instead of height.");

    bool isData2D = false;
    cliApp.add_flag("--flatData", isData2D, "Whether the input file is text and 2D.");

    int simplificationThreshold = 10;
    cliApp.add_option("--threshold, -t", simplificationThreshold, "The number of connected components to display. Default is top 10 sorted by height/volume.");

    string outputFilename = "";
    cliApp.add_option("--output, -o", outputFilename, "Save output to a vtm (MultiBlockData) file instead of using the in house engine to visualise it.");

    bool performanceTestOnly = false;
    cliApp.add_flag("--performance, -p", performanceTestOnly, "Compute the branch decomposition and quit.");

    string cinemaOutputFilename = "";
    cliApp.add_option("--cinemadb, -c", cinemaOutputFilename, "Name of the cinema db output fileName.");

    Float64 isovalue = 0.0;
    cliApp.add_option("--isovalueValue", isovalue, "Used to visualise a single isosurface. Debuging feature.");

    string selectionType = "sort";
    cliApp.add_option("--selectionType", selectionType, "Used to select the feature selection method sort or root out");

    string branchIsovalue = "epsilon";
    cliApp.add_option("--branchIsovalue", branchIsovalue, "Used to select where on the branch we take the features either half way or epsilon away from the root.");

    string outputCTFilename = "";
    cliApp.add_option("--outputCT", outputCTFilename, "Compute and store the contour tree, nothing else.");

    string inputCTFilename = "";
    cliApp.add_option("--inputCT", inputCTFilename, "Read in contour tree instead of computing it.");

    //int numThreads = tbb::task_scheduler_init::default_num_threads();
    int numThreads = 1;
    cliApp.add_option("--numThreads", numThreads, "Tells TBB how many threads to use.");

    vtkm::Id meshIdA = 0;
    cliApp.add_option("--meshIdA", meshIdA, "Tells TBB how many threads to use.");

    vtkm::Id meshIdB = 0;
    cliApp.add_option("--meshIdB", meshIdB, "Tells TBB how many threads to use.");

    CLI11_PARSE(cliApp, argc, argv);

//    // Read in Data File
//    vtkm::cont::DataSet inputData;
//    if (std::string::npos != fileName.find(".vtk"))
//    {
//        try
//        {
//            vtkm::io::VTKDataSetReader reader(fileName);
//            inputData = reader.ReadDataSet();
//        }
//        catch(string message)
//        {
//            cerr << message;
//            return 1;
//        }
//    }
//    else if (std::string::npos != fileName.find(".bov"))
//    {
//            vtkm::io::BOVDataSetReader reader(fileName);
//            inputData = reader.ReadDataSet();
//            //return 1;
//    }
//    else if (std::string::npos != fileName.find(".txt") && true == isData2D)
//    {
//        try
//        {
//            inputData = read2DUniformGridData(fileName, fieldName);
//            // The application is not set up to visualise 2D contours, to avoid rendering them we set this to 0
//            simplificationThreshold = 0;
//        }
//        catch(string message)
//        {
//            cerr << message;
//            return 1;
//        }
//    }
//    else
//    {
//        cerr << "Please provide a vti file or a txt file in the format xdim ydim zdim <data values>.";
//        return 1;
//    }

    ///////////////////////////////////////////////
    // Read the input data
    ///////////////////////////////////////////////

    std::vector<vtkm::Float32>::size_type nDims = 0;
    vtkm::cont::DataSet inputData;
    std::vector<vtkm::Float64> values;
    std::vector<vtkm::Id> dims;


    // 2025-03-18 Added ContourTreeMesh
  //  vtkm::worklet::contourtree_augmented::ContourTreeMesh<ValueType> contourTreeMesh;
  //  contourTreeMesh.Load(fileName.c_str()); // this breaks

    if (fileName.compare(fileName.length() - 3, 3, "bov") == 0)
    {
      vtkm::io::BOVDataSetReader reader(fileName);
      inputData = reader.ReadDataSet();
      nDims = 3;
    }
    else // Read ASCII data input
    {
      std::cout << "Reading file: " << fileName << "\n";
      std::ifstream inFile(fileName);
      if (inFile.bad())
        return 0;

      // Read the dimensions of the mesh, i.e,. number of elementes in x, y, and z
      std::string line;
      getline(inFile, line);
      std::istringstream linestream(line);
      vtkm::Id dimVertices;
      while (linestream >> dimVertices)
      {
        dims.push_back(dimVertices);
      }

      // Compute the number of vertices, i.e., xdim * ydim * zdim
      nDims = static_cast<unsigned short>(dims.size());
      std::size_t numVertices = static_cast<std::size_t>(
        std::accumulate(dims.begin(), dims.end(), std::size_t(1), std::multiplies<std::size_t>()));

      // Check for fatal input errors
      // Check the the number of dimension is either 2D or 3D
      bool invalidNumDimensions = (nDims < 2 || nDims > 3);
      // Log any errors if found on rank 0

      // If we found any errors in the setttings than finalize MPI and exit the execution
      if (invalidNumDimensions)
      {
        std::cerr << "The input mesh is " << nDims << "D. The input data must be either 2D or 3D.";
        return EXIT_SUCCESS;
      }

      // Read data
      values.resize(numVertices);
      for (std::size_t vertex = 0; vertex < numVertices; ++vertex)
      {
        inFile >> values[vertex];
      }

      // finish reading the data
      inFile.close();

      // swap dims order
      std::swap(dims[0], dims[1]);

      // build the input dataset
      vtkm::cont::DataSetBuilderUniform dsb;
      // 2D data
      if (nDims == 2)
      {
        vtkm::Id2 vdims;
        vdims[0] = static_cast<vtkm::Id>(dims[0]);
        vdims[1] = static_cast<vtkm::Id>(dims[1]);
        inputData = dsb.Create(vdims);
      }
      // 3D data
      else
      {
        vtkm::Id3 vdims;
        vdims[0] = static_cast<vtkm::Id>(dims[0]);
        vdims[1] = static_cast<vtkm::Id>(dims[1]);
        vdims[2] = static_cast<vtkm::Id>(dims[2]);
        inputData = dsb.Create(vdims);
      }

//      inputData.AddPointField("values", values);
      inputData.AddPointField(fieldName, values);
    } // END ASCII Read


    fieldName = inputData.GetPointField(fieldName).GetName();


    // 
    // Only callled to write the ct to file
    //
    if (false == outputCTFilename.empty())
    {
        
        worklet::contourtree_augmented::ContourTree ct2;
        vtkm::cont::ArrayHandle<vtkm::Id> ctSortOrder2;
        vtkm::cont::ArrayHandle<vtkm::Id> ctSortIndices;
        Id ctNumIterations2;

        tie(ct2, ctSortOrder2, ctSortIndices, ctNumIterations2) = cv1k::ct::getContourTree(inputData, fieldName);

        cv1k::ct::writeContourTree(ct2, ctSortOrder2, ctNumIterations2, outputCTFilename);

        return 0;
    }

    std::cout << "ComputeMostSignificantContours" << std::endl;
    vtkm::cont::PartitionedDataSet outputDataSets = cv1k::interface::computeMostSignificantContours(inputData,
                                                                                                    fieldName,
                                                                                                    inputCTFilename,
                                                                                                    simplificationThreshold,
                                                                                                    decompositionType,
                                                                                                    selectionType,
                                                                                                    branchIsovalue,
                                                                                                    performanceTestOnly);

    if (false == outputFilename.empty())
    {
        for (size_t i = 0 ; i < outputDataSets.GetNumberOfPartitions() ; i++)
        {

            //
            // Save Contours Directly
            //

            // Get some metadata for the file name:
            vtkm::Range branchIDrange[1];
            outputDataSets.GetPartition(i).GetPointField("branchIDpt").GetRange(branchIDrange);
            // WARNING: This assumes everything has gone well ...
            // ... and each cell (triangle) has the SAME branch ID assigned
            // (this is the expected outcome, therefore, we can take the range center ...
            //  ... since each point must be assigned the same branch ID)
            // Note: vtkm::Range gives [min,max] range of the array
            vtkm::Id contourBranchId = (vtkm::Id)branchIDrange[0].Center();
            // we repeat the same for the isovalues:
            vtkm::Range isovaluerange[1];
            outputDataSets.GetPartition(i).GetPointField("isovaluePoints").GetRange(isovaluerange);
            vtkm::Float64 contourIsovalue = isovaluerange[0].Center();

            string currentFilename = outputFilename;

            currentFilename.append("contour-");
            currentFilename.append(std::to_string(i));
            currentFilename.append("-br");
            currentFilename.append(std::to_string(contourBranchId));
            currentFilename.append("-iso-");
            currentFilename.append(std::to_string(contourIsovalue));
            currentFilename.append(".vtk");

            cout << "The output filename is " << currentFilename << endl;

            vtkm::io::VTKDataSetWriter writer(currentFilename);
            writer.WriteDataSet(outputDataSets.GetPartition(i));

            //
            // Save Images (Cinema Database)
            //
            //            auto currentDataSet = outputDataSets.GetPartition(i);
            //            vtkm::Bounds bounds = currentDataSet.GetCoordinateSystem().GetBounds();

            //            CinemaExporter::RenderAndWrite(
            //                    i, // int contourID
            //                    currentDataSet, // vtkm::cont::DataSet& dataSet
            //                    bounds, // vtkm::Bounds& bounds
            //                    "./output/",
            //                    100, 100, // int resX, int resY,
            //                    0, 270, 90, // int elevation, int elevation1, int elevationStep,
            //                    -60, 60, 60 // int azimuth, int azimuth1, int azimuthStep
            //                    );
        }
    }

    return 0;
}

