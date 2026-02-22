//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
// Copyright (c) 2018, The Regents of the University of California, through
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
//
//=============================================================================
//
//  This code is an extension of the algorithm presented in the paper:
//  Parallel Peak Pruning for Scalable SMP Contour Tree Computation.
//  Hamish Carr, Gunther Weber, Christopher Sewell, and James Ahrens.
//  Proceedings of the IEEE Symposium on Large Data Analysis and Visualization
//  (LDAV), October 2016, Baltimore, Maryland.
//
//  The PPP2 algorithm and software were jointly developed by
//  Hamish Carr (University of Leeds), Gunther H. Weber (LBNL), and
//  Oliver Ruebel (LBNL)
//==============================================================================


#ifndef vtk_m_worklet_contourtree_augmented_process_contourtree_h
#define vtk_m_worklet_contourtree_augmented_process_contourtree_h

// global includes
#include <algorithm>
#include <iomanip>
#include <iostream>

// local includes
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/PrintVectors.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>

//VTKM includes
#include <vtkm/Pair.h>
#include <vtkm/Types.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/ArrayHandleView.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ArrayTransforms.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ContourTree.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/PrintVectors.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/Branch.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/SuperArcVolumetricComparator.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/SuperNodeBranchComparator.h>

#include <vtkm/cont/Invoker.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/HypersweepWorklets.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/PointerDoubling.h>

// for sleeping
#include <chrono>
#include <thread>

// for memory usage
#include <sys/resource.h>
#include <unistd.h>

#define DEBUG_PRINT_PACTBD 0
#define WRITE_FILES 1


namespace process_contourtree_inc_ns =
  vtkm::worklet::contourtree_augmented::process_contourtree_inc;
  
//using ValueType = vtkm::Float64; //vtkm::FloatDefault;
using ValueType = vtkm::Float32; //vtkm::FloatDefault;
//using ValueType = vtkm::FloatDefault;
using FloatArrayType = vtkm::cont::ArrayHandle<ValueType>;

namespace vtkm
{
namespace worklet
{
namespace contourtree_augmented
{

struct BettiCoefficients
{
    long num_vtx;
    long num_edg;
    long num_fac;
    long num_tet;

    long betti0;
    long betti1;
    long betti3;
};

struct Coefficients
{
    long double h1;
    long double h2;
    long double h3;
    long double h4;
};


// TODO Many of the post processing routines still need to be parallelized
// Class with routines for post processing the contour tree
class ProcessContourTree
{ // class ProcessContourTree
public:
  // initialises contour tree arrays - rest is done by another class
  ProcessContourTree()
  { // ProcessContourTree()
  } // ProcessContourTree()

  // collect the sorted arcs
  void static CollectSortedArcs(const ContourTree& contourTree,
                                const IdArrayType& sortOrder,
                                EdgePairArray& sortedArcs)
  { // CollectSortedArcs
    // create an array for sorting the arcs
    std::vector<EdgePair> arcSorter;

    // fill it up
    auto arcsPortal = contourTree.Arcs.ReadPortal();
    auto sortOrderPortal = sortOrder.ReadPortal();

    for (vtkm::Id node = 0; node < contourTree.Arcs.GetNumberOfValues(); node++)
    { // per node
      // retrieve ID of target supernode
      vtkm::Id arcTo = arcsPortal.Get(node);

      // if this is true, it is the last pruned vertex & is omitted
      if (NoSuchElement(arcTo))
        continue;

      // otherwise, strip out the flags
      arcTo = MaskedIndex(arcTo);

      // now convert to mesh IDs from sort IDs
      // otherwise, we need to convert the IDs to regular mesh IDs
      vtkm::Id regularID = sortOrderPortal.Get(node);

      // retrieve the regular ID for it
      vtkm::Id regularTo = sortOrderPortal.Get(arcTo);

      // how we print depends on which end has lower ID
      if (regularID < regularTo)
        arcSorter.push_back(EdgePair(regularID, regularTo));
      else
        arcSorter.push_back(EdgePair(regularTo, regularID));
    } // per vertex

    // now sort it
    // Setting saddlePeak reference to the make_ArrayHandle directly does not work
    sortedArcs = vtkm::cont::make_ArrayHandle(arcSorter, vtkm::CopyFlag::On);
    vtkm::cont::Algorithm::Sort(sortedArcs, SaddlePeakSort());
  } // CollectSortedArcs

  // collect the sorted superarcs
  void static CollectSortedSuperarcs(const ContourTree& contourTree,
                                     const IdArrayType& sortOrder,
                                     EdgePairArray& saddlePeak)
  { // CollectSortedSuperarcs()
    // create an array for sorting the arcs
    std::vector<EdgePair> superarcSorter;

    // fill it up
    auto supernodesPortal = contourTree.Supernodes.ReadPortal();
    auto superarcsPortal = contourTree.Superarcs.ReadPortal();
    auto sortOrderPortal = sortOrder.ReadPortal();

    for (vtkm::Id supernode = 0; supernode < contourTree.Supernodes.GetNumberOfValues();
         supernode++)
    { // per supernode
      // sort ID of the supernode
      vtkm::Id sortID = supernodesPortal.Get(supernode);

      // retrieve ID of target supernode
      vtkm::Id superTo = superarcsPortal.Get(supernode);

      // if this is true, it is the last pruned vertex & is omitted
      if (NoSuchElement(superTo))
        continue;

      // otherwise, strip out the flags
      superTo = MaskedIndex(superTo);

      // otherwise, we need to convert the IDs to regular mesh IDs
      vtkm::Id regularID = sortOrderPortal.Get(MaskedIndex(sortID));

      // retrieve the regular ID for it
      vtkm::Id regularTo = sortOrderPortal.Get(MaskedIndex(supernodesPortal.Get(superTo)));

      // how we print depends on which end has lower ID
      if (regularID < regularTo)
      { // from is lower
        // extra test to catch duplicate edge
        if (superarcsPortal.Get(superTo) != supernode)
        {
          superarcSorter.push_back(EdgePair(regularID, regularTo));
        }
      } // from is lower
      else
      {
        superarcSorter.push_back(EdgePair(regularTo, regularID));
      }
    } // per vertex

    // Setting saddlePeak reference to the make_ArrayHandle directly does not work
    saddlePeak = vtkm::cont::make_ArrayHandle(superarcSorter, vtkm::CopyFlag::On);

    // now sort it
    vtkm::cont::Algorithm::Sort(saddlePeak, SaddlePeakSort());
  } // CollectSortedSuperarcs()
































	
   // 2026-02-13 refit Betti computation to Marching Cubes meshes
   void static printMemoryUsage(const std::string& message)
    {
        // Red text formatting for highlighting some console output:
        const std::string ORANGE = "\033[38;2;255;165;0m";  // Start red text
        const std::string LIGHT_BLUE = "\033[38;5;117m";  // Light blue in 256-color
        const std::string RESET = "\033[0m"; // End red text

        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);

        std::ifstream status_file("/proc/self/status");
        std::string line;

        size_t current_usage = 0;

        while(std::getline(status_file, line))
        {
            if(line.find("VmRSS:") == 0)
            {
                std::istringstream iss(line);
                std::string key;
                size_t memory; // memory value in kB
                std::string unit;

                iss >> key >> memory >> unit;
                current_usage = memory; // return in KB
            }
        }

//        std::cout << ORANGE << message << LIGHT_BLUE << " - Memory usage (peak): " << usage.ru_maxrss
//                  << " KB | (current) " << current_usage << " KB" << RESET << std::endl;

        std::cout << LIGHT_BLUE << message << " - Memory usage (peak): " << usage.ru_maxrss
                  << " KB | (current) " << current_usage << " KB" << RESET << std::endl;

    }

    // A simple edge struct for clarity
    struct Edge {
        vtkm::Id v0, v1;
    };
    
    struct TriangleFace {
        vtkm::Id v0, v1, v2;
        bool boundary;
    };
    
    
    
    struct ConnectivityOutput
{
    std::vector<std::array<vtkm::Id, 2>> edges;
    std::vector<vtkm::UInt8> edgeBoundary;

    std::vector<std::array<vtkm::Id, 4>> faces;
    std::vector<vtkm::UInt8> faceBoundary;

    std::vector<std::array<vtkm::Id, 8>> cubes;
    std::vector<vtkm::UInt8> cubeBoundary;
};

ConnectivityOutput static ExtractStructuredConnectivity(
	const std::vector<vtkm::Id>& sortID,
    const vtkm::cont::DataSet& input)
{
    ConnectivityOutput result;

    // --- Get structured cell set ---
    auto cellSet = input.GetCellSet();

    if (!cellSet.IsType<vtkm::cont::CellSetStructured<3>>())
    {
        throw std::runtime_error("Input must be 3D structured grid.");
    }

    auto structured =
        cellSet.AsCellSet<vtkm::cont::CellSetStructured<3>>();

    vtkm::Id3 dims = structured.GetPointDimensions();
    vtkm::Id nx = dims[0];
    vtkm::Id ny = dims[1];
    vtkm::Id nz = dims[2];

    // --- Vertex ID helper ---
    auto vid = [&](vtkm::Id i,
                   vtkm::Id j,
                   vtkm::Id k) -> vtkm::Id
    {
        return i + j*nx + k*nx*ny;
    };

    // ==========================================================
    // EDGES
    // ==========================================================
    for (vtkm::Id k = 0; k < nz; ++k)
    {
        for (vtkm::Id j = 0; j < ny; ++j)
        {
            for (vtkm::Id i = 0; i < nx; ++i)
            {
                vtkm::Id v0 = vid(i,j,k);

                // X edge
                if (i + 1 < nx)
                {
                    vtkm::Id v1 = vid(i+1,j,k);
                    //result.edges.push_back({v0,v1});
					if (sortID[v0] < sortID[v1]) result.edges.push_back({sortID[v0],sortID[v1]});   
					else result.edges.push_back({sortID[v1],sortID[v0]});
                    

                    bool boundary =
                        (j == 0 || j == ny-1 ||
                         k == 0 || k == nz-1);

                    result.edgeBoundary.push_back(boundary);
                }

                // Y edge
                if (j + 1 < ny)
                {
                    vtkm::Id v1 = vid(i,j+1,k);
                    //result.edges.push_back({v0,v1});
					if (sortID[v0] < sortID[v1]) result.edges.push_back({sortID[v0],sortID[v1]});   
					else result.edges.push_back({sortID[v1],sortID[v0]});

                    bool boundary =
                        (i == 0 || i == nx-1 ||
                         k == 0 || k == nz-1);

                    result.edgeBoundary.push_back(boundary);
                }

                // Z edge
                if (k + 1 < nz)
                {
                    vtkm::Id v1 = vid(i,j,k+1);
                    //result.edges.push_back({v0,v1});
					if (sortID[v0] < sortID[v1]) result.edges.push_back({sortID[v0],sortID[v1]});   
					else result.edges.push_back({sortID[v1],sortID[v0]});


                    bool boundary =
                        (i == 0 || i == nx-1 ||
                         j == 0 || j == ny-1);

                    result.edgeBoundary.push_back(boundary);
                }
            }
        }
    }

    // ==========================================================
    // FACES (quads)
    // ==========================================================
    for (vtkm::Id k = 0; k < nz; ++k)
    {
        for (vtkm::Id j = 0; j < ny-1; ++j)
        {
            for (vtkm::Id i = 0; i < nx-1; ++i)
            {
                // XY face
                vtkm::Id v0 = sortID[vid(i,  j,  k)];
                vtkm::Id v1 = sortID[vid(i+1,j,  k)];
                vtkm::Id v2 = sortID[vid(i,  j+1,k)];
                vtkm::Id v3 = sortID[vid(i+1,j+1,k)];
                
				// Collect all vertices
				std::array<vtkm::Id,4> vxs = {v0, v1, v2, v3};

				// Sort array
				std::sort(vxs.begin(), vxs.end());

				// Push into faces
				result.faces.push_back(vxs);

                bool boundary =
                    (k == 0 || k == nz-1);

                result.faceBoundary.push_back(boundary);
            }
        }
    }

    // XZ faces
    for (vtkm::Id k = 0; k < nz-1; ++k)
    {
        for (vtkm::Id j = 0; j < ny; ++j)
        {
            for (vtkm::Id i = 0; i < nx-1; ++i)
            {
                vtkm::Id v0 = sortID[vid(i,  j,  k)];
                vtkm::Id v1 = sortID[vid(i+1,j,  k)];
                vtkm::Id v2 = sortID[vid(i,  j,  k+1)];
                vtkm::Id v3 = sortID[vid(i+1,j,  k+1)];

                //result.faces.push_back({v0,v1,v2,v3});
                
				// Collect all vertices
				std::array<vtkm::Id,4> vxs = {v0, v1, v2, v3};

				// Sort array
				std::sort(vxs.begin(), vxs.end());

				// Push into faces
				result.faces.push_back(vxs);

                bool boundary =
                    (j == 0 || j == ny-1);

                result.faceBoundary.push_back(boundary);
            }
        }
    }

    // YZ faces
    for (vtkm::Id k = 0; k < nz-1; ++k)
    {
        for (vtkm::Id j = 0; j < ny-1; ++j)
        {
            for (vtkm::Id i = 0; i < nx; ++i)
            {
                vtkm::Id v0 = sortID[vid(i,  j,  k)];
                vtkm::Id v1 = sortID[vid(i,  j+1,k)];
                vtkm::Id v2 = sortID[vid(i,  j,  k+1)];
                vtkm::Id v3 = sortID[vid(i,  j+1,k+1)];

                //result.faces.push_back({v0,v1,v2,v3});
                
				// Collect all vertices
				std::array<vtkm::Id,4> vxs = {v0, v1, v2, v3};

				// Sort array
				std::sort(vxs.begin(), vxs.end());

				// Push into faces
				result.faces.push_back(vxs);

                bool boundary =
                    (i == 0 || i == nx-1);

                result.faceBoundary.push_back(boundary);
            }
        }
    }

    // ==========================================================
    // CUBES
    // ==========================================================
    for (vtkm::Id k = 0; k < nz-1; ++k)
    {
        for (vtkm::Id j = 0; j < ny-1; ++j)
        {
            for (vtkm::Id i = 0; i < nx-1; ++i)
            {
                vtkm::Id v0 = sortID[vid(i,  j,  k)  ];
                vtkm::Id v1 = sortID[vid(i+1,j,  k)  ];
                vtkm::Id v2 = sortID[vid(i,  j+1,k)  ];
                vtkm::Id v3 = sortID[vid(i+1,j+1,k)  ];
                vtkm::Id v4 = sortID[vid(i,  j,  k+1)];
                vtkm::Id v5 = sortID[vid(i+1,j,  k+1)];
                vtkm::Id v6 = sortID[vid(i,  j+1,k+1)];
                vtkm::Id v7 = sortID[vid(i+1,j+1,k+1)];

                //result.cubes.push_back(
                    //{v0,v1,v2,v3,v4,v5,v6,v7});
                    
				std::array<vtkm::Id,8> vxs = {v0,v1,v2,v3,v4,v5,v6,v7};

				// Sort array
				std::sort(vxs.begin(), vxs.end());

				// Push into faces
				result.cubes.push_back(vxs);

                bool boundary =
                    (i == 0 || j == 0 || k == 0 ||
                     i == nx-2 ||
                     j == ny-2 ||
                     k == nz-2);

                result.cubeBoundary.push_back(boundary);
            }
        }
    }

    return result;
}

void static PrintConnectivity(const ConnectivityOutput& c)
{
    std::cout << "\n=== EDGES ===\n";
    for (std::size_t i = 0; i < c.edges.size(); ++i)
    {
        std::cout << i << "\t"
                  << c.edges[i][0] << "\t"
                  << c.edges[i][1]
                  << "\t" << int(c.edgeBoundary[i])
                  << "\n";
    }

    std::cout << "\n=== FACES ===\n";
    for (std::size_t i = 0; i < c.faces.size(); ++i)
    {
        std::cout << i << "\t"
                  << c.faces[i][0] << "\t"
                  << c.faces[i][1] << "\t"
                  << c.faces[i][2] << "\t"
                  << c.faces[i][3]
                  << "\t" << int(c.faceBoundary[i])
                  << "\n";
    }

    std::cout << "\n=== CUBES ===\n";
    for (std::size_t i = 0; i < c.cubes.size(); ++i)
    {
        std::cout << i << "\t";
        for (int v = 0; v < 8; ++v)
            std::cout << c.cubes[i][v] << "\t";

        std::cout << "\t"
                  << int(c.cubeBoundary[i]) << "\n";
    }
}



	// 2026-02-13 Betti computation with LU Stars (2004 Pascucci Parallel)
    void static LUstars(// INPUTS
                        int numVertices, // in sort order, enough to have the total number, since we start from 0 incrementing by 1 up to N
                        std::vector<Edge>&              edges,
                        std::vector<TriangleFace>&      triangles,
                        std::vector<std::vector<int>>&  tetrahedra,
                        // OUTPUTS
                        std::vector<int>& lowerStars,
                        std::vector<int>& upperStars,
                        std::vector<int>& deltaBoundary)
    {
        // 1. initialise LU, US, dB:
        lowerStars.resize(numVertices, 1);
        upperStars.resize(numVertices, 1);
        deltaBoundary.resize(numVertices, 0);

#if DEBUG_PRINT_PACTBD
        std::cout << "LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

        // 2. for each edge:
        int i,j;
        for(int it = 0; it < edges.size(); it++)
        {
            i = edges[it].v0;
            j = edges[it].v1;
            if(i < j)
            {
                lowerStars[j]--;
                upperStars[i]--;
            }
        }

#if DEBUG_PRINT_PACTBD
        std::cout << "(Edge)LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

        // 3. for each edge:
        i=0;
        j=0;
        int k;
        bool b;
        for(int it = 0; it < triangles.size(); it++)
        {
            i = triangles[it].v0;
            j = triangles[it].v1;
            k = triangles[it].v2;
            b = triangles[it].boundary;
            if ((i < j) && (j < k))
            {
                lowerStars[k]++;
                upperStars[i]++;
                if (b)
                {
                    deltaBoundary[k]--;
                    deltaBoundary[i]++;
                }
            }
        }

#if DEBUG_PRINT_PACTBD
        std::cout << "(Triangle)LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

        // 3. for each tetrahedron:
        i=0;
        j=0;
        k=0;
        int l;
        for(int it = 0; it < tetrahedra.size(); it++)
        {
            i = tetrahedra[it][0];
            j = tetrahedra[it][1];
            k = tetrahedra[it][2];
            l = tetrahedra[it][3];

            if ((i < j) && (j < k) && (k < l))
            {
                lowerStars[l]--;
                upperStars[i]--;
            }
        }

#if DEBUG_PRINT_PACTBD
        std::cout << "(Tet)LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

    }
    
    
    
    
void static LUstarsBettiMC(// INPUTS
				int numVertices, // in sort order, enough to have the total number, since we start from 0 incrementing by 1 up to N
				ConnectivityOutput& conn,
				// OUTPUTS
				std::vector<int>& lowerStars,
				std::vector<int>& upperStars,
				std::vector<int>& deltaBoundary)
    {
        // 1. initialise LU, US, dB:
        lowerStars.resize(numVertices, 1);
        upperStars.resize(numVertices, 1);
        deltaBoundary.resize(numVertices, 0);

#if DEBUG_PRINT_PACTBD
        std::cout << "LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

        // 2. for each edge:
        int i,j;
        for(int it = 0; it < conn.edges.size(); it++)
        {
            i = conn.edges[it][0]; //.v0;
            j = conn.edges[it][1]; //.v1;
            if(i < j)
            {
                lowerStars[j]--;
                upperStars[i]--;
            }
        }

#if DEBUG_PRINT_PACTBD
        std::cout << "(Edge)LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

        // 3. for each face:
        i=0;
        j=0;
        int k;
        int l;
        bool b;
        for(int it = 0; it < conn.faces.size(); it++)
        {
            i = conn.faces[it][0]; //.v0;
            j = conn.faces[it][1]; //.v1;
            k = conn.faces[it][2]; //.v2;
            l = conn.faces[it][3];
            b = conn.faceBoundary[it];
            //if ((i < j) && (j < k))
            if ((i < j) && (j < k) && (k < l))
            {
                lowerStars[l]++;
                upperStars[i]++;
                if (b)
                {
                    deltaBoundary[l]--;
                    deltaBoundary[i]++;
                }
            }
        }

#if DEBUG_PRINT_PACTBD
        std::cout << "(Triangle)LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

        // 3. for each cell:
        i=0;
        j=0;
        k=0;
        l=0;
        int m;
        int n;
        int o;
        int p;
        for(int it = 0; it < conn.cubes.size(); it++)
        {
            i = conn.cubes[it][0];
            j = conn.cubes[it][1];
            k = conn.cubes[it][2];
            l = conn.cubes[it][3];
            m = conn.cubes[it][4];
            n = conn.cubes[it][5];
            o = conn.cubes[it][6];
            p = conn.cubes[it][7];

            if ((i < j) && (j < k) && (k < l) && (l < m) && (m < n) && (n < o) && (o < p))
            {
                lowerStars[p]--;
                upperStars[i]--;
            }
        }

#if DEBUG_PRINT_PACTBD
        std::cout << "(Tet)LU\tUS\tdB:" << std::endl;
        for(int i = 0; i < numVertices; i++)
        {
            std::cout << i << "\t" << lowerStars[i] << "\t" << upperStars[i] << "\t" << deltaBoundary[i] << std::endl;
        }
#endif

    }


std::vector<vtkm::Id> static ComputeSortIDStdVector(const vtkm::cont::DataSet& input)
{
	std::vector<std::string> sortOrder = {"values", "z", "y", "x"};
	//std::vector<std::string> sortOrder = {"values", "x", "y", "z"};
	
    // Extract the "values" field
    auto field = input.GetField("values");
    if (field.GetAssociation() != vtkm::cont::Field::Association::Points)
        throw std::runtime_error("'values' must be point-associated.");

    vtkm::Id n = field.GetData().GetNumberOfValues();
    std::vector<vtkm::Id> indices(n);
    for (vtkm::Id i = 0; i < n; ++i) indices[i] = i;

    auto coordData = input.GetCoordinateSystem().GetData();

    // Cast the values field
    vtkm::cont::CastAndCall(field.GetData(), [&](const auto& values) {
        auto valPortal = values.ReadPortal();

        // Cast the coordinate system
        vtkm::cont::CastAndCall(coordData, [&](const auto& points) {
            auto ptPortal = points.ReadPortal();

            // Generic comparator
            auto comparator = [&](vtkm::Id a, vtkm::Id b) {
                for (const auto& key : sortOrder)
                {
                    if (key == "values")
                    {
                        auto va = valPortal.Get(a);
                        auto vb = valPortal.Get(b);
                        if (va != vb) return va < vb;
                    }
                    else if (key == "x")
                    {
                        auto pa = ptPortal.Get(a);
                        auto pb = ptPortal.Get(b);
                        if (pa[0] != pb[0]) return pa[0] < pb[0];
                    }
                    else if (key == "y")
                    {
                        auto pa = ptPortal.Get(a);
                        auto pb = ptPortal.Get(b);
                        if (pa[1] != pb[1]) return pa[1] < pb[1];
                    }
                    else if (key == "z")
                    {
                        auto pa = ptPortal.Get(a);
                        auto pb = ptPortal.Get(b);
                        if (pa[2] != pb[2]) return pa[2] < pb[2];
                    }
                    else
                    {
                        throw std::runtime_error("Unknown sort key: " + key);
                    }
                }
                return false; // equal in all keys
            };

            // Sort indices using the generic comparator
            std::sort(indices.begin(), indices.end(), comparator);
        });
    });

    // Build sortID: oldID -> new sorted position
    std::vector<vtkm::Id> sortID(n);
    for (vtkm::Id newPos = 0; newPos < n; ++newPos)
        sortID[indices[newPos]] = newPos;

    return sortID;
}
   
   
   // 2025-10-11 COMPUTE BETTI NUMBERS FOR EACH REGULAR BRANCH
   // BASED ON 2004 Pascucci Parallel Computation of the Topology of Level Sets
    void static ComputeBettiNumbersForRegularArcs(const vtkm::cont::DataSet& input, // the coefficient-based version additionally requires tetrahedral connections and vertex coordinates
                                                  const ContourTree& contourTree,
//                                                  ContourTree& contourTree, // modify the contour tree betti numbers, no longer const
                                                                            // (but only used for explicitly modifying contourTree.SupernodeBetti)
                                                  const vtkm::Id nIterations,
                                                  vtkm::cont::ArrayHandle<Coefficients>& superarcIntrinsicWeightCoeff, // (output)
                                                  vtkm::cont::ArrayHandle<Coefficients>& superarcDependentWeightCoeff, // (output)
                                                  vtkm::cont::ArrayHandle<Coefficients>& supernodeTransferWeightCoeff, // (output)
                                                  vtkm::cont::ArrayHandle<Coefficients>& hyperarcDependentWeightCoeff, // (output)
                                                  // Added 2025-01-30
                                                  // We use simple weights for the branch decomposition
                                                  FloatArrayType& superarcIntrinsicWeight, // (output)
                                                  FloatArrayType& superarcDependentWeight, // (output)
                                                  FloatArrayType& supernodeTransferWeight, // (output)
                                                  FloatArrayType& hyperarcDependentWeight) // (output))
    { // ComputeBettiNumbersForRegularArcs()
        std::cout << "[ProcessContourTree.h::ComputeBettiNumbersForRegularArcs] Compute Betti Numbers for each Regular Arc" << std::endl;
        printMemoryUsage("[ProcessContourTree.h::ComputeVolumeWeightsSerialStructCoefficients] Checkpoint 1/4 - START");

        using TetCellSet = vtkm::cont::CellSetSingleType<>;
        const auto& unknown = input.GetCellSet();
        
        std::vector<vtkm::Id> sortIDLookup = ComputeSortIDStdVector(input);
        
        ConnectivityOutput cubeConnectivity = ExtractStructuredConnectivity(sortIDLookup, input);
        
        //PrintConnectivity(cubeConnectivity);
        
        vtkm::cont::ArrayHandle<ValueType> dataField;
	    input.GetField("values").GetData().AsArrayHandle(dataField);
	    //auto dataField1Portal = dataField.ReadPortal();
	    auto dataPortal = dataField.ReadPortal();
	    
	    //std::cout << "@@@@@@@@@@@@@@@@@@@@@" << std::endl;
	    ////for(int i = 0; i < dataField1Portal.GetNumberOfValues(); i++)
	    //for(int i = 0; i < dataPortal.GetNumberOfValues(); i++)
	    //{
			////std::cout << i << " " << dataField1Portal.Get(i) << std::endl;
			//std::cout << i << " " << dataPortal.Get(i) << std::endl;
		//}
		
		
		
		////auto sortIDLookup = ComputeSortIDStdVector(input);
		//std::cout << "@@@@@@@@@@@@@@@@@@@@@" << std::endl;
		//for (vtkm::Id i = 0; i < sortIDLookup.size(); ++i)
		//{
			//std::cout << i 
					  //<< "\t" 
					  //<< sortIDLookup[i] << "\n";
		//}

		//auto portalSort = sortIDLookup.ReadPortal();

		//for (vtkm::Id i = 0; i < portalSort.GetNumberOfValues(); ++i)
		//{
			//std::cout << "Point "
					  //<< i
					  //<< " sorted position = "
					  //<< portalSort.Get(i)
					  //<< "\n";
		//}


        std::vector<int> lowerStars;
        std::vector<int> upperStars;
        std::vector<int> deltaBoundary;
        
		std::cout << "Running LUstars ..." << std::endl;
		LUstarsBettiMC( contourTree.Arcs.GetNumberOfValues(),
						cubeConnectivity, 
						lowerStars,		// output
						upperStars,		// output
						deltaBoundary);	// output
        
        // ... 
        




















        std::cout << "Betti Number Regular Instrinsic Pre-Processing ... " << std::endl;

        auto arcsPortal = contourTree.Arcs.ReadPortal();
        auto nodesPortal = contourTree.Nodes.ReadPortal();
        auto superarcsPortal = contourTree.Superarcs.ReadPortal();
        auto supernodesPortal = contourTree.Supernodes.ReadPortal();
        auto superparentsPortal = contourTree.Superparents.ReadPortal();

        std::vector<vtkm::Id> chi_xij, chi_x;
        std::vector<vtkm::Id> be_ij,   bei;
        chi_xij.resize( contourTree.Arcs.GetNumberOfValues(), 0);
        chi_x.resize(   contourTree.Arcs.GetNumberOfValues(), 0);
        bei.resize(     contourTree.Arcs.GetNumberOfValues(), 0);
        be_ij.resize(   contourTree.Arcs.GetNumberOfValues(), 0);

        for (vtkm::Id sortedNode = 0; sortedNode < contourTree.Arcs.GetNumberOfValues(); sortedNode++)
        {// for each sortedNode
            vtkm::Id i_sortID = nodesPortal.Get(sortedNode); // regular grid implementation does not assume sorted
            //vtkm::Id i_sortID = sortIDLookup[nodesPortal.Get(sortedNode)];
            vtkm::Id i_superparent = superparentsPortal.Get(i_sortID);
            //vtkm::Id i_superparent = superparentsPortal.Get(nodesPortal.Get(sortedNode));

            //vtkm::Id j_sortID = nodesPortal.Get(sortedNode+1);
            //vtkm::Id j_superparent = superparentsPortal.Get(j_sortID );
            
			vtkm::Id j_sortID, j_superparent;
            
            if(sortedNode+1 < contourTree.Arcs.GetNumberOfValues())
            {
				j_sortID = nodesPortal.Get(sortedNode+1);
				//j_sortID = sortIDLookup[nodesPortal.Get(sortedNode+1)]; // regular grid implementation does not assume sorted
				j_superparent = superparentsPortal.Get(j_sortID );		
				//j_superparent = superparentsPortal.Get(nodesPortal.Get(sortedNode+1) );		
				
			}
            else
            {
				j_sortID = vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT; //177; //sortedNode+1);
				j_superparent = vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT; //superparentsPortal.Get(j_sortID );				
			}

            vtkm::Id tailend = supernodesPortal.Get(MaskedIndex(superarcsPortal.Get(i_superparent)));

            vtkm::Id delta = 1;

            if (i_superparent == j_superparent)
            {
                tailend = j_sortID;
            }



            if(i_sortID > tailend)
            {
                delta = -1;
            }

            chi_xij[sortedNode] = delta * (chi_x[i_sortID] - upperStars[i_sortID] + lowerStars[i_sortID]);
            be_ij[sortedNode]   = delta * (bei[i_sortID] + deltaBoundary[i_sortID]);

            chi_x[tailend] += delta * chi_xij[sortedNode];
            bei[tailend] += delta * be_ij[sortedNode];
            
#if DEBUG_PRINT_PACTBD
            std::cout << sortedNode << "\t" << i_sortID << "\t" << tailend << std::endl;
#endif
        }
        
        std::cout << "First loop finished" << std::endl;

        std::vector<vtkm::Id> regular_nodes_to_insert;
        std::vector<vtkm::Id> node_ascend;

        std::vector<vtkm::Id> nodes_to_relabel_superparent; // 2026-01-03 addition
        std::vector<vtkm::Id> nodes_to_relabel_hyperparent;
        std::vector<double> nodes_to_relabel_dataflip;
        std::vector<vtkm::Id> nodes_to_relabel_regularID;
        std::vector<vtkm::Id> nodes_to_relabel_regularID_betti_1;
        int previous_betti1 = 0;

#if DEBUG_PRINT_PACTBD
        std::cout << "VTK-m FIELDS:" << std::endl;
#endif

        // Loop over all fields
        //const vtkm::cont::Field& field = input.GetPointField("var"); // parcel fields = "var"
        //const vtkm::cont::Field& field = input.GetPointField("values");
        
        ////dataField

        //// Get the UnknownArrayHandle
        //vtkm::cont::UnknownArrayHandle ua = field.GetData();

        //// Cast to the correct array type:
        //// (replace float with the actual value type)
        //vtkm::cont::ArrayHandle<double> array;
        //ua.AsArrayHandle(array);

        // Get read-only access
        //auto dataPortal = array.ReadPortal();
        vtkm::Id n = dataPortal.GetNumberOfValues();

#if DEBUG_PRINT_PACTBD
        for (vtkm::Id i = 0; i < n; ++i)
        {
            std::cout << i << "\t" << dataPortal.Get(i) << std::endl;
        }
#endif


//        auto superparentsPortal = contourTree.Superparents.ReadPortal();
        auto hyperparentsPortal = contourTree.Hyperparents.ReadPortal();
        auto hypernodesPortal = contourTree.Hypernodes.ReadPortal();
        auto hyperarcsPortal = contourTree.Hyperarcs.ReadPortal();
//        auto superarcsPortal = contourTree.Superarcs.ReadPortal();
//        auto nodesPortal = contourTree.Nodes.ReadPortal();

#if WRITE_FILES

        std::ofstream fileeulerchi("ContourTreeEulerChi.txt");

        fileeulerchi << "Euler chi & delta border edges" << std::endl;
        fileeulerchi << std::setw(10) << "sortID->tailend" << "\t" << "chi_xij" << "\t" << "be_ij"
                              << "\t" << "betti0" << "\t" << "betti1" << "\t" << "betti2";//<< std::endl;
#endif

		std::cout << "regular nodecount: " << nodesPortal.GetNumberOfValues() << std::endl;
		std::cout << "regular arc-count: " << contourTree.Arcs.GetNumberOfValues() << std::endl;


        for (vtkm::Id sortedNode = 0; sortedNode < contourTree.Arcs.GetNumberOfValues(); sortedNode++)
        {
            vtkm::Id i_sortID = nodesPortal.Get(sortedNode); 
            //vtkm::Id i_sortID = sortIDLookup[nodesPortal.Get(sortedNode)]; // regular grid implementation does not assume sorted
            vtkm::Id i_superparent = superparentsPortal.Get(i_sortID);
            //vtkm::Id i_superparent = superparentsPortal.Get(nodesPortal.Get(sortedNode));
            
            vtkm::Id j_sortID, j_superparent;
            
            if(sortedNode+1 < contourTree.Arcs.GetNumberOfValues())
            {
				j_sortID = nodesPortal.Get(sortedNode+1);
				//j_sortID = sortIDLookup[nodesPortal.Get(sortedNode+1)]; // regular grid implementation does not assume sorted
				j_superparent = superparentsPortal.Get(j_sortID );		
				//j_superparent = superparentsPortal.Get(nodesPortal.Get(sortedNode+1) );		
			}
            else
            {
				j_sortID = vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT; //177; //sortedNode+1);
				j_superparent = vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT; //superparentsPortal.Get(j_sortID );				
			}
            
			// 2026-02-16
            //std::cout << sortedNode << " " << i_sortID << " -> " << sortedNode+1 << " " << j_sortID << std::endl;

            vtkm::Id tailend = supernodesPortal.Get(MaskedIndex(superarcsPortal.Get(i_superparent)));

            vtkm::Id delta = 1;

            if (i_superparent == j_superparent)
            {
                tailend = j_sortID;
            }

            vtkm::Id betti0 = 1;
            vtkm::Id betti1 = 1;
            vtkm::Id betti2 = 1;

            if(be_ij[sortedNode] > 0)
            {
                betti2 = 0; // no void, border detected
            }

            betti1 = betti0 + betti2 - chi_xij[sortedNode];

#if WRITE_FILES
            fileeulerchi << std::setw(10) << i_sortID << "->" << tailend << "\t" << chi_xij[sortedNode] << "\t" << be_ij[sortedNode]
                                  << "\t" << betti0 << "\t" << betti1 << "\t" << betti2;//<< std::endl;

            fileeulerchi << "\t\t" << i_sortID << "\t" << supernodesPortal.Get(i_superparent);
            fileeulerchi << "\t\t" << i_sortID << "\t" << supernodesPortal.Get(i_superparent) << std::endl;
#endif


            if((betti1 != previous_betti1) && (betti2 != 0))// 2026-01-31 added betti2 border tet check!
            //if((betti1 != previous_betti1) && (betti2 != 0) && (betti1 % 2 == 0))// 2026-01-31 added betti2 border tet check!
            //if((betti1 != previous_betti1) && (betti2 != 0) && (betti1 % 2 == 0) && (previous_betti1 % 2 == 0))// 2026-02-16 added betti1 previous non-1 check!
            //if((betti1 != previous_betti1) && (betti2 != 0) && (betti1 % 2 == 0))// 2026-02-16 added betti1 previous non-1 check!
            {
                regular_nodes_to_insert.push_back(i_sortID);

                nodes_to_relabel_superparent.push_back(i_superparent); // 2026-01-03 addition (original SPs of nodes to be upgraded to supernodes)
                nodes_to_relabel_hyperparent.push_back(hyperparentsPortal.Get(i_superparent)); // 2026-01-03 hyperparents failing when not matched
                nodes_to_relabel_regularID.push_back(i_sortID);

                // keep track of betti at a regular ID
//                nodes_to_relabel_regularID_betti_1.push_back(betti1);
                nodes_to_relabel_regularID_betti_1.push_back(previous_betti1);

                if(i_sortID > tailend)
                {
                    node_ascend.push_back(-1);
//                    nodes_to_relabel_dataflip.push_back(dataPortal.Get(i_sortID) * -1.0); //data value can be the same, regular ID won't
                        nodes_to_relabel_dataflip.push_back((double)i_sortID * -1.0);
                }
                else
                {
                    node_ascend.push_back(1);
//                    nodes_to_relabel_dataflip.push_back(dataPortal.Get(i_sortID) * 1.0);  //data value can be the same, regular ID won't
                        nodes_to_relabel_dataflip.push_back((double)i_sortID * 1.0);
                }

            }
            //else 
            else //if (betti1 % 2 == 0)
            {// if betti number didn't change, but dealing with a supernode ...
             // ... still need to relabel it
                if(i_sortID == supernodesPortal.Get(i_superparent))
                {
                    nodes_to_relabel_superparent.push_back(i_superparent);
                    nodes_to_relabel_hyperparent.push_back(hyperparentsPortal.Get(i_superparent)); // 2026-01-03 hyperparents failing when not matched
                    nodes_to_relabel_regularID.push_back(i_sortID);

                    // keep track of betti at a regular ID
                    nodes_to_relabel_regularID_betti_1.push_back(-betti1); // - minus betti to show if it's a supernode from before (for debug)


                    if(i_sortID > tailend)
                    {
                        node_ascend.push_back(-1);
//                        nodes_to_relabel_dataflip.push_back(dataPortal.Get(i_sortID) * -1.0); //data value can be the same, regular ID won't
                        nodes_to_relabel_dataflip.push_back((double)i_sortID * -1.0);
                    }
                    else
                    {
                        node_ascend.push_back(1);
//                        nodes_to_relabel_dataflip.push_back(dataPortal.Get(i_sortID) * 1.0);  //data value can be the same, regular ID won't
                        nodes_to_relabel_dataflip.push_back((double)i_sortID * 1.0);
                    }

                }
            }

            previous_betti1 = betti1;
        }


#if DEBUG_PRINT_PACTBD
        std::ofstream fileregbetti("ContourTreeBetti--RegToBetti.txt");
        std::cout << "Augment the tree with Betti Numbers ..." << std::endl;

//        for(int i = 0; i < ; i++)
//        {
//            vtkm::Id regularId = nodesPortal.Get(sortID);
//            vtkm::Id superparentId = superparentsPortal.Get(regularId);
//            vtkm::Id hyperparentId = hyperparentsPortal.Get(superparentId);
//        }

        for(int i = 0; i < nodes_to_relabel_regularID.size(); i++)
        {
            std::cout << nodes_to_relabel_regularID[i] << "\t" << nodes_to_relabel_regularID_betti_1[i] << std::endl;
            fileregbetti << nodes_to_relabel_regularID[i] << "\t" << nodes_to_relabel_regularID_betti_1[i] << std::endl;
        }

        //std::this_thread::sleep_for(std::chrono::seconds(3));
#endif

        // Augment the tree with Betti Numbers
//        HierarchicalAugmenter<MESH_TRIANGULATION_T> betti_augmenter;
//        betti_augmenter.Initialize(block, &hierarchicalTrees[block], &augmentedTrees[block], &meshes[block]);
//        betti_augmenter.BuildAugmentedTree();

#if DEBUG_PRINT_PACTBD

        std::ofstream filebettiaug("ContourTreeBetti--BettiAugmentation.txt");

        auto augNodesPortal = contourTree.Augmentnodes.ReadPortal();
        auto augArcsPortal = contourTree.Augmentarcs.ReadPortal();
        auto transferPortal = contourTree.WhenTransferred.ReadPortal();

        std::cout << "Augmented Nodes (" << augNodesPortal.GetNumberOfValues() << ")" << std::endl;
        filebettiaug << "Augmented Nodes (" << augNodesPortal.GetNumberOfValues() << ")" << std::endl;
        for(int i = 0; i < augNodesPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id augnode = augNodesPortal.Get(i);
            std::cout << i << "\t" << augnode << std::endl;
            filebettiaug << i << "\t" << augnode << std::endl;
        }
        std::cout << "Augmented Arcs (" << augArcsPortal.GetNumberOfValues() << ")" << std::endl;
        filebettiaug << "Augmented Arcs (" << augArcsPortal.GetNumberOfValues() << ")" << std::endl;
        for(int i = 0; i < augArcsPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id augarc = augArcsPortal.Get(i);
            std::cout << i << "\t" << augarc << std::endl;
            filebettiaug << i << "\t" << augarc << std::endl;
        }
        std::cout << "Transferred When (" << transferPortal.GetNumberOfValues() << ")" << std::endl;
        filebettiaug << "Transferred When (" << transferPortal.GetNumberOfValues() << ")" << std::endl;
        for(int i = 0; i < transferPortal.GetNumberOfValues(); i++)
        {
            std::cout << i << "\t" << vtkm::worklet::contourtree_augmented::MaskedIndex(transferPortal.Get(i)) << std::endl;
            filebettiaug << i << "\t" << vtkm::worklet::contourtree_augmented::MaskedIndex(transferPortal.Get(i)) << std::endl;
        }

        std::cout << "Nodes" << std::endl;
        filebettiaug << "Nodes" << std::endl;
        for(int sortID = 0; sortID < nodesPortal.GetNumberOfValues(); sortID++)
        {
            vtkm::Id regularId = nodesPortal.Get(sortID);
            std::cout << sortID << "\t" << regularId << std::endl;
            filebettiaug << sortID << "\t" << regularId << std::endl;
        }

        std::cout << "Arcs" << std::endl;
        filebettiaug << "Arcs" << std::endl;
        for(int sortID = 0; sortID < arcsPortal.GetNumberOfValues(); sortID++)
        {
            vtkm::Id regularId = arcsPortal.Get(sortID);
            vtkm::Id maskedArc = vtkm::worklet::contourtree_augmented::MaskedIndex(regularId);
            std::cout << sortID << "\t" << regularId << "\t" << maskedArc << std::endl;
            filebettiaug << sortID << "\t" << regularId << "\t" << maskedArc << std::endl;
        }


        std::cout << "Supernodes:" << std::endl;
        filebettiaug << "Supernodes:" << std::endl;
//         auto supernodesPortal = contourTree.Supernodes.ReadPortal();
        for(int i = 0; i < supernodesPortal.GetNumberOfValues(); i++)
        {
            std::cout << i << "\t" << supernodesPortal.Get(i) << std::endl;
            filebettiaug << i << "\t" << supernodesPortal.Get(i) << std::endl;
        }
        std::cout << "Superparents:" << std::endl;
        filebettiaug << "Superparents:" << std::endl;
        for(int i = 0; i < superparentsPortal.GetNumberOfValues(); i++)
        {
            std::cout << i << "\t" << superparentsPortal.Get(i) << std::endl;
            filebettiaug << i << "\t" << superparentsPortal.Get(i) << std::endl;
        }
        std::cout << "Superarcs:" << std::endl;
        filebettiaug << "Superarcs:" << std::endl;
        for(int i = 0; i < superarcsPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id maskedSuperarc = vtkm::worklet::contourtree_augmented::MaskedIndex(superarcsPortal.Get(i));
            std::cout << i << "\t" << superarcsPortal.Get(i) << "\t" << maskedSuperarc << std::endl;
            filebettiaug << i << "\t" << superarcsPortal.Get(i) << "\t" << maskedSuperarc << std::endl;
        }

        auto firstSupernodeIterationPortal = contourTree.FirstSupernodePerIteration.ReadPortal();
        auto firstHypernodeIterationPortal = contourTree.FirstHypernodePerIteration.ReadPortal();

        std::cout << "firstSupernodeIterationPortal:" << std::endl;
        filebettiaug << "firstSupernodeIterationPortal:" << std::endl;
        for(int i = 0; i < firstSupernodeIterationPortal.GetNumberOfValues(); i++)
        {
            std::cout << i << "\t" << firstSupernodeIterationPortal.Get(i) << std::endl;
            filebettiaug << i << "\t" << firstSupernodeIterationPortal.Get(i) << std::endl;
        }

        std::cout << "Hypernodes:" << std::endl;
        filebettiaug << "Hypernodes:" << std::endl;
        for(int i = 0; i < hypernodesPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id hypernodeID = hypernodesPortal.Get(i);
            vtkm::Id hyperparentID = hyperparentsPortal.Get(hypernodeID);
            std::cout << i << "\t" << hypernodeID << std::endl; //<< "\t" << hyperparentID << std::endl;
            filebettiaug << i << "\t" << hypernodeID << std::endl; //<< "\t" << hyperparentID << std::endl;
        }

        std::cout << "Hyperparents:" << std::endl;
        filebettiaug << "Hyperparents:" << std::endl;
        for(int i = 0; i < hyperparentsPortal.GetNumberOfValues(); i++)
        {
            std::cout << i << ") " << hyperparentsPortal.Get(i) << std::endl;
            filebettiaug << i << ") " << hyperparentsPortal.Get(i) << std::endl;
        }

        std::cout << "Hyperarcs:" << std::endl;
        filebettiaug << "Hyperarcs:" << std::endl;
        for(int i = 0; i < hyperarcsPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id maskedHyperarc = vtkm::worklet::contourtree_augmented::MaskedIndex(hyperarcsPortal.Get(i));
            std::cout << i << "\t" << hyperarcsPortal.Get(i) << "\t" << maskedHyperarc << std::endl;
            filebettiaug << i << "\t" << hyperarcsPortal.Get(i) << "\t" << maskedHyperarc << std::endl;
        }

        std::cout << "firstHypernodeIterationPortal:" << std::endl;
        filebettiaug << "firstHypernodeIterationPortal:" << std::endl;
        for(int i = 0; i < firstHypernodeIterationPortal.GetNumberOfValues(); i++)
        {
            std::cout << i << "\t" << firstHypernodeIterationPortal.Get(i) << std::endl;
            filebettiaug << i << "\t" << firstHypernodeIterationPortal.Get(i) << std::endl;
        }

        std::cout << "node->supernode->superarc(superparent)->hypernode->hyperarc(hyperparent) mappings" << std::endl;
        filebettiaug << "node->supernode->superarc(superparent)->hypernode->hyperarc(hyperparent) mappings" << std::endl;
        for(int sortID = 0; sortID < nodesPortal.GetNumberOfValues(); sortID++)
        {
            vtkm::Id regularId = nodesPortal.Get(sortID);
            vtkm::Id superparentId = superparentsPortal.Get(regularId);
            vtkm::Id hyperparentId = hyperparentsPortal.Get(superparentId);

//            std::cout << sortID << ")" << regularID << "->" << superparentID << "(" << hyperparentID << ")" << std::endl;
//            std::cout << sortID << "\t" << regularId << "\t" << superparentId << "\t" << hyperparentId << std::endl;

//            std::cout << "Probing HyperPath\n";
//            std::cout << "Node:        " << sortID << std::endl;
//            std::cout << "Regular ID: ";
//            vtkm::worklet::contourtree_augmented::PrintIndexType(regularId, std::cout);
//            resultStream << "  Value: " << vtkm::cont::ArrayGetValue(regularId, this->DataValues);
//            resultStream << " Global ID: ";
//            vtkm::worklet::contourtree_augmented::PrintIndexType(
//              vtkm::cont::ArrayGetValue(regularId, this->RegularNodeGlobalIds), std::cout);
//            resultStream << " Regular ID: ";
//            vtkm::worklet::contourtree_augmented::PrintIndexType(regularId, std::cout);
//            resultStream << " SNode ID: ";
//            vtkm::worklet::contourtree_augmented::PrintIndexType(
//              vtkm::cont::ArrayGetValue(regularId, this->Regular2Supernode), std::cout);
//            std::cout << "Superparents: ";
//            vtkm::worklet::contourtree_augmented::PrintIndexType(
//              vtkm::cont::ArrayGetValue(regularId, contourTree.Superparents), std::cout << "\t");

//            vtkm::Id hypertarget = vtkm::cont::ArrayGetValue(hyperparentId, contourTree.Hyperarcs);
//            std::cout << "Hypertarget: " << vtkm::cont::ArrayGetValue(hypertarget, contourTree.Hyperarcs) << std::endl;
//            std::cout << "Hypertargets: ";
            vtkm::Id hypertarget = vtkm::cont::ArrayGetValue(hyperparentId, contourTree.Hyperarcs);
            vtkm::Id maskedHypertarget = vtkm::worklet::contourtree_augmented::MaskedIndex(hypertarget);
//            vtkm::worklet::contourtree_augmented::PrintIndexType(
//              vtkm::cont::ArrayGetValue(hypertarget, contourTree.Superparents));
//            std::cout << hypertarget << " - " /*<< vtkm::cont::ArrayGetValue(hypertarget, contourTree.Superparents)*/ << std::endl;

            vtkm::Id supertarget = vtkm::cont::ArrayGetValue(superparentId, contourTree.Superarcs);
            vtkm::Id maskedSupertarget = vtkm::worklet::contourtree_augmented::MaskedIndex(supertarget);
//            std::cout << supertarget << " - " /*<< vtkm::cont::ArrayGetValue(supertarget, contourTree.Superparents)*/ << std::endl;

             std::cout << sortID << "\t" << regularId << "\t" << superparentId << "(" << maskedSupertarget << ")"
                       << "\t" << hyperparentId << "(" << maskedHypertarget << ")" << std::endl;
             filebettiaug << sortID << "\t" << regularId << "\t" << superparentId << "(" << maskedSupertarget << ")"
                       << "\t" << hyperparentId << "(" << maskedHypertarget << ")" << std::endl;
        }

#endif


//        int num_betti_change_nodes = 10;
//        int bid_size = supernodesPortal.GetNumberOfValues() + num_betti_change_nodes;

        using vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT;

        // following the template to resize arrays (from HierarchicalAugmenter.h ResizeArrays(vtkm::Id roundNumber):959)
//        vtkm::worklet::contourtree_augmented::ResizeVector(
//          &contourTree.Supernodes,
//          bid_size,
//          vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);

        // first    : sort on HP
        // secondary: on data value (with its comparitor) with the flip ascending/descending
        // tertiary : sort on regular ID (out of paranoia)
//        std::cout << "id)\thpID\tval_filp\tregularID" << std::endl;
//        for(int i = 0; i < supernodesPortal.GetNumberOfValues(); i++)
//        {
//            vtkm::Id regularId = supernodesPortal.Get(i);
//            vtkm::Id superparentId = superparentsPortal.Get(regularId);
//            vtkm::Id hyperparentId = hyperparentsPortal.Get(superparentId);

////            vtkm::Id i_sortID = supernodesPortal.Get(i+1);
////            vtkm::Id i_superparent = superparentsPortal.Get(i_sortID);

//            vtkm::Id regularId_j = supernodesPortal.Get(i+1);
////            vtkm::Id j_superparent = superparentsPortal.Get(j_sortID );

//            std::cout << i << ")  " << supernodesPortal.Get(i)
//                      << "\t" << hyperparentId
//                      << "\t" << regularId
//                      << "\t(" << supernodesPortal.Get(superparentId) << ")"
//                      << std::endl;
//        }

//        for(int i = 0; i < regular_nodes_to_insert.size(); i++)
//        {
//            vtkm::Id regularId = regular_nodes_to_insert[i];
//            vtkm::Id superparentId = superparentsPortal.Get(regularId);
//            vtkm::Id hyperparentId = hyperparentsPortal.Get(superparentId);

//            std::cout << i << ")  " << regular_nodes_to_insert[i]
//                      << "\t" << hyperparentId
//                      << "\t" << regularId
//                      << "\t(" << supernodesPortal.Get(superparentId) << ")"
//                      << std::endl;
//        }

//        contourTree.SupernodeBetti.resize(nodes_to_relabel_regularID.size()); // resize to write at [i]
        contourTree.SupernodeBetti.Allocate(nodes_to_relabel_regularID.size()); // resize to write at [i]
        contourTree.BettiOriginalSuperparents.Allocate(nodes_to_relabel_regularID.size()); // resize to write at [i]

        auto ct_betti_portal = contourTree.SupernodeBetti.WritePortal();
        auto ct_BettiOriginalSuperparents_portal = contourTree.BettiOriginalSuperparents.WritePortal();
#if DEBUG_PRINT_PACTBD
        filebettiaug << "The BID array for rehooking up the super{hyper}structure:" << std::endl;
        filebettiaug << "id)\thpID\tval_filp\tregularID\tSP" << std::endl;
        for(int i = 0; i < nodes_to_relabel_regularID.size(); i++)
        {
//            vtkm::Id regularId = nodes_to_relabel_regularID[i];
            vtkm::Id superparentId = superparentsPortal.Get(nodes_to_relabel_regularID[i]);
            vtkm::Id hyperparentId = hyperparentsPortal.Get(superparentId);

            filebettiaug << i << ")  " << nodes_to_relabel_hyperparent[i]
                           << "\t"  << nodes_to_relabel_dataflip[i]
                           << "\t"  << nodes_to_relabel_regularID[i]
                           << "\t(" << supernodesPortal.Get(superparentId) << ")"
                           << "\t"  << nodes_to_relabel_superparent[i]
                           << "\t"  << nodes_to_relabel_regularID_betti_1[i] // include betti numbers for superparents
                           << std::endl;
        }

#endif



        vtkm::cont::ArrayHandle<vtkm::Id> ah_betti;
        ah_betti.Allocate(nodes_to_relabel_regularID_betti_1.size());
        auto portal_betti = ah_betti.WritePortal();
        for (vtkm::Id i = 0; i < vtkm::Id(nodes_to_relabel_regularID_betti_1.size()); ++i)
            portal_betti.Set(i, nodes_to_relabel_regularID_betti_1[i]);


        vtkm::cont::ArrayHandle<vtkm::Id> ah_super;
        ah_super.Allocate(nodes_to_relabel_superparent.size());
        auto portal_sp = ah_super.WritePortal();
        for (vtkm::Id i = 0; i < vtkm::Id(nodes_to_relabel_superparent.size()); ++i)
            portal_sp.Set(i, nodes_to_relabel_superparent[i]);


        vtkm::cont::ArrayHandle<vtkm::Id> ah_hyper;
        ah_hyper.Allocate(nodes_to_relabel_hyperparent.size());
        auto portal = ah_hyper.WritePortal();
        for (vtkm::Id i = 0; i < vtkm::Id(nodes_to_relabel_hyperparent.size()); ++i)
            portal.Set(i, nodes_to_relabel_hyperparent[i]);

        vtkm::cont::ArrayHandle<double> ah_data;
        ah_data.Allocate(nodes_to_relabel_dataflip.size());
        auto portal_data = ah_data.WritePortal();
        for (vtkm::Id i = 0; i < vtkm::Id(nodes_to_relabel_dataflip.size()); ++i)
            portal_data.Set(i, nodes_to_relabel_dataflip[i]);

        vtkm::cont::ArrayHandle<vtkm::Id> ah_regular;
        ah_regular.Allocate(nodes_to_relabel_regularID.size());
        auto portal_regular = ah_regular.WritePortal();
        for (vtkm::Id i = 0; i < vtkm::Id(nodes_to_relabel_regularID.size()); ++i)
            portal_regular.Set(i, nodes_to_relabel_regularID[i]);

        // first zip 2 arrays
//        auto zipped12 = vtkm::cont::make_ArrayHandleZip(ah_hyper, ah_data);
//        auto zipped34 = vtkm::cont::make_ArrayHandleZip(ah_regular, ah_super); // 2026-01-03

        auto zipped12 = vtkm::cont::make_ArrayHandleZip(ah_super, ah_data);
        auto zipped34 = vtkm::cont::make_ArrayHandleZip(ah_regular, ah_hyper); // 2026-01-03
        auto zipped345 = vtkm::cont::make_ArrayHandleZip(zipped34, ah_betti); // 2026-01-03

        // then zip the previous zip with the final array
//        auto zipped123 = vtkm::cont::make_ArrayHandleZip(zipped12, ah_regular); // 2026-01-03
//        auto zipped123 = vtkm::cont::make_ArrayHandleZip(zipped12, ah_regular); // 2026-01-03
//        auto zipped123 = vtkm::cont::make_ArrayHandleZip(zipped12, zipped34); // 2026-01-03
        auto zipped123 = vtkm::cont::make_ArrayHandleZip(zipped12, zipped345); // 2026-01-15

        // Sort lexicographically
        vtkm::cont::Algorithm::Sort(zipped123);

        int num_original_supernodes = supernodesPortal.GetNumberOfValues();

#if DEBUG_PRINT_PACTBD
        std::cout << "!!!!!!!!!!!!!!!! PREVIOUS SUPERNODE LIST MAX ID (LEN) !!!!!!!!!!!!!!!!" << std::endl;
        std::cout << num_original_supernodes << std::endl;

        std::cout << "!!!!!!!!!!!!!!!! SORTED !!!!!!!!!!!!!!!!" << std::endl;

        std::cout << "i"
                  << "\t" << "HP"
                  << "\t" << "valflip"
                  << "\t" << "regID"
                  << "\t" << "SP"
//                  << "\t" << "isNew"
                  << "\t" << "+1==HP"
                  << "\t" << "HT"
//                  << "\t" << "pfixsum"
                  << "\t" << "relabel"
                  << "\t" << "new ST"
                  << std::endl;
#endif

        auto zipPortal = zipped123.ReadPortal();
        int new_nodes_pfix_sum = 0;

        std::vector<vtkm::Id> newSuperIDsRelabelled; // contain either old or new super ID names in single array
        std::vector<vtkm::Id> newSupernodes;

        // array holding the REGULAR IDs of the to-become new supernodes thanks to betti augmentation
        // using NEWSUPERID to index these regular ids
        newSupernodes.resize(nodes_to_relabel_regularID.size()); //20); // hack-resolved

#if DEBUG_PRINT_PACTBD
        std::cout << "nodes_to_relabel_regularID size = " << nodes_to_relabel_regularID.size() << std::endl;
#endif

        for (vtkm::Id i = 0; i < zipPortal.GetNumberOfValues(); ++i)
        {
            auto triple = zipPortal.Get(i);
            // Because of nested pairs:
//            auto zipped34 = vtkm::cont::make_ArrayHandleZip(ah_regular, ah_hyper); // 2026-01-03
//            auto zipped345 = vtkm::cont::make_ArrayHandleZip(zipped34, ah_betti); // 2026-01-03
            double  dataflip     = triple.first.second;   // (second of inner pair)
            vtkm::Id superparent   = triple.first.first; //  triple.second.second         // (second of outer pair) // 2026-01-03 was triple.second before
            vtkm::Id hyperparent = triple.second.first.second; //triple.second.second; //triple.first.first;    // (first of outer pair) -> first of inner pair
            vtkm::Id regularID   = triple.second.first.first;   // triple.second.first;         // (second of outer pair) triple.second;
            vtkm::Id superID     = superparentsPortal.Get(regularID);

            bool isNew = false;
            vtkm::Id new_superID_relabel = superparent; //hyperparent; // 2026-01-03 actually superparent here

            if(regularID != supernodesPortal.Get(superID))
            {
                isNew = true;
                new_nodes_pfix_sum++;
                new_superID_relabel = (num_original_supernodes-1)+new_nodes_pfix_sum;
            }

            newSuperIDsRelabelled.push_back(new_superID_relabel);

            // superID to regularID NEW mapping:
            newSupernodes[new_superID_relabel] = regularID;


        }

#if DEBUG_PRINT_PACTBD
        std::cout << "newSupernodes array: " << newSupernodes.size() << std::endl;
        for(int i = 0; i < newSupernodes.size(); i++)
        {
            std::cout << i << "\t" << newSupernodes[i] << std::endl;
        }

        //std::this_thread::sleep_for(std::chrono::seconds(3));
#endif

        vtkm::Id num_added_supernodes = newSupernodes.size() - contourTree.Supernodes.GetNumberOfValues();

        bool plus1test = false;
        vtkm::Id newSuperTarget = -1;

        std::vector<vtkm::Id> newSuperTargets;

#if WRITE_FILES
        std::ofstream filebettitable("ContourTreeBetti--BettiTable.txt");

        filebettitable << "i"
                  << "\thyperparent"
                  << "\tdataflip"
                  << "\tregularID"
                  << "\tsupernodesPortal.Get(superID)"
                  << "\tnborSuperparent"
                  << "\tplus1test"
                  << "\tsuperparent"
                  << "\tsupertarget"
                  << "\tnewSuperIDsRelabelled[i]"
                  << "\tnewSuperTargets[i]"
                  << "\taug_betti_num"
                  << std::endl;
#endif

		std::cout << "total zips: " << zipPortal.GetNumberOfValues() << std::endl;

        for (vtkm::Id i = 0; i < zipPortal.GetNumberOfValues(); ++i)
        {
			//std::cout << "processing " << i << "th zip" << std::endl;
            
            plus1test = false;
            auto triple = zipPortal.Get(i);

            // Because of nested pairs:
//            vtkm::Id hyperparent = triple.second.second; //triple.first.first;    // (first of outer pair) -> first of inner pair
            double  dataflip     = triple.first.second;   // (second of inner pair)
//            vtkm::Id regularID   = triple.second.first;         // (second of outer pair) triple.second;
            vtkm::Id superparent   = triple.first.first; //  triple.second.second         // (second of outer pair) // 2026-01-03 was triple.second before

            vtkm::Id hyperparent = triple.second.first.second; //triple.second.second; //triple.first.first;    // (first of outer pair) -> first of inner pair
            vtkm::Id regularID   = triple.second.first.first;   // triple.second.first;         // (second of outer pair) triple.second;

            vtkm::Id aug_betti_num   = triple.second.second;   // triple.second.first;         // (second of outer pair) triple.second;

            //auto nextTriple = zipPortal.Get(i+1);
////                vtkm::Id nborHyperparent = nextTriple.first.first;    // (first of outer pair) -> first of inner pair
////            vtkm::Id nborSuperparent = nextTriple.second.second;    // 2026-01-09 bug found - this is now HP! (first of outer pair) -> first of inner pair 2026-01-03 use SPs
            //vtkm::Id nborSuperparent = nextTriple.first.first;    // (first of outer pair) -> first of inner pair 2026-01-03 use SPs
            vtkm::Id nborSuperparent = vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT;

            // 2026-01-27 track the changes in betti instead of the betti itself, then use it to get the biggest change instead of raw betti1
//            vtkm::cont::ArrayHandleZip previousTriple;
//            if(i > 0)
//            {
//                auto previousTriple = zipPortal.Get(i-1);
//                aug_betti_num = abs(abs(aug_betti_num) - abs(previousTriple.second.second));
//            }

			//std::cout << "           " << i << " (a)" << std::endl;

            if(i+1 >= zipPortal.GetNumberOfValues())
            {
                // if the last element, do nothing
            }
            else
            {
				
            auto nextTriple = zipPortal.Get(i+1);
//                vtkm::Id nborHyperparent = nextTriple.first.first;    // (first of outer pair) -> first of inner pair
//            vtkm::Id nborSuperparent = nextTriple.second.second;    // 2026-01-09 bug found - this is now HP! (first of outer pair) -> first of inner pair 2026-01-03 use SPs
            nborSuperparent = nextTriple.first.first;    // (first of outer pair) -> first of inner pair 2026-01-03 use SPs				
				
//                auto nextTriple = zipPortal.Get(i+1);
////                vtkm::Id nborHyperparent = nextTriple.first.first;    // (first of outer pair) -> first of inner pair
//                vtkm::Id nborSuperparent = nextTriple.second.second;    // (first of outer pair) -> first of inner pair 2026-01-03 use SPs

                if(superparent == nborSuperparent)
//                    if(hyperparent == nborHyperparent)
                {
                    // segmented test:
                    // if the next row in the sort is on the same superarc, ...
                    // ... set the segmented flag
                    plus1test = true;
                }
            }
            
            //std::cout << "           " << i << " (b)" << std::endl;

            if(plus1test)
            { // if not end of segment yet ...
              // ... set the supertarget as the next new super ID
//                newSuperTarget = newSuperIDsRelabelled[i+1];
                newSuperTargets.push_back(newSuperIDsRelabelled[i+1]);
            }
            else
            {// if it's the end of the segment, ...
             // ... set the supertarget to be the previous hypertarget (not the next supernode)
                // (reached end of the last superarc)
//                newSuperTarget = vtkm::worklet::contourtree_augmented::MaskedIndex(vtkm::cont::ArrayGetValue(hyperparent, contourTree.Hyperarcs));
//                newSuperTargets.push_back(vtkm::worklet::contourtree_augmented::MaskedIndex(vtkm::cont::ArrayGetValue(hyperparent, contourTree.Hyperarcs)));
//                newSuperTargets.push_back(vtkm::worklet::contourtree_augmented::MaskedIndex(vtkm::cont::ArrayGetValue(hyperparent, contourTree.Hyperarcs)));
                newSuperTargets.push_back(vtkm::worklet::contourtree_augmented::MaskedIndex(vtkm::cont::ArrayGetValue(superparent, contourTree.Superarcs)));
            }
            
            //std::cout << "           " << i << " (c)" << std::endl;
            
            vtkm::Id superID     = superparentsPortal.Get(regularID);
            
            //std::cout << "           " << i << " (d)" << std::endl;

//            bool isNew = false;
//            vtkm::Id new_superID_relabel = hyperparent;

//            if(regularID != supernodesPortal.Get(superID))
//            {
//                isNew = true;
//                new_nodes_pfix_sum++;

//                new_superID_relabel = (num_original_supernodes-1)+new_nodes_pfix_sum;
//            }

#if WRITE_FILES
            filebettitable << i
                      << "\t" << hyperparent
                      << "\t" << dataflip
                      << "\t" << regularID
                      << "\t" << supernodesPortal.Get(superID)
                      << "\t" << nborSuperparent
//                      << "\t" << isNew
                      << "\t" << plus1test
//                      << "\t" << vtkm::worklet::contourtree_augmented::MaskedIndex(vtkm::cont::ArrayGetValue(hyperparent, contourTree.Hyperarcs))
                      << "\t" << superparent
                      << "\t" << vtkm::worklet::contourtree_augmented::MaskedIndex(vtkm::cont::ArrayGetValue(superparent, contourTree.Superarcs))
//                      << "\t" << new_nodes_pfix_sum
                      << "\t" << newSuperIDsRelabelled[i]
                         << "\t" << newSuperTargets[i]
                            << "\t" << aug_betti_num
                      << std::endl;
#endif

            // keep track of betti numbers per supernode:
//            contourTree.SupernodeBetti[newSuperIDsRelabelled] = aug_betti_num;

			//std::cout << "           " << i << " (e)" << std::endl;
			
			//std::cout << "ct_betti_portal.size() = " << ct_betti_portal.GetNumberOfValues() << std::endl;
			//std::cout << "ct_BettiOriginalSuperparents_portal.size() = " << ct_BettiOriginalSuperparents_portal.GetNumberOfValues() << std::endl;

            ct_betti_portal.Set	(newSuperIDsRelabelled[i], aug_betti_num);
            ct_BettiOriginalSuperparents_portal.Set(newSuperIDsRelabelled[i], superparent);
            
            //std::cout << "           " << i << " (m)" << std::endl;
        }

#if WRITE_FILES
        filebettitable << "sortID\tregularID\tSP" << std::endl;
        //std::this_thread::sleep_for(std::chrono::seconds(3));

        for(int sortID = 0; sortID < nodesPortal.GetNumberOfValues(); sortID++)
        {
            vtkm::Id regularId = nodesPortal.Get(sortID);
            filebettitable << sortID << "\t" << regularId << "\t" << superparentsPortal.Get(regularId) << std::endl;
        }
#endif

        auto arcsWritePortal = contourTree.Arcs.WritePortal();
        auto nodesWritePortal = contourTree.Nodes.WritePortal();
//        auto supernodesWritePortal = contourTree.Supernodes.WritePortal();


        vtkm::Id originalSize = superarcsPortal.GetNumberOfValues();

#if DEBUG_PRINT_PACTBD
        std::cout << "ORIGINAL Superarcs:" << std::endl;
        for(int i = 0; i < superarcsPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id maskedSuperarc = vtkm::worklet::contourtree_augmented::MaskedIndex(superarcsPortal.Get(i));
            std::cout << i << "\t" << superarcsPortal.Get(i) << "\t" << maskedSuperarc << std::endl;
        }
        std::cout << "Increasing array size to: " << originalSize + num_added_supernodes << std::endl; // hack-resolved
#endif
        contourTree.Superarcs.Allocate(contourTree.Superarcs.GetNumberOfValues() + num_added_supernodes, vtkm::CopyFlag::On); // hack-resolved


        auto superarcsWritePortal = contourTree.Superarcs.WritePortal();


        // RELABEL SUPERARCS
#if DEBUG_PRINT_PACTBD
        std::cout << "RELABEL SUPERARCS: " << std::endl;
#endif
        for(int i = 0; i < newSuperIDsRelabelled.size(); i++)
        {
            // check superarc direction, if ascending, add the IS_ASCENDING flag
            if (newSupernodes[newSuperIDsRelabelled[i]] < newSupernodes[newSuperTargets[i]])
            {
                // RELABEL--
                superarcsWritePortal.Set(newSuperIDsRelabelled[i], newSuperTargets[i] | vtkm::worklet::contourtree_augmented::IS_ASCENDING);
#if DEBUG_PRINT_PACTBD
                std::cout << newSuperIDsRelabelled[i] << "\t" << newSupernodes[newSuperIDsRelabelled[i]] << " < "
                          << newSupernodes[newSuperTargets[i]] << "\tIS_ASCENDING" << std::endl;
#endif
            }
            else
            {   // id descending, don't need a flag
                // RELABEL--
                superarcsWritePortal.Set(newSuperIDsRelabelled[i], newSuperTargets[i]);
#if DEBUG_PRINT_PACTBD
                std::cout << newSuperIDsRelabelled[i] << "\t" << newSupernodes[newSuperIDsRelabelled[i]] << " > "
                          << newSupernodes[newSuperTargets[i]] << std::endl;
#endif
            }

            if(newSuperTargets[i] == 0)
            {// if root node, the flag is NO_SUCH_ELEMENT by convention
                // RELABEL--
                superarcsWritePortal.Set(newSuperIDsRelabelled[i], newSuperTargets[i] | vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
#if DEBUG_PRINT_PACTBD
                std::cout << newSuperIDsRelabelled[i] << "\t" << "\tROOT" << std::endl;
#endif
            }
        }

        auto superarcsReinvPortal = contourTree.Superarcs.ReadPortal();

#if DEBUG_PRINT_PACTBD
        std::cout << "RELABELLED SUPERARCS: " << superarcsReinvPortal.GetNumberOfValues() << std::endl;
        for(int i = 0; i < superarcsReinvPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id maskedSuperarc = vtkm::worklet::contourtree_augmented::MaskedIndex(superarcsReinvPortal.Get(i));
            std::cout << i << "\t" << superarcsReinvPortal.Get(i) << "\t" << maskedSuperarc << "\t"
                      << vtkm::worklet::contourtree_augmented::IsAscending(superarcsReinvPortal.Get(i)) << std::endl;

//            vtkm::worklet::contourtree_augmented::NoSuchElement for finding NAN (such as root) nodes
//            std::cout << newSupernodes[i] << "\t" << newSupernodes[maskedSuperarc] << std::endl;
        }
#endif


        std::vector<vtkm::Id> newSupernodeHyperparents;
#if DEBUG_PRINT_PACTBD
        std::cout << "NEW HYPERPARENTS:" << std::endl;
#endif
        for(int i = 0; i < newSupernodes.size(); i++)
        {
#if DEBUG_PRINT_PACTBD
            std::cout << i << "\t" << newSupernodes[i] << "\t"
                      << superparentsPortal.Get(newSupernodes[i]) << "\t"
                      << hyperparentsPortal.Get(superparentsPortal.Get(newSupernodes[i])) << std::endl;
#endif

            newSupernodeHyperparents.push_back(hyperparentsPortal.Get(superparentsPortal.Get(newSupernodes[i])));
        }

        contourTree.Hyperparents.Allocate(contourTree.Hyperparents.GetNumberOfValues() + num_added_supernodes, vtkm::CopyFlag::On);

        auto hyperparentsWritePortal = contourTree.Hyperparents.WritePortal();

        for(int i = 0; i < newSupernodeHyperparents.size(); i++)
        {
            hyperparentsWritePortal.Set(i, newSupernodeHyperparents[i]);
        }

//        auto hyperparentsRelabelledPortal = contourTree.Hyperparents.ReadPortal();

//        std::cout << "RELABELLED HYPERPARENTS" << std::endl;
//        for(int i = 0; i < hyperparentsRelabelledPortal.GetNumberOfValues(); i++)
//        {
//            std::cout << i << "\t " << hyperparentsRelabelledPortal.Get(i) << std::endl;
//        }


#if DEBUG_PRINT_PACTBD
        std::cout << "do a regular walk from superarcs to relabel superparents:" << std::endl;
        std::cout << "from\tto\tset\tgetseg" << std::endl;
#endif

        std::vector<vtkm::Id> replaceSuperparentsWith;
        std::vector<vtkm::Id> targetSegments;
        std::vector<vtkm::Id> fromRegID;
        std::vector<vtkm::Id> toRegID;

        std::vector<vtkm::Id> oldSuperparents; // used for telling whenTransferred
        oldSuperparents.resize(superarcsReinvPortal.GetNumberOfValues());

        for(int i = 0; i < superarcsReinvPortal.GetNumberOfValues(); i++)
        {
            vtkm::Id maskedSuperarc = vtkm::worklet::contourtree_augmented::MaskedIndex(superarcsReinvPortal.Get(i));
#if DEBUG_PRINT_PACTBD
            std::cout << newSupernodes[i] << "\t" << newSupernodes[maskedSuperarc] << "\t" << i <<  "\t"
                      << superparentsPortal.Get(newSupernodes[i]);// << std::endl;
#endif

            oldSuperparents[i] = superparentsPortal.Get(newSupernodes[i]);

            if(i != superparentsPortal.Get(newSupernodes[i]))
            {
                targetSegments.push_back(superparentsPortal.Get(newSupernodes[i]));
#if DEBUG_PRINT_PACTBD
                std::cout << "\tadded";
#endif
                fromRegID.push_back(newSupernodes[i]);
                toRegID.push_back(newSupernodes[maskedSuperarc]);

                replaceSuperparentsWith.push_back(i);
            }
#if DEBUG_PRINT_PACTBD
            std::cout << std::endl;
#endif
        }

        std::vector<vtkm::Id> segmentA, segmentB;
#if DEBUG_PRINT_PACTBD
        std::cout << "RELABEL SUPERPARENTS ... " << std::endl;
        std::cout << "sortID\tregularID\tSP" << std::endl;
#endif
        for(int sortID = 0; sortID < nodesPortal.GetNumberOfValues(); sortID++)
        {
            vtkm::Id regularId = nodesPortal.Get(sortID);
#if DEBUG_PRINT_PACTBD
            std::cout << sortID << "\t"
                      << regularId << "\t"
                      << superparentsPortal.Get(regularId)
                      << std::endl;
#endif

            segmentA.push_back(regularId);
            segmentB.push_back(superparentsPortal.Get(regularId));
        }


//        std::this_thread::sleep_for(std::chrono::seconds(3));

        vtkm::Id targetSegment = 6;

        auto superparentsWritePortal = contourTree.Superparents.WritePortal();

        for(int i = 0; i < targetSegments.size(); i++)
        {
            // Find the range of segment
            auto beginIt = std::lower_bound(
                segmentB.begin(), segmentB.end(), targetSegments[i]);

            auto endIt = std::upper_bound(
                segmentB.begin(), segmentB.end(), targetSegments[i]);

            std::size_t begin = std::distance(segmentB.begin(), beginIt);
            std::size_t end   = std::distance(segmentB.begin(), endIt);

#if DEBUG_PRINT_PACTBD
            std::cout << "Segment " << targetSegments[i]
                      << " reg range: [" << fromRegID[i] << ", " << toRegID[i] << ")\n"
                      << " idx range: [" << begin << ", " << end << ")\n";
#endif

//            oldSuperparents[replaceSuperparentsWith[i]] =

            // Extract corresponding A values
            for (std::size_t j = begin; j < end; j++)
            {
#if DEBUG_PRINT_PACTBD
                std::cout << "A[" << j << "] = " << segmentA[j];// << "\n";
#endif
                if(
                    ( (segmentA[j] <= fromRegID[i]) && (segmentA[j] > toRegID[i]))
                    ||
                    ( (segmentA[j] >= fromRegID[i]) && (segmentA[j] < toRegID[i]))
                  )
                {
#if DEBUG_PRINT_PACTBD
//                    std::cout << "\t" << superparentsPortal.Get(j) << "\treplace to: " << replaceSuperparentsWith[i];
                    std::cout << "\t" << superparentsPortal.Get(segmentA[j]) << "\treplace to: " << replaceSuperparentsWith[i];
#endif

                    // RELABEL SUPERPARENTS:
                    // RELABEL--
                    superparentsWritePortal.Set(segmentA[j], replaceSuperparentsWith[i]);

                }
#if DEBUG_PRINT_PACTBD
                std::cout << std::endl;
#endif
            }

        }

        auto superparentsRewritePortal = contourTree.Superparents.ReadPortal();
#if DEBUG_PRINT_PACTBD
        std::cout << "REPLACED SUPERPARENTS ..." << std::endl;
        for(int i = 0; i < superparentsRewritePortal.GetNumberOfValues(); i++)
        {
            std::cout << i << "\t" << superparentsRewritePortal.Get(i) << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(3));

        std::cout << "NEW SUPERNODES:" << std::endl;
        for(int i = 0; i < newSupernodes.size(); i++)
        {
            std::cout << i << "\t" << newSupernodes[i] << "\t"
                      << superparentsPortal.Get(newSupernodes[i]) << "\t"
                      << hyperparentsPortal.Get(superparentsPortal.Get(newSupernodes[i])) << std::endl;
        }

        std::cout << "Increasing array size to: " << originalSize + num_added_supernodes << std::endl; // hack-resolved
#endif
//        vtkm::Id num_original_supernodes already defined ...
//        vtkm::Id num_added_supernodes = newSupernodes.size() - contourTree.Supernodes.GetNumberOfValues();
        contourTree.Supernodes.Allocate(contourTree.Supernodes.GetNumberOfValues() + num_added_supernodes, vtkm::CopyFlag::On);

#if DEBUG_PRINT_PACTBD
        std::cout << "RELABEL SUPERNODES:" << std::endl;
#endif
        auto supernodesWritePortal = contourTree.Supernodes.WritePortal();
        for(int i = num_original_supernodes;
                i < num_original_supernodes + num_added_supernodes;
                i++)
        {
#if DEBUG_PRINT_PACTBD
            std::cout << i << "\t" << newSupernodes[i] << std::endl;
#endif
            // RELABEL--
            supernodesWritePortal.Set(i,newSupernodes[i]);
        }

//        std::cout << "RELABELLED SUPERNODES:" << std::endl;
//        auto supernodesRelabelledPortal = contourTree.Supernodes.ReadPortal();
//        for(int i = 0; i < supernodesRelabelledPortal.GetNumberOfValues(); i++)
//        {
//            std::cout << i << "\t" << supernodesRelabelledPortal.Get(i) << std::endl;
//        }

#if DEBUG_PRINT_PACTBD
        std::cout << "WhenTransferred" << std::endl;
#endif
//                auto whenTransferredWritePortal = contourTree.WhenTransferred.WritePortal();
        auto whenTransferredPortal = contourTree.WhenTransferred.ReadPortal();

        std::vector<vtkm::Id> whenTransferredVec;

        for(int i = 0; i < oldSuperparents.size(); i++)
        {
#if DEBUG_PRINT_PACTBD
            std::cout << i << "\t" << oldSuperparents[i] << "\t"
                      << whenTransferredPortal.Get(oldSuperparents[i]) << "\t"
                      << vtkm::worklet::contourtree_augmented::MaskedIndex(whenTransferredPortal.Get(oldSuperparents[i]))
                      << std::endl;
#endif
            whenTransferredVec.push_back(whenTransferredPortal.Get(oldSuperparents[i]));
        }

        contourTree.WhenTransferred.Allocate(contourTree.WhenTransferred.GetNumberOfValues() + num_added_supernodes, vtkm::CopyFlag::On);

        auto whenTransferredWritePortal = contourTree.WhenTransferred.WritePortal();
#if DEBUG_PRINT_PACTBD
        std::cout << "RELABEL WhenTransferred (" << contourTree.WhenTransferred.GetNumberOfValues() << ")\n";
#endif
        for(int i = 0; i < contourTree.WhenTransferred.GetNumberOfValues(); i++)
        {
            // RELABEL--
            whenTransferredWritePortal.Set(i, whenTransferredVec[i]);
        }

//        std::cout << "RELABELLED WhenTransferred (" << contourTree.WhenTransferred.GetNumberOfValues() << ")\n";
//        auto whenTransferredRelabelledPortal = contourTree.WhenTransferred.ReadPortal();
//        for(int i = 0; i < whenTransferredRelabelledPortal.GetNumberOfValues(); i++)
//        {
//            std::cout << i << "\t" << whenTransferredRelabelledPortal.Get(i) << std::endl;
//        }


//        vtkm::cont::ArrayHandle<vtkm::Id> Ahandle =
//            vtkm::cont::make_ArrayHandle(segmentA, vtkm::CopyFlag::Off);

//        vtkm::cont::ArrayHandle<vtkm::Id> Bhandle =
//            vtkm::cont::make_ArrayHandle(segmentB, vtkm::CopyFlag::Off);

//        // Segment ID(s) to extract
//        vtkm::cont::ArrayHandle<vtkm::Int32> segments2get =
//            vtkm::cont::make_ArrayHandle<vtkm::Int32>({ 6 });

//        vtkm::cont::ArrayHandle<vtkm::Id> outputLow =
//            vtkm::cont::Algorithm::LowerBounds(Bhandle, segments2get);

//        vtkm::cont::ArrayHandle<vtkm::Id> outputHigh =
//            vtkm::cont::Algorithm::UpperBounds(Bhandle, segments2get);

//        // Portals
//        auto aPortal  = Ahandle.ReadPortal();
//        auto olPortal = outputLow.ReadPortal();
//        auto ohPortal = outputHigh.ReadPortal();

//        // Correct range extraction
//        vtkm::Id begin = olPortal.Get(0);
//        vtkm::Id end   = ohPortal.Get(0);

//        for (vtkm::Id i = begin; i < end; i++)
//        {
//            std::cout << aPortal.Get(i) << std::endl;
//        }

        std::cout << "// ComputeBettiNumbersForRegularArcs() finished" << std::endl;























    } // ComputeBettiNumbersForRegularArcs()
































  // routine to compute the volume for each hyperarc and superarc
  void static ComputeVolumeWeightsSerial(const ContourTree& contourTree,
                                         const vtkm::Id nIterations,
                                         IdArrayType& superarcIntrinsicWeight,
                                         IdArrayType& superarcDependentWeight,
                                         IdArrayType& supernodeTransferWeight,
                                         IdArrayType& hyperarcDependentWeight)
  { // ContourTreeMaker::ComputeWeights()
    // start by storing the first sorted vertex ID for each superarc
    IdArrayType firstVertexForSuperparent;
    firstVertexForSuperparent.Allocate(contourTree.Superarcs.GetNumberOfValues());
    superarcIntrinsicWeight.Allocate(contourTree.Superarcs.GetNumberOfValues());
    auto superarcIntrinsicWeightPortal = superarcIntrinsicWeight.WritePortal();
    auto firstVertexForSuperparentPortal = firstVertexForSuperparent.WritePortal();
    auto superparentsPortal = contourTree.Superparents.ReadPortal();
    auto hyperparentsPortal = contourTree.Hyperparents.ReadPortal();
    auto hypernodesPortal = contourTree.Hypernodes.ReadPortal();
    auto hyperarcsPortal = contourTree.Hyperarcs.ReadPortal();
    // auto superarcsPortal = contourTree.Superarcs.ReadPortal();
    auto nodesPortal = contourTree.Nodes.ReadPortal();
    // auto whenTransferredPortal = contourTree.WhenTransferred.ReadPortal();
    for (vtkm::Id sortedNode = 0; sortedNode < contourTree.Arcs.GetNumberOfValues()-1; sortedNode++)
    { // per node in sorted order
      vtkm::Id sortID = nodesPortal.Get(sortedNode);
      vtkm::Id superparent = superparentsPortal.Get(sortID);
      if (sortedNode == 0)
        firstVertexForSuperparentPortal.Set(superparent, sortedNode);
      else if (superparent != superparentsPortal.Get(nodesPortal.Get(sortedNode - 1)))
        firstVertexForSuperparentPortal.Set(superparent, sortedNode);
    } // per node in sorted order
    // now we use that to compute the intrinsic weights
    for (vtkm::Id superarc = 0; superarc < contourTree.Superarcs.GetNumberOfValues(); superarc++)
      if (superarc == contourTree.Superarcs.GetNumberOfValues() - 1)
        superarcIntrinsicWeightPortal.Set(superarc,
                                          contourTree.Arcs.GetNumberOfValues() -
                                            firstVertexForSuperparentPortal.Get(superarc));
      else
        superarcIntrinsicWeightPortal.Set(superarc,
                                          firstVertexForSuperparentPortal.Get(superarc + 1) -
                                            firstVertexForSuperparentPortal.Get(superarc));

    // now initialise the arrays for transfer & dependent weights
    vtkm::cont::ArrayCopy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, contourTree.Superarcs.GetNumberOfValues()),
      superarcDependentWeight);
    vtkm::cont::ArrayCopy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, contourTree.Supernodes.GetNumberOfValues()),
      supernodeTransferWeight);
    vtkm::cont::ArrayCopy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, contourTree.Hyperarcs.GetNumberOfValues()),
      hyperarcDependentWeight);

    // set up the array which tracks which supernodes to deal with on which iteration
    auto firstSupernodePerIterationPortal = contourTree.FirstSupernodePerIteration.ReadPortal();
    auto firstHypernodePerIterationPortal = contourTree.FirstHypernodePerIteration.ReadPortal();
    auto supernodeTransferWeightPortal = supernodeTransferWeight.WritePortal();
    auto superarcDependentWeightPortal = superarcDependentWeight.WritePortal();
    auto hyperarcDependentWeightPortal = hyperarcDependentWeight.WritePortal();

    /*
    vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, nIterations + 1),
                          firstSupernodePerIteration);
    auto firstSupernodePerIterationPortal = firstSupernodePerIteration.WritePortal();
    for (vtkm::Id supernode = 0; supernode < contourTree.Supernodes.GetNumberOfValues();
         supernode++)
    { // per supernode
      vtkm::Id when = MaskedIndex(whenTransferredPortal.Get(supernode));
      if (supernode == 0)
      { // zeroth supernode
        firstSupernodePerIterationPortal.Set(when, supernode);
      } // zeroth supernode
      else if (when != MaskedIndex(whenTransferredPortal.Get(supernode - 1)))
      { // non-matching supernode
        firstSupernodePerIterationPortal.Set(when, supernode);
      } // non-matching supernode
    }   // per supernode
    for (vtkm::Id iteration = 1; iteration < nIterations; ++iteration)
      if (firstSupernodePerIterationPortal.Get(iteration) == 0)
        firstSupernodePerIterationPortal.Set(iteration,
                                             firstSupernodePerIterationPortal.Get(iteration + 1));

    // set the sentinel at the end of the array
    firstSupernodePerIterationPortal.Set(nIterations, contourTree.Supernodes.GetNumberOfValues());

    // now use that array to construct a similar array for hypernodes
    IdArrayType firstHypernodePerIteration;
    firstHypernodePerIteration.Allocate(nIterations + 1);
    auto firstHypernodePerIterationPortal = firstHypernodePerIteration.WritePortal();
    auto supernodeTransferWeightPortal = supernodeTransferWeight.WritePortal();
    auto superarcDependentWeightPortal = superarcDependentWeight.WritePortal();
    auto hyperarcDependentWeightPortal = hyperarcDependentWeight.WritePortal();
    for (vtkm::Id iteration = 0; iteration < nIterations; iteration++)
      firstHypernodePerIterationPortal.Set(
        iteration, hyperparentsPortal.Get(firstSupernodePerIterationPortal.Get(iteration)));
    firstHypernodePerIterationPortal.Set(nIterations, contourTree.Hypernodes.GetNumberOfValues());
    */

    // now iterate, propagating weights inwards
    for (vtkm::Id iteration = 0; iteration < nIterations; iteration++)
    { // per iteration
      // pull the array bounds into register
      vtkm::Id firstSupernode = firstSupernodePerIterationPortal.Get(iteration);
      vtkm::Id lastSupernode = firstSupernodePerIterationPortal.Get(iteration + 1);
      vtkm::Id firstHypernode = firstHypernodePerIterationPortal.Get(iteration);
      vtkm::Id lastHypernode = firstHypernodePerIterationPortal.Get(iteration + 1);

      // Recall that the superarcs are sorted by (iteration, hyperarc), & that all superarcs for a given hyperarc are processed
      // in the same iteration.  Assume therefore that:
      //      i. we now have the intrinsic weight assigned for each superarc, and
      // ii. we also have the transfer weight assigned for each supernode.
      //
      // Suppose we have a sequence of superarcs
      //                      s11 s12 s13 s14 s21 s22 s23 s31
      // with transfer weights at their origins and intrinsic weights along them
      //      sArc                     s11 s12 s13 s14 s21 s22 s23 s31
      //      transfer wt               0   1   2   1   2   3   1   0
      //      intrinsic wt              1   2   1   5   2   6   1   1
      //
      //  now, if we do a prefix sum on each of these and add the two sums together, we get:
      //      sArc                                  s11 s12 s13 s14 s21 s22 s23 s31
      //      hyperparent sNode ID                  s11 s11 s11 s11 s21 s21 s21 s31
      //      transfer weight                       0   1   2   1   2   3   1   0
      //      intrinsic weight                      1   2   1   5   2   6   1   1
      //      sum(xfer + intrinsic)                 1   3   3   6   4   9   2   1
      //  prefix sum (xfer + int)                   1   4   7  13  17  26  28  29
      //  prefix sum (xfer + int - previous hArc)   1   4   7  13  4   13  15  16

      // so, step 1: add xfer + int & store in dependent weight
      for (vtkm::Id supernode = firstSupernode; supernode < lastSupernode; supernode++)
      {
        superarcDependentWeightPortal.Set(supernode,
                                          supernodeTransferWeightPortal.Get(supernode) +
                                            superarcIntrinsicWeightPortal.Get(supernode));
      }

      // step 2: perform prefix sum on the dependent weight range
      for (vtkm::Id supernode = firstSupernode + 1; supernode < lastSupernode; supernode++)
        superarcDependentWeightPortal.Set(supernode,
                                          superarcDependentWeightPortal.Get(supernode) +
                                            superarcDependentWeightPortal.Get(supernode - 1));

      // step 3: subtract out the dependent weight of the prefix to the entire hyperarc. This will be a transfer, but for now, it's easier
      // to show it in serial. NB: Loops backwards so that computation uses the correct value
      // As a bonus, note that we test > firstsupernode, not >=.  This is because we've got unsigned integers, & otherwise it will not terminate
      // But the first is always correct anyway (same reason as the short-cut termination on hyperparent), so we're fine
      for (vtkm::Id supernode = lastSupernode - 1; supernode > firstSupernode; supernode--)
      { // per supernode
        // retrieve the hyperparent & convert to a supernode ID
        vtkm::Id hyperparent = hyperparentsPortal.Get(supernode);
        vtkm::Id hyperparentSuperID = hypernodesPortal.Get(hyperparent);

        // if the hyperparent is the first in the sequence, dependent weight is already correct
        if (hyperparent == firstHypernode)
          continue;

        // otherwise, subtract out the dependent weight *immediately* before the hyperparent's supernode
        superarcDependentWeightPortal.Set(
          supernode,
          superarcDependentWeightPortal.Get(supernode) -
            superarcDependentWeightPortal.Get(hyperparentSuperID - 1));
      } // per supernode

      // step 4: transfer the dependent weight to the hyperarc's target supernode
      for (vtkm::Id hypernode = firstHypernode; hypernode < lastHypernode; hypernode++)
      { // per hypernode
        // last superarc for the hyperarc
        vtkm::Id lastSuperarc;
        // special case for the last hyperarc
        if (hypernode == contourTree.Hypernodes.GetNumberOfValues() - 1)
          // take the last superarc in the array
          lastSuperarc = contourTree.Supernodes.GetNumberOfValues() - 1;
        else
          // otherwise, take the next hypernode's ID and subtract 1
          lastSuperarc = hypernodesPortal.Get(hypernode + 1) - 1;

        // now, given the last superarc for the hyperarc, transfer the dependent weight
        hyperarcDependentWeightPortal.Set(hypernode,
                                          superarcDependentWeightPortal.Get(lastSuperarc));

        // note that in parallel, this will have to be split out as a sort & partial sum in another array
        vtkm::Id hyperarcTarget = MaskedIndex(hyperarcsPortal.Get(hypernode));
        supernodeTransferWeightPortal.Set(hyperarcTarget,
                                          supernodeTransferWeightPortal.Get(hyperarcTarget) +
                                            hyperarcDependentWeightPortal.Get(hypernode));
      } // per hypernode
    }   // per iteration
  }     // ContourTreeMaker::ComputeWeights()

  // routine to compute the branch decomposition by volume
  void static ComputeVolumeBranchDecompositionSerial(const ContourTree& contourTree,
                                                     const IdArrayType& superarcDependentWeight,
                                                     const IdArrayType& superarcIntrinsicWeight,
                                                     IdArrayType& whichBranch,
                                                     IdArrayType& branchMinimum,
                                                     IdArrayType& branchMaximum,
                                                     IdArrayType& branchSaddle,
                                                     IdArrayType& branchParent)
  { // ComputeVolumeBranchDecomposition()
    auto superarcDependentWeightPortal = superarcDependentWeight.ReadPortal();
    auto superarcIntrinsicWeightPortal = superarcIntrinsicWeight.ReadPortal();

    // cache the number of non-root supernodes & superarcs
    vtkm::Id nSupernodes = contourTree.Supernodes.GetNumberOfValues();
    vtkm::Id nSuperarcs = nSupernodes - 1;

    // STAGE I:  Find the upward and downwards weight for each superarc, and set up arrays
    IdArrayType upWeight;
    upWeight.Allocate(nSuperarcs);
    auto upWeightPortal = upWeight.WritePortal();
    IdArrayType downWeight;
    downWeight.Allocate(nSuperarcs);
    auto downWeightPortal = downWeight.WritePortal();
    IdArrayType bestUpward;
    auto noSuchElementArray =
      vtkm::cont::ArrayHandleConstant<vtkm::Id>((vtkm::Id)NO_SUCH_ELEMENT, nSupernodes);
    vtkm::cont::ArrayCopy(noSuchElementArray, bestUpward);
    IdArrayType bestDownward;
    vtkm::cont::ArrayCopy(noSuchElementArray, bestDownward);
    vtkm::cont::ArrayCopy(noSuchElementArray, whichBranch);
    auto bestUpwardPortal = bestUpward.WritePortal();
    auto bestDownwardPortal = bestDownward.WritePortal();

    // STAGE II: Pick the best (largest volume) edge upwards and downwards
    // II A. Pick the best upwards weight by sorting on lower vertex then processing by segments
    // II A 1.  Sort the superarcs by lower vertex
    // II A 2.  Per segment, best superarc writes to the best upwards array
    vtkm::cont::ArrayHandle<EdgePair> superarcList;
    vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleConstant<EdgePair>(EdgePair(-1, -1), nSuperarcs),
                          superarcList);
    auto superarcListWritePortal = superarcList.WritePortal();
    vtkm::Id totalVolume = contourTree.Nodes.GetNumberOfValues();
#ifdef DEBUG_PRINT
    std::cout << "Total Volume: " << totalVolume << std::endl;
#endif
    auto superarcsPortal = contourTree.Superarcs.ReadPortal();

    // NB: Last element in array is guaranteed to be root superarc to infinity,
    // so we can easily skip it by not indexing to the full size
    for (vtkm::Id superarc = 0; superarc < nSuperarcs; superarc++)
    { // per superarc
      if (IsAscending(superarcsPortal.Get(superarc)))
      { // ascending superarc
        superarcListWritePortal.Set(superarc,
                                    EdgePair(superarc, MaskedIndex(superarcsPortal.Get(superarc))));
        upWeightPortal.Set(superarc, superarcDependentWeightPortal.Get(superarc));
        // at the inner end, dependent weight is the total in the subtree.  Then there are vertices along the edge itself (intrinsic weight), including the supernode at the outer end
        // So, to get the "dependent" weight in the other direction, we start with totalVolume - dependent, then subtract (intrinsic - 1)
        downWeightPortal.Set(superarc,
                             (totalVolume - superarcDependentWeightPortal.Get(superarc)) +
                               (superarcIntrinsicWeightPortal.Get(superarc) - 1));
      } // ascending superarc
      else
      { // descending superarc
        superarcListWritePortal.Set(superarc,
                                    EdgePair(MaskedIndex(superarcsPortal.Get(superarc)), superarc));
        downWeightPortal.Set(superarc, superarcDependentWeightPortal.Get(superarc));
        // at the inner end, dependent weight is the total in the subtree.  Then there are vertices along the edge itself (intrinsic weight), including the supernode at the outer end
        // So, to get the "dependent" weight in the other direction, we start with totalVolume - dependent, then subtract (intrinsic - 1)
        upWeightPortal.Set(superarc,
                           (totalVolume - superarcDependentWeightPortal.Get(superarc)) +
                             (superarcIntrinsicWeightPortal.Get(superarc) - 1));
      } // descending superarc
    }   // per superarc

#ifdef DEBUG_PRINT
    std::cout << "II A. Weights Computed" << std::endl;
    PrintHeader(upWeight.GetNumberOfValues());
    //PrintIndices("Intrinsic Weight", superarcIntrinsicWeight);
    //PrintIndices("Dependent Weight", superarcDependentWeight);
    PrintIndices("Upwards Weight", upWeight);
    PrintIndices("Downwards Weight", downWeight);
    std::cout << std::endl;
#endif

    // II B. Pick the best downwards weight by sorting on upper vertex then processing by segments
    // II B 1.      Sort the superarcs by upper vertex
    IdArrayType superarcSorter;
    superarcSorter.Allocate(nSuperarcs);
    auto superarcSorterPortal = superarcSorter.WritePortal();
    for (vtkm::Id superarc = 0; superarc < nSuperarcs; superarc++)
      superarcSorterPortal.Set(superarc, superarc);

    vtkm::cont::Algorithm::Sort(
      superarcSorter,
      process_contourtree_inc_ns::SuperArcVolumetricComparator(upWeight, superarcList, false));

    // Initialize after in-place sort algorithm. (Kokkos)
    auto superarcSorterReadPortal = superarcSorter.ReadPortal();

    // II B 2.  Per segment, best superarc writes to the best upward array
    for (vtkm::Id superarc = 0; superarc < nSuperarcs; superarc++)
    { // per superarc
      vtkm::Id superarcID = superarcSorterReadPortal.Get(superarc);
      const EdgePair& edge = superarcListWritePortal.Get(superarcID);
      // if it's the last one
      if (superarc == nSuperarcs - 1)
        bestDownwardPortal.Set(edge.second, edge.first);
      else
      { // not the last one
        const EdgePair& nextEdge =
          superarcListWritePortal.Get(superarcSorterReadPortal.Get(superarc + 1));
        // if the next edge belongs to another, we're the highest
        if (nextEdge.second != edge.second)
          bestDownwardPortal.Set(edge.second, edge.first);
      } // not the last one
    }   // per superarc

    // II B 3.  Repeat for lower vertex
    vtkm::cont::Algorithm::Sort(
      superarcSorter,
      process_contourtree_inc_ns::SuperArcVolumetricComparator(downWeight, superarcList, true));

    // Re-initialize after in-place sort algorithm. (Kokkos)
    superarcSorterReadPortal = superarcSorter.ReadPortal();

    // II B 2.  Per segment, best superarc writes to the best upward array
    for (vtkm::Id superarc = 0; superarc < nSuperarcs; superarc++)
    { // per superarc
      vtkm::Id superarcID = superarcSorterReadPortal.Get(superarc);
      const EdgePair& edge = superarcListWritePortal.Get(superarcID);
      // if it's the last one
      if (superarc == nSuperarcs - 1)
        bestUpwardPortal.Set(edge.first, edge.second);
      else
      { // not the last one
        const EdgePair& nextEdge =
          superarcListWritePortal.Get(superarcSorterReadPortal.Get(superarc + 1));
        // if the next edge belongs to another, we're the highest
        if (nextEdge.first != edge.first)
          bestUpwardPortal.Set(edge.first, edge.second);
      } // not the last one
    }   // per superarc

#ifdef DEBUG_PRINT
    std::cout << "II. Best Edges Selected" << std::endl;
    PrintHeader(bestUpward.GetNumberOfValues());
    PrintIndices("Best Upwards", bestUpward);
    PrintIndices("Best Downwards", bestDownward);
    std::cout << std::endl;
#endif

    ProcessContourTree::ComputeBranchData(contourTree,
                                          whichBranch,
                                          branchMinimum,
                                          branchMaximum,
                                          branchSaddle,
                                          branchParent,
                                          bestUpward,
                                          bestDownward);
  }

  // routine to compute the branch decomposition by volume
  void static ComputeBranchData(const ContourTree& contourTree,
                                IdArrayType& whichBranch,
                                IdArrayType& branchMinimum,
                                IdArrayType& branchMaximum,
                                IdArrayType& branchSaddle,
                                IdArrayType& branchParent,
                                IdArrayType& bestUpward,
                                IdArrayType& bestDownward)
  { // ComputeBranchData()

    // Set up constants
    vtkm::Id nSupernodes = contourTree.Supernodes.GetNumberOfValues();
    auto noSuchElementArray =
      vtkm::cont::ArrayHandleConstant<vtkm::Id>((vtkm::Id)NO_SUCH_ELEMENT, nSupernodes);
    vtkm::cont::ArrayCopy(noSuchElementArray, whichBranch);

    // STAGE III: For each vertex, identify which neighbours are on same branch
    // Let v = BestUp(u). Then if u = BestDown(v), copy BestUp(u) to whichBranch(u)
    // Otherwise, let whichBranch(u) = BestUp(u) | TERMINAL to mark the end of the side branch
    // NB 1: Leaves already have the flag set, but it's redundant so its safe
    // NB 2: We don't need to do it downwards because it's symmetric
    vtkm::cont::Invoker invoke;
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::PropagateBestUpDown
      propagateBestUpDownWorklet;
    invoke(propagateBestUpDownWorklet, bestUpward, bestDownward, whichBranch);

#ifdef DEBUG_PRINT
    std::cout << "III. Branch Neighbours Identified" << std::endl;
    PrintHeader(whichBranch.GetNumberOfValues());
    PrintIndices("Which Branch", whichBranch);
    std::cout << std::endl;
#endif

    // STAGE IV: Use pointer-doubling on whichBranch to propagate branches
    // Compute the number of log steps required in this pass
    vtkm::Id numLogSteps = 1;
    for (vtkm::Id shifter = nSupernodes; shifter != 0; shifter >>= 1)
      numLogSteps++;

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::PointerDoubling pointerDoubling(
      nSupernodes);

    // use pointer-doubling to build the branches
    for (vtkm::Id iteration = 0; iteration < numLogSteps; iteration++)
    { // per iteration
      invoke(pointerDoubling, whichBranch);
    } // per iteration


#ifdef DEBUG_PRINT
    std::cout << "IV. Branch Chains Propagated" << std::endl;
    PrintHeader(whichBranch.GetNumberOfValues());
    PrintIndices("Which Branch", whichBranch);
    std::cout << std::endl;
#endif

    // Initialise
    IdArrayType chainToBranch;
    vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, nSupernodes), chainToBranch);

    // Set 1 to every relevant
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::PrepareChainToBranch
      prepareChainToBranchWorklet;
    invoke(prepareChainToBranchWorklet, whichBranch, chainToBranch);

    // Prefix scanto get IDs
    vtkm::Id nBranches = vtkm::cont::Algorithm::ScanInclusive(chainToBranch, chainToBranch);

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::FinaliseChainToBranch
      finaliseChainToBranchWorklet;
    invoke(finaliseChainToBranchWorklet, whichBranch, chainToBranch);

    // V B.  Create the arrays for the branches
    auto noSuchElementArrayNBranches =
      vtkm::cont::ArrayHandleConstant<vtkm::Id>((vtkm::Id)NO_SUCH_ELEMENT, nBranches);
    vtkm::cont::ArrayCopy(noSuchElementArrayNBranches, branchMinimum);
    vtkm::cont::ArrayCopy(noSuchElementArrayNBranches, branchMaximum);
    vtkm::cont::ArrayCopy(noSuchElementArrayNBranches, branchSaddle);
    vtkm::cont::ArrayCopy(noSuchElementArrayNBranches, branchParent);

#ifdef DEBUG_PRINT
    std::cout << "V. Branch Arrays Created" << std::endl;
    PrintHeader(chainToBranch.GetNumberOfValues());
    PrintIndices("Chain To Branch", chainToBranch);
    PrintHeader(nBranches);
    PrintIndices("Branch Minimum", branchMinimum);
    PrintIndices("Branch Maximum", branchMaximum);
    PrintIndices("Branch Saddle", branchSaddle);
    PrintIndices("Branch Parent", branchParent);
#endif

    IdArrayType supernodeSorter;
    vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleIndex(nSupernodes), supernodeSorter);

    vtkm::cont::Algorithm::Sort(
      supernodeSorter,
      process_contourtree_inc_ns::SuperNodeBranchComparator(whichBranch, contourTree.Supernodes));

    IdArrayType permutedBranches;
    permutedBranches.Allocate(nSupernodes);
    PermuteArray<vtkm::Id>(whichBranch, supernodeSorter, permutedBranches);

    IdArrayType permutedRegularID;
    permutedRegularID.Allocate(nSupernodes);
    PermuteArray<vtkm::Id>(contourTree.Supernodes, supernodeSorter, permutedRegularID);

#ifdef DEBUG_PRINT
    std::cout << "VI A. Sorted into Branches" << std::endl;
    PrintHeader(nSupernodes);
    PrintIndices("Supernode IDs", supernodeSorter);
    PrintIndices("Branch", permutedBranches);
    PrintIndices("Regular ID", permutedRegularID);
#endif

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::WhichBranchNewId
      whichBranchNewIdWorklet;
    invoke(whichBranchNewIdWorklet, chainToBranch, whichBranch);

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::BranchMinMaxSet
      branchMinMaxSetWorklet(nSupernodes);
    invoke(branchMinMaxSetWorklet, supernodeSorter, whichBranch, branchMinimum, branchMaximum);

#ifdef DEBUG_PRINT
    std::cout << "VI. Branches Set" << std::endl;
    PrintHeader(nBranches);
    PrintIndices("Branch Maximum", branchMaximum);
    PrintIndices("Branch Minimum", branchMinimum);
    PrintIndices("Branch Saddle", branchSaddle);
    PrintIndices("Branch Parent", branchParent);
#endif

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::BranchSaddleParentSet
      branchSaddleParentSetWorklet;
    invoke(branchSaddleParentSetWorklet,
           whichBranch,
           branchMinimum,
           branchMaximum,
           bestDownward,
           bestUpward,
           branchSaddle,
           branchParent);

#ifdef DEBUG_PRINT
    std::cout << "VII. Branches Constructed" << std::endl;
    PrintHeader(nBranches);
    PrintIndices("Branch Maximum", branchMaximum);
    PrintIndices("Branch Minimum", branchMinimum);
    PrintIndices("Branch Saddle", branchSaddle);
    PrintIndices("Branch Parent", branchParent);
#endif

  } // ComputeBranchData()

  // Create branch decomposition from contour tree
  template <typename T, typename StorageType>
  static process_contourtree_inc_ns::Branch<T>* ComputeBranchDecomposition(
    const IdArrayType& contourTreeSuperparents,
    const IdArrayType& contourTreeSupernodes,
    const IdArrayType& whichBranch,
    const IdArrayType& branchMinimum,
    const IdArrayType& branchMaximum,
    const IdArrayType& branchSaddle,
    const IdArrayType& branchParent,
    const IdArrayType& sortOrder,
    const vtkm::cont::ArrayHandle<T, StorageType>& dataField,
    bool dataFieldIsSorted)
  {
    return process_contourtree_inc_ns::Branch<T>::ComputeBranchDecomposition(
      contourTreeSuperparents,
      contourTreeSupernodes,
      whichBranch,
      branchMinimum,
      branchMaximum,
      branchSaddle,
      branchParent,
      sortOrder,
      dataField,
      dataFieldIsSorted);
  }
  
  
  
  
  
  
  
  
  
  
  
  
    // Create branch decomposition from contour tree
  template <typename T, typename StorageType>
  static process_contourtree_inc_ns::Branch<T>* ComputeBranchDecomposition(
    const IdArrayType& contourTreeSuperparents,
    const IdArrayType& contourTreeSupernodes,
          const IdArrayType& contourTreeSuperarcs,
    const IdArrayType& whichBranch,
    const IdArrayType& branchMinimum,
    const IdArrayType& branchMaximum,
    const IdArrayType& branchSaddle,
    const IdArrayType& branchParent,
    const IdArrayType& sortOrder,
          const vtkm::cont::ArrayHandle<T, StorageType>& valueField,
    const vtkm::cont::ArrayHandle<T, StorageType>& dataField,
    bool dataFieldIsSorted,
    const IdArrayType& superarcIntrinsicWeight,
    const IdArrayType& superarcDependentWeight,            // NEW: passed intrincid
          const vtkm::Id& contourTreeRootnode,                // NEW: used to get the augmented betti nodes (which are past the root node in index)
          const IdArrayType& contourTreeSupernodeBetti,       // NEW: added Supernode-Betti number mappings after implementing Betti augmentation
          std::vector<process_contourtree_inc_ns::Branch<T>*>& branches) // output
  {
    std::cout << "ContourTreeApp->(ProcessContourTree)->Branch.h->ComputeBranchDecomposition()" << std::endl;
    // Branch double-call (Branch function called from here, not just via ctaug_ns::ProcessContourTree::ComputeBranchDecomposition<ValueType>)
    return process_contourtree_inc_ns::Branch<T>::ComputeBranchDecomposition(
      contourTreeSuperparents,
      contourTreeSupernodes,
                contourTreeSuperarcs,
      whichBranch,
      branchMinimum,
      branchMaximum,
      branchSaddle,
      branchParent,
      sortOrder,
                valueField,
      dataField,
      dataFieldIsSorted,
      superarcIntrinsicWeight,
      superarcDependentWeight,
                contourTreeRootnode,
                contourTreeSupernodeBetti,
                branches);
  }
  
  
  
  
  
  
  
  

  // routine to compute the branch decomposition by volume
  void static ComputeVolumeBranchDecomposition(const ContourTree& contourTree,
                                               const vtkm::Id nIterations,
                                               IdArrayType& whichBranch,
                                               IdArrayType& branchMinimum,
                                               IdArrayType& branchMaximum,
                                               IdArrayType& branchSaddle,
                                               IdArrayType& branchParent)
  { // ComputeHeightBranchDecomposition()

    vtkm::cont::Invoker Invoke;

    // STEP 1. Compute the number of nodes in every superarc, that's the intrinsic weight
    IdArrayType superarcIntrinsicWeight;
    superarcIntrinsicWeight.Allocate(contourTree.Superarcs.GetNumberOfValues());

    IdArrayType firstVertexForSuperparent;
    firstVertexForSuperparent.Allocate(contourTree.Superarcs.GetNumberOfValues());

    // Compute the number of regular nodes on every superarcs (the intrinsic weight)
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::SetFirstVertexForSuperparent
      setFirstVertexForSuperparent;
    Invoke(setFirstVertexForSuperparent,
           contourTree.Nodes,
           contourTree.Superparents,
           firstVertexForSuperparent);

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::ComputeIntrinsicWeight
      computeIntrinsicWeight;
    Invoke(computeIntrinsicWeight,
           contourTree.Arcs,
           contourTree.Superarcs,
           firstVertexForSuperparent,
           superarcIntrinsicWeight);


    // Cache the number of non-root supernodes & superarcs
    vtkm::Id nSupernodes = contourTree.Supernodes.GetNumberOfValues();
    auto noSuchElementArray =
      vtkm::cont::ArrayHandleConstant<vtkm::Id>((vtkm::Id)NO_SUCH_ELEMENT, nSupernodes);

    // Set up bestUpward and bestDownward array, these are the things we want to compute in this routine.
    IdArrayType bestUpward, bestDownward;
    vtkm::cont::ArrayCopy(noSuchElementArray, bestUpward);
    vtkm::cont::ArrayCopy(noSuchElementArray, bestDownward);

    // We initiale with the weight of the superarcs, once we sum those up we'll get the hypersweep weight
    IdArrayType sumValues;
    vtkm::cont::ArrayCopy(superarcIntrinsicWeight, sumValues);

    // This should be 0 here, because we're not changing the root
    vtkm::cont::ArrayHandle<vtkm::Id> howManyUsed;
    vtkm::cont::ArrayCopy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, contourTree.Hyperarcs.GetNumberOfValues()),
      howManyUsed);

    // Perform a sum hypersweep
    hyperarcScan<decltype(vtkm::Sum())>(contourTree.Supernodes,
                                        contourTree.Hypernodes,
                                        contourTree.Hyperarcs,
                                        contourTree.Hyperparents,
                                        contourTree.Hyperparents,
                                        contourTree.WhenTransferred,
                                        howManyUsed,
                                        nIterations,
                                        vtkm::Sum(),
                                        sumValues);

    // For every directed arc store the volume of it's associate subtree
    vtkm::cont::ArrayHandle<vtkm::worklet::contourtree_augmented::EdgeDataVolume> arcs;
    arcs.Allocate(contourTree.Superarcs.GetNumberOfValues() * 2 - 2);

    vtkm::Id totalVolume = contourTree.Nodes.GetNumberOfValues();
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::InitialiseArcsVolume initArcs(
      totalVolume);
    Invoke(initArcs, sumValues, superarcIntrinsicWeight, contourTree.Superarcs, arcs);

    // Sort arcs to obtain the best up and down
    vtkm::cont::Algorithm::Sort(arcs, vtkm::SortLess());

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::SetBestUpDown setBestUpDown;
    Invoke(setBestUpDown, bestUpward, bestDownward, arcs);

    ProcessContourTree::ComputeBranchData(contourTree,
                                          whichBranch,
                                          branchMinimum,
                                          branchMaximum,
                                          branchSaddle,
                                          branchParent,
                                          bestUpward,
                                          bestDownward);

  } // ComputeHeightBranchDecomposition()

  // routine to compute the branch decomposition by volume
  void static ComputeHeightBranchDecomposition(const ContourTree& contourTree,
                                               const cont::ArrayHandle<Float64> fieldValues,
                                               const IdArrayType& ctSortOrder,
                                               const vtkm::Id nIterations,
                                               IdArrayType& whichBranch,
                                               IdArrayType& branchMinimum,
                                               IdArrayType& branchMaximum,
                                               IdArrayType& branchSaddle,
                                               IdArrayType& branchParent)
  { // ComputeHeightBranchDecomposition()

    // Cache the number of non-root supernodes & superarcs
    vtkm::Id nSupernodes = contourTree.Supernodes.GetNumberOfValues();
    auto noSuchElementArray =
      vtkm::cont::ArrayHandleConstant<vtkm::Id>((vtkm::Id)NO_SUCH_ELEMENT, nSupernodes);

    // Set up bestUpward and bestDownward array, these are the things we want to compute in this routine.
    IdArrayType bestUpward, bestDownward;
    vtkm::cont::ArrayCopy(noSuchElementArray, bestUpward);
    vtkm::cont::ArrayCopy(noSuchElementArray, bestDownward);

    // maxValues and minValues store the values from the max and min hypersweep respectively.
    IdArrayType minValues, maxValues;
    vtkm::cont::ArrayCopy(contourTree.Supernodes, maxValues);
    vtkm::cont::ArrayCopy(contourTree.Supernodes, minValues);

    // Store the direction of the superarcs in the min and max hypersweep (find a way to get rid of these, the only differing direction is on the path from the root to the min/max).
    IdArrayType minParents, maxParents;
    vtkm::cont::ArrayCopy(contourTree.Superarcs, minParents);
    vtkm::cont::ArrayCopy(contourTree.Superarcs, maxParents);

    auto minParentsPortal = minParents.WritePortal();
    auto maxParentsPortal = maxParents.WritePortal();

    // Cache the glonal minimum and global maximum (these will be the roots in the min and max hypersweep)
    Id minSuperNode = MaskedIndex(contourTree.Superparents.ReadPortal().Get(0));
    Id maxSuperNode = MaskedIndex(
      contourTree.Superparents.ReadPortal().Get(contourTree.Nodes.GetNumberOfValues() - 1));

    // Find the path from the global minimum to the root, not parallelisable (but it's fast, no need to parallelise)
    auto minPath = findSuperPathToRoot(contourTree.Superarcs.ReadPortal(), minSuperNode);

    // Find the path from the global minimum to the root, not parallelisable (but it's fast, no need to parallelise)
    auto maxPath = findSuperPathToRoot(contourTree.Superarcs.ReadPortal(), maxSuperNode);

    // Reserve the direction of the superarcs on the min path.
    for (std::size_t i = 1; i < minPath.size(); i++)
    {
      minParentsPortal.Set(minPath[i], minPath[i - 1]);
    }
    minParentsPortal.Set(minPath[0], 0);

    // Reserve the direction of the superarcs on the max path.
    for (std::size_t i = 1; i < maxPath.size(); i++)
    {
      maxParentsPortal.Set(maxPath[i], maxPath[i - 1]);
    }
    maxParentsPortal.Set(maxPath[0], 0);

    vtkm::cont::Invoker Invoke;
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::UnmaskArray unmaskArrayWorklet;
    Invoke(unmaskArrayWorklet, minValues);
    Invoke(unmaskArrayWorklet, maxValues);

    // Thse arrays hold the changes hyperarcs in the min and max hypersweep respectively
    vtkm::cont::ArrayHandle<vtkm::Id> minHyperarcs, maxHyperarcs;
    vtkm::cont::ArrayCopy(contourTree.Hyperarcs, minHyperarcs);
    vtkm::cont::ArrayCopy(contourTree.Hyperarcs, maxHyperarcs);

    // These arrays hold the changed hyperarcs for the min and max hypersweep
    vtkm::cont::ArrayHandle<vtkm::Id> minHyperparents, maxHyperparents;
    vtkm::cont::ArrayCopy(contourTree.Hyperparents, minHyperparents);
    vtkm::cont::ArrayCopy(contourTree.Hyperparents, maxHyperparents);

    auto minHyperparentsPortal = minHyperparents.WritePortal();
    auto maxHyperparentsPortal = maxHyperparents.WritePortal();

    for (std::size_t i = 0; i < minPath.size(); i++)
    {
      // Set a unique dummy Id (something that the prefix scan by key will leave alone)
      minHyperparentsPortal.Set(minPath[i],
                                contourTree.Hypernodes.GetNumberOfValues() + minPath[i]);
    }

    for (std::size_t i = 0; i < maxPath.size(); i++)
    {
      // Set a unique dummy Id (something that the prefix scan by key will leave alone)
      maxHyperparentsPortal.Set(maxPath[i],
                                contourTree.Hypernodes.GetNumberOfValues() + maxPath[i]);
    }

    // These arrays hold the number of nodes in each hypearcs that are on the min or max path for the min and max hypersweep respectively.
    vtkm::cont::ArrayHandle<vtkm::Id> minHowManyUsed, maxHowManyUsed;
    vtkm::cont::ArrayCopy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, maxHyperarcs.GetNumberOfValues()),
      minHowManyUsed);
    vtkm::cont::ArrayCopy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, maxHyperarcs.GetNumberOfValues()),
      maxHowManyUsed);

    // Min Hypersweep
    const auto minOperator = vtkm::Minimum();

    // Cut hyperarcs at the first node on the path from the max to the root
    editHyperarcs(contourTree.Hyperparents.ReadPortal(),
                  minPath,
                  minHyperarcs.WritePortal(),
                  minHowManyUsed.WritePortal());

    // Perform an ordinary hypersweep on those new hyperarcs
    hyperarcScan<decltype(vtkm::Minimum())>(contourTree.Supernodes,
                                            contourTree.Hypernodes,
                                            minHyperarcs,
                                            contourTree.Hyperparents,
                                            minHyperparents,
                                            contourTree.WhenTransferred,
                                            minHowManyUsed,
                                            nIterations,
                                            vtkm::Minimum(),
                                            minValues);

    // Prefix sum along the path from the min to the root
    fixPath(vtkm::Minimum(), minPath, minValues.WritePortal());

    // Max Hypersweep
    const auto maxOperator = vtkm::Maximum();

    // Cut hyperarcs at the first node on the path from the max to the root
    editHyperarcs(contourTree.Hyperparents.ReadPortal(),
                  maxPath,
                  maxHyperarcs.WritePortal(),
                  maxHowManyUsed.WritePortal());

    // Perform an ordinary hypersweep on those new hyperarcs
    hyperarcScan<decltype(vtkm::Maximum())>(contourTree.Supernodes,
                                            contourTree.Hypernodes,
                                            maxHyperarcs,
                                            contourTree.Hyperparents,
                                            maxHyperparents,
                                            contourTree.WhenTransferred,
                                            maxHowManyUsed,
                                            nIterations,
                                            vtkm::Maximum(),
                                            maxValues);

    // Prefix sum along the path from the max to the root
    fixPath(vtkm::Maximum(), maxPath, maxValues.WritePortal());

    // For every directed edge (a, b) consider that subtree who's root is b and does not contain a.
    // We have so far found the min and max in all sub subtrees, now we compare those to a and incorporate a into that.
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::IncorporateParent<decltype(
      vtkm::Minimum())>
      incorporateParentMinimumWorklet(minOperator);
    Invoke(incorporateParentMinimumWorklet, minParents, contourTree.Supernodes, minValues);

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::IncorporateParent<decltype(
      vtkm::Maximum())>
      incorporateParentMaximumWorklet(maxOperator);
    Invoke(incorporateParentMaximumWorklet, maxParents, contourTree.Supernodes, maxValues);

    // Initialise all directed superarcs in the contour tree. Those will correspond to subtrees whos height we need for the branch decomposition.
    vtkm::cont::ArrayHandle<vtkm::worklet::contourtree_augmented::EdgeDataHeight> arcs;
    arcs.Allocate(contourTree.Superarcs.GetNumberOfValues() * 2 - 2);

    vtkm::worklet::contourtree_augmented::process_contourtree_inc::InitialiseArcs initArcs(
      0, contourTree.Arcs.GetNumberOfValues() - 1, minPath[minPath.size() - 1]);

    Invoke(initArcs, minParents, maxParents, minValues, maxValues, contourTree.Superarcs, arcs);

    // Use the min & max to compute the height of all subtrees
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::ComputeSubtreeHeight
      computeSubtreeHeight;
    Invoke(computeSubtreeHeight, fieldValues, ctSortOrder, contourTree.Supernodes, arcs);

    // Sort all directed edges based on the height of their subtree
    vtkm::cont::Algorithm::Sort(arcs, vtkm::SortLess());

    // Select a best up and best down neighbour for every vertex in the contour tree using heights of all subtrees
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::SetBestUpDown setBestUpDown;
    Invoke(setBestUpDown, bestUpward, bestDownward, arcs);

    // Having computed the bestUp/Down we can propagte those to obtain the branches of the branch decomposition
    ProcessContourTree::ComputeBranchData(contourTree,
                                          whichBranch,
                                          branchMinimum,
                                          branchMaximum,
                                          branchSaddle,
                                          branchParent,
                                          bestUpward,
                                          bestDownward);

  } // ComputeHeightBranchDecomposition()

  std::vector<Id> static findSuperPathToRoot(
    vtkm::cont::ArrayHandle<vtkm::Id>::ReadPortalType parentsPortal,
    vtkm::Id vertex)
  {
    // Initialise the empty path and starting vertex
    std::vector<vtkm::Id> path;
    vtkm::Id current = vertex;

    // Go up the parent list until we reach the root
    while (MaskedIndex(parentsPortal.Get(current)) != 0)
    {
      path.push_back(current);
      current = MaskedIndex(parentsPortal.Get(current));
    }
    path.push_back(current);

    return path;
  }

  // Given a path from a leaf (the global min/max) to the root of the contour tree and a hypersweep (where all hyperarcs are cut at the path)
  // This function performs a prefix scan along that path to obtain the correct hypersweep values (as in the global min/max is the root of the hypersweep)
  void static fixPath(const std::function<vtkm::Id(vtkm::Id, vtkm::Id)> operation,
                      const std::vector<vtkm::Id> path,
                      vtkm::cont::ArrayHandle<vtkm::Id>::WritePortalType minMaxIndex)
  {
    using vtkm::worklet::contourtree_augmented::MaskedIndex;

    // Fix path from the old root to the new root. Parallelisble with a prefix scan, but sufficiently fast for now.
    for (auto i = path.size() - 2; i > 0; i--)
    {
      const auto vertex = path[i + 1];
      const auto parent = path[i];

      const auto vertexValue = minMaxIndex.Get(vertex);
      const auto parentValue = minMaxIndex.Get(parent);

      minMaxIndex.Set(parent, operation(vertexValue, parentValue));
    }
  }

  // This function edits all the hyperarcs which contain vertices which are on the supplied path. This path is usually the path between the global min/max to the root of the tree.
  // This function effectively cuts hyperarcs at the first node they encounter along that path.
  // In addition to this it computed the number of supernodes every hyperarc has on that path. This helps in the function hyperarcScan for choosing the new target of the cut hyperarcs.
  // NOTE: It is assumed that the supplied path starts at a leaf and ends at the root of the contour tree.
  void static editHyperarcs(
    const vtkm::cont::ArrayHandle<vtkm::Id>::ReadPortalType hyperparentsPortal,
    const std::vector<vtkm::Id> path,
    vtkm::cont::ArrayHandle<vtkm::Id>::WritePortalType hyperarcsPortal,
    vtkm::cont::ArrayHandle<vtkm::Id>::WritePortalType howManyUsedPortal)
  {
    using vtkm::worklet::contourtree_augmented::MaskedIndex;

    std::size_t i = 0;
    while (i < path.size())
    {
      // Cut the hyperacs at the first point
      hyperarcsPortal.Set(MaskedIndex(hyperparentsPortal.Get(path[i])), path[i]);

      Id currentHyperparent = MaskedIndex(hyperparentsPortal.Get(path[i]));

      // Skip the rest of the supernodes which are on the same hyperarc
      while (i < path.size() && MaskedIndex(hyperparentsPortal.Get(path[i])) == currentHyperparent)
      {
        const auto value = howManyUsedPortal.Get(MaskedIndex(hyperparentsPortal.Get(path[i])));
        howManyUsedPortal.Set(MaskedIndex(hyperparentsPortal.Get(path[i])), value + 1);
        i++;
      }
    }
  }

  template <class BinaryFunctor>
  void static hyperarcScan(const vtkm::cont::ArrayHandle<vtkm::Id> supernodes,
                           const vtkm::cont::ArrayHandle<vtkm::Id> hypernodes,
                           const vtkm::cont::ArrayHandle<vtkm::Id> hyperarcs,
                           const vtkm::cont::ArrayHandle<vtkm::Id> hyperparents,
                           const vtkm::cont::ArrayHandle<vtkm::Id> hyperparentKeys,
                           const vtkm::cont::ArrayHandle<vtkm::Id> whenTransferred,
                           const vtkm::cont::ArrayHandle<vtkm::Id> howManyUsed,
                           const vtkm::Id nIterations,
                           const BinaryFunctor operation,
                           vtkm::cont::ArrayHandle<vtkm::Id> minMaxIndex)
  {
    using vtkm::worklet::contourtree_augmented::MaskedIndex;

    vtkm::cont::Invoker invoke;

    auto supernodesPortal = supernodes.ReadPortal();
    auto hypernodesPortal = hypernodes.ReadPortal();
    auto hyperparentsPortal = hyperparents.ReadPortal();

    // Set the first supernode per iteration
    vtkm::cont::ArrayHandle<vtkm::Id> firstSupernodePerIteration;
    vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, nIterations + 1),
                          firstSupernodePerIteration);

    // The first different from the previous is the first in the iteration
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::SetFirstSupernodePerIteration
      setFirstSupernodePerIteration;
    invoke(setFirstSupernodePerIteration, whenTransferred, firstSupernodePerIteration);

    auto firstSupernodePerIterationPortal = firstSupernodePerIteration.WritePortal();
    for (vtkm::Id iteration = 1; iteration < nIterations; ++iteration)
    {
      if (firstSupernodePerIterationPortal.Get(iteration) == 0)
      {
        firstSupernodePerIterationPortal.Set(iteration,
                                             firstSupernodePerIterationPortal.Get(iteration + 1));
      }
    }

    // set the sentinel at the end of the array
    firstSupernodePerIterationPortal.Set(nIterations, supernodesPortal.GetNumberOfValues());

    // Set the first hypernode per iteration
    vtkm::cont::ArrayHandle<vtkm::Id> firstHypernodePerIteration;
    vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleConstant<vtkm::Id>(0, nIterations + 1),
                          firstHypernodePerIteration);
    auto firstHypernodePerIterationPortal = firstHypernodePerIteration.WritePortal();

    for (vtkm::Id iteration = 0; iteration < nIterations; iteration++)
    {
      firstHypernodePerIterationPortal.Set(
        iteration, hyperparentsPortal.Get(firstSupernodePerIterationPortal.Get(iteration)));
    }

    // Set the sentinel at the end of the array
    firstHypernodePerIterationPortal.Set(nIterations, hypernodesPortal.GetNumberOfValues());

    // This workled is used in every iteration of the following loop, so it's initialised outside.
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::AddDependentWeightHypersweep<
      BinaryFunctor>
      addDependentWeightHypersweepWorklet(operation);

    // For every iteration do a prefix scan on all hyperarcs in that iteration and then transfer the scanned value to every hyperarc's target supernode
    for (vtkm::Id iteration = 0; iteration < nIterations; iteration++)
    {
      // Determine the first and last hypernode in the current iteration (all hypernodes between them are also in the current iteration)
      vtkm::Id firstHypernode = firstHypernodePerIterationPortal.Get(iteration);
      vtkm::Id lastHypernode = firstHypernodePerIterationPortal.Get(iteration + 1);
      lastHypernode = vtkm::Minimum()(lastHypernode, hypernodes.GetNumberOfValues() - 1);

      // Determine the first and last supernode in the current iteration (all supernode between them are also in the current iteration)
      vtkm::Id firstSupernode = MaskedIndex(hypernodesPortal.Get(firstHypernode));
      vtkm::Id lastSupernode = MaskedIndex(hypernodesPortal.Get(lastHypernode));
      lastSupernode = vtkm::Minimum()(lastSupernode, hyperparents.GetNumberOfValues() - 1);

      // Prefix scan along all hyperarcs in the current iteration
      auto subarrayValues = vtkm::cont::make_ArrayHandleView(
        minMaxIndex, firstSupernode, lastSupernode - firstSupernode);
      auto subarrayKeys = vtkm::cont::make_ArrayHandleView(
        hyperparentKeys, firstSupernode, lastSupernode - firstSupernode);
      vtkm::cont::Algorithm::ScanInclusiveByKey(
        subarrayKeys, subarrayValues, subarrayValues, operation);

      // Array containing the Ids of the hyperarcs in the current iteration
      vtkm::cont::ArrayHandleCounting<vtkm::Id> iterationHyperarcs(
        firstHypernode, 1, lastHypernode - firstHypernode);

      // Transfer the value accumulated in the last entry of the prefix scan to the hypernode's targe supernode
      invoke(addDependentWeightHypersweepWorklet,
             iterationHyperarcs,
             hypernodes,
             hyperarcs,
             howManyUsed,
             minMaxIndex);
    }
  }
}; // class ProcessContourTree
} // namespace contourtree_augmented
} // worklet
} // vtkm

#endif
