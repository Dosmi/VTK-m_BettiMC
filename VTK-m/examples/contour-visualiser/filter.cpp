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

#include "./filter.h"

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/SetTriangleSuperarcId.h>

using namespace std;
using namespace vtkm;

constexpr vtkm::Id INDEX_MASK = std::numeric_limits<vtkm::Id>::max() / 16;


void cv1k::filter::computeTriangleIds(worklet::contourtree_augmented::ContourTree contourTree,
                                      worklet::contourtree_augmented::DataSetMesh mesh,
                                      worklet::contourtree_augmented::MeshExtrema extrema,
                                      cont::ArrayHandle<Float64> fieldArray,
                                      vtkm::cont::ArrayHandle<cv1k::Triangle> triangles,
                                      Float64 isovalue)
{

    // Each point has it's own isovalue
    cont::ArrayHandle<Float64> isovalueArray;
    isovalueArray.Allocate(triangles.GetNumberOfValues());

    // Endpoints of path in the contour tree
    cont::ArrayHandle<Vec<Id, 2>> endpoints;
    endpoints.Allocate(triangles.GetNumberOfValues());

    for(int i = 0 ; i < triangles.GetNumberOfValues() ; i++)
    {
        isovalueArray.WritePortal().Set(i, isovalue);
        endpoints.WritePortal().Set(i, triangles.ReadPortal().Get(i).representativeEdge);
    }

    // Output array that will store the superarc on the path from endpoint[0] to endpoint[1] at an isovalue
    cont::ArrayHandle<Id> superarcIds;
    superarcIds.Allocate(endpoints.GetNumberOfValues());

    // Set up the worklet
    vtkm::worklet::contourtree_augmented::process_contourtree_inc::SetTriangleSuperarcId setTrianglesId(contourTree.Hypernodes.GetNumberOfValues(),
                                                                                                        contourTree.Supernodes.GetNumberOfValues());
    cont::Invoker Invoke;

    // Run the worklet
    Invoke(
            setTrianglesId,
            endpoints,
            fieldArray,
            isovalueArray,
            mesh.SortOrder, // (input)
            mesh.SortIndices, // (input)
            contourTree.Superparents, // (input)
            contourTree.WhenTransferred, // (input)
            contourTree.Hyperparents, // (input)
            contourTree.Hyperarcs, // (input)
            contourTree.Hypernodes, // (input)
            contourTree.Supernodes, // (input)
            extrema.Peaks, // (input)
            extrema.Pits,
            superarcIds
          ); // (input)

    for(int i = 0 ; i < triangles.GetNumberOfValues() ; i++)
    {
        // Set triangle ID
        Triangle triangle = triangles.ReadPortal().Get(i);
        triangle.superarcId = superarcIds.ReadPortal().Get(i);
        triangles.WritePortal().Set(i, triangle);
    }
}

    //// Each point has it's own isovalue
    //cont::ArrayHandle<Float64> isovalueArray;
    //isovalueArray.Allocate(triangles.GetNumberOfValues());

    //// Endpoints of path in the contour tree
    //cont::ArrayHandle<Vec<Id, 2>> endpoints;
    //endpoints.Allocate(triangles.GetNumberOfValues());

    //for(int i = 0 ; i < triangles.GetNumberOfValues() ; i++)
    //{
        //isovalueArray.GetPortalControl().Set(i, isovalue);
        //endpoints.GetPortalControl().Set(i, triangles.GetPortalControl().Get(i).representativeEdge);
    //}

    //cont::ArrayHandle<Id> superarcIds = cv1k::filter::runFilter(contourTree, inputData, fieldName, endpoints, isovalueArray);

    //for(int i = 0 ; i < triangles.GetNumberOfValues() ; i++)
    //{
        //Triangle triangle = triangles.ReadPortal().Get(i);
        //triangle.superarcId = superarcIds.ReadPortal().Get(i);
        //triangles.GetPortalControl().Set(i, triangle);
    //}
