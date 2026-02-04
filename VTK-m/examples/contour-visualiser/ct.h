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

#pragma once

#include <string>

#include <vtkm/cont/DataSet.h>

// the following file has moved to a different location (this code was from VTK-m v1.8):
//#include <vtkm/worklet/contourtree_augmented/ContourTree.h>
// Now (in VTK-m 2.2) this file is in:
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ContourTree.h>

namespace cv1k
{
namespace ct
{

    // Compute the contour tree of a data set
    std::tuple<vtkm::worklet::contourtree_augmented::ContourTree, vtkm::cont::ArrayHandle<vtkm::Id>, vtkm::cont::ArrayHandle<vtkm::Id>, vtkm::Id> getContourTree(vtkm::cont::DataSet inputData, std::string fieldName);
    
    // Print a contour tree array by unmasking the elemets
    void printContourTreeArray(vtkm::cont::ArrayHandle<vtkm::Id>);


    void writeContourTree(vtkm::worklet::contourtree_augmented::ContourTree, vtkm::cont::ArrayHandle<vtkm::Id>, vtkm::Id, std::string);

    std::tuple<vtkm::worklet::contourtree_augmented::ContourTree, vtkm::cont::ArrayHandle<vtkm::Id>, vtkm::Id> readContourTree(std::string);

}
}
