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

#include <algorithm>
#include <queue>

// the following file has moved to a different location (this code was from VTK-m v1.8):
//#include <vtkm/filter/ContourTreeUniformAugmented.h>
// Now (in VTK-m 2.2) this file is in:
#include <vtkm/filter/scalar_topology/ContourTreeUniformAugmented.h>

#include <vtkm/filter/scalar_topology/ContourTreeUniformAugmented.h>

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/MeshExtrema.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/DataSetMesh.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>


using namespace std;
using namespace vtkm;

tuple<worklet::contourtree_augmented::ContourTree,
vtkm::worklet::contourtree_augmented::IdArrayType,
vtkm::worklet::contourtree_augmented::IdArrayType,
vtkm::Id> cv1k::ct::getContourTree(cont::DataSet inputData, string fieldName)
{
    // The actual CT Computation Happens in worklet/ContourTreeUniformAugmented::RunContourTree
    // Set up the filter Contour Tree
    filter::scalar_topology::ContourTreeAugmented ctFilter(true, true);
    ctFilter.SetActiveField(fieldName);

    // It's better to have the contour tree directly than to have the packaged output of the filter
    //cont::DataSet ctOutputData = ctFilter.Execute(inputData);
    ctFilter.Execute(inputData);

    // These are permutation that map ct and mesh nodes to one another
    // sortOrder : ctNodeId   -> meshNodeId
    // sortIndex : meshNodeId -> ctNodeId
    const auto ctSortOrderPortal = ctFilter.GetSortOrder().ReadPortal();

    // They array is the inverse permutation of ctSortOrder, it's not available from the filter, so I need to make it
    vtkm::cont::ArrayHandle<vtkm::Id> ctSortIndices;
    ctSortIndices.Allocate(ctSortOrderPortal.GetNumberOfValues());

    for (int i = 0; i < ctSortOrderPortal.GetNumberOfValues(); i++)
    {
        ctSortIndices.WritePortal().Set(ctSortOrderPortal.Get(i), i);
    }

    return { ctFilter.GetContourTree(), ctFilter.GetSortOrder(), ctSortIndices, ctFilter.GetNumIterations() };
}

void cv1k::ct::printContourTreeArray(vtkm::cont::ArrayHandle<vtkm::Id> ctArcs)
{
    // Get Read Portal to the arcs field
    //const vtkm::cont::ArrayHandle<long long, vtkm::cont::StorageTagBasic>::PortalConstControl readArcsDataPortal = arcsData.GetPortalConstControl();
    const auto readArcsDataPortal = ctArcs.ReadPortal();

    //cout << "Printing Contour Tree..." << endl;
    for (vtkm::Id i = 0; i < readArcsDataPortal.GetNumberOfValues(); i++)
    {
        auto unmaskedIndex = worklet::contourtree_augmented::MaskedIndex(readArcsDataPortal.Get(i));

        cout << i << " - " << unmaskedIndex << endl;
    }
}


void writeArray(FILE *file, vtkm::cont::ArrayHandle<vtkm::Id> currentArray)
{
    vtkm::Id arraySize = currentArray.GetNumberOfValues();
    fwrite(&arraySize, sizeof(arraySize), 1, file);

    vtkm::Id *rawData = vtkm::cont::ArrayHandleBasic<vtkm::Id>(currentArray).GetWritePointer();

    fwrite(rawData, sizeof(rawData[0]), arraySize, file);
}

void readArray(FILE *file, vtkm::cont::ArrayHandle<vtkm::Id> &currentArray)
{
    vtkm::Id arraySize;
    fread(&arraySize, sizeof(arraySize), 1, file);

    vtkm::Id *rawData = new vtkm::Id[arraySize];
    fread(rawData, sizeof(rawData[0]), arraySize, file);

    currentArray.Allocate(arraySize);

    // The fix: add an explicit vtkm::CopyFlag parameter
    vtkm::cont::ArrayCopy(
        vtkm::cont::make_ArrayHandle(rawData, arraySize, vtkm::CopyFlag::On),
        currentArray);

    delete[] rawData;
}


void cv1k::ct::writeContourTree(vtkm::worklet::contourtree_augmented::ContourTree ct, vtkm::cont::ArrayHandle<vtkm::Id> sortOrder, vtkm::Id numIterations, std::string filename)
{
    FILE *file;
    file = fopen(filename.c_str(),"w");
    if (!file){ throw "Could not open binary file for writing.\n"; }

    fwrite(&numIterations, sizeof(numIterations), 1, file);

    writeArray(file, sortOrder);
    writeArray(file, ct.Nodes);
    writeArray(file, ct.Arcs);
    writeArray(file, ct.Superparents);
    writeArray(file, ct.Supernodes);
    writeArray(file, ct.Superarcs);
    writeArray(file, ct.Hyperparents);
    writeArray(file, ct.Hypernodes);
    writeArray(file, ct.Hyperarcs);
    writeArray(file, ct.WhenTransferred);
    writeArray(file, ct.Augmentnodes);
    writeArray(file, ct.Augmentarcs);
    writeArray(file, ct.FirstSupernodePerIteration);
    writeArray(file, ct.FirstHypernodePerIteration);

    fclose(file);
}


std::tuple<vtkm::worklet::contourtree_augmented::ContourTree, vtkm::cont::ArrayHandle<vtkm::Id>, vtkm::Id> cv1k::ct::readContourTree(std::string filename)
{
    FILE *file;
    file = fopen(filename.c_str(),"r");
    //file.open (filename, ios::in | ios::binary);

    if (!file){ throw "Could not open binary file for writing.\n"; }

    vtkm::Id numIterations;
    fread(&numIterations, sizeof(numIterations), 1, file);

    vtkm::cont::ArrayHandle<vtkm::Id> sortOrder;
    readArray(file, sortOrder);

    vtkm::worklet::contourtree_augmented::ContourTree ct;
    ct.Init(sortOrder.GetNumberOfValues());

    readArray(file, ct.Nodes);
    readArray(file, ct.Arcs);
    readArray(file, ct.Superparents);
    readArray(file, ct.Supernodes);
    readArray(file, ct.Superarcs);
    readArray(file, ct.Hyperparents);
    readArray(file, ct.Hypernodes);
    readArray(file, ct.Hyperarcs);
    readArray(file, ct.WhenTransferred);
    readArray(file, ct.Augmentnodes);
    readArray(file, ct.Augmentarcs);
    readArray(file, ct.FirstSupernodePerIteration);
    readArray(file, ct.FirstHypernodePerIteration);

    fclose(file);

    return {ct, sortOrder, numIterations};
}

