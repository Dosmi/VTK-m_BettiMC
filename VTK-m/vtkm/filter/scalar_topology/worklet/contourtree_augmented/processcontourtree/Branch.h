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

#ifndef vtk_m_worklet_contourtree_augmented_process_contourtree_inc_branch_h
#define vtk_m_worklet_contourtree_augmented_process_contourtree_inc_branch_h

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ContourTree.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/PiecewiseLinearFunction.h>

#include <cmath>

#define DEBUG_PRINT_PACTBD 0
//#define WRITE_FILES 1


namespace vtkm
{
namespace worklet
{
namespace contourtree_augmented
{
namespace process_contourtree_inc
{
	
//using ValueType = vtkm::Float64; //vtkm::FloatDefault;
using ValueType = vtkm::Float32; //vtkm::FloatDefault; //vtkm::FloatDefault;
using FloatArrayType = vtkm::cont::ArrayHandle<ValueType>;

// TODO The pointered list structure and use of std::vector don't seem to fit well with using Branch with VTKM
template <typename T>
class Branch
{
public:
  vtkm::Id OriginalId;              // Index of the extremum in the mesh
  vtkm::Id Extremum;                // Index of the extremum in the mesh
  T ExtremumVal;                    // Value at the extremum:w
  vtkm::Id Saddle;                  // Index of the saddle in the mesh (or minimum for root branch)
  T SaddleVal;                      // Corresponding value
  vtkm::Id Volume;                  // Volume
  //ValueType VolumeFloat;
  Branch<T>* Parent;                // Pointer to parent, or nullptr if no parent
  std::vector<Branch<T>*> Children; // List of pointers to children

  // 2025-12-15 NEW: adding the Betti-augmented supernode information to each branch
  // (Then we will be able to get isosurfaces at which topology changes per branch)
  std::vector<vtkm::Id> BettiChanges; // List of pointers to children
  std::vector<vtkm::FloatDefault> BettiChangesDataValue; // List data values at which there is a Betti number change
  //std::vector<ValueType> BettiArcVolumes; // List of volumes for arcs that have betti number changes
  std::vector<vtkm::Id> Betti1Numbers;     // Betti value of the branch

  vtkm::Id           TopBetti1Number;     // Betti value of the branch
  vtkm::FloatDefault TopBettiChangeDataValue;     // Betti value of the branch

  // Create branch decomposition from contour tree
  template <typename StorageType>
  static Branch<T>* ComputeBranchDecomposition(
    const IdArrayType& contourTreeSuperparents,
    const IdArrayType& contourTreeSupernodes,
    const IdArrayType& whichBranch,
    const IdArrayType& branchMinimum,
    const IdArrayType& branchMaximum,
    const IdArrayType& branchSaddle,
    const IdArrayType& branchParent,
    const IdArrayType& sortOrder,
    const vtkm::cont::ArrayHandle<T, StorageType>& dataField,
    bool dataFieldIsSorted);
    
  template <typename StorageType>
  static Branch<T>* ComputeBranchDecomposition(
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
  const IdArrayType& superarcIntrinsicWeight,            // NEW: passed intrincic ...
  const IdArrayType& superarcDependentWeight,            // ... and dependent weights
  const vtkm::Id& contourTreeRootnode,                      // added explicit root node because after betti augmentation the root not guaranteed to be last
  const IdArrayType& contourTreeSuperodeBetti,              // NEW: added Supernode-Betti number mappings after implementing Betti augmentation
        std::vector<Branch<T>*>& branches);

  // Simplify branch composition down to target size (i.e., consisting of targetSize branches)
  void SimplifyToSize(vtkm::Id targetSize, bool usePersistenceSorter = true);

  // Print the branch decomposition
  void PrintBranchDecomposition(std::ostream& os, std::string::size_type indent = 0) const;

  // Persistence of branch
  T Persistence() { return std::fabs(ExtremumVal - SaddleVal); }

  // Destroy branch (deleting children and propagating Volume to parent)
  ~Branch();

  // Compute list of relevant/interesting isovalues
  void GetRelevantValues(int type, T eps, std::vector<T>& values) const;

  void AccumulateIntervals(int type, T eps, PiecewiseLinearFunction<T>& plf) const;
  
    void PrintBranchInformation(); // TODO

//  static void PrintBranchInformation(Branch<T>* root);// TODO
  static void PrintBranchInformation(Branch<T>* branch,
                                     std::vector<std::vector<vtkm::Id>>& branch_SP_map,
                                     vtkm::Id bid)
  {
//      std::cout << bid << ":" << std::endl;
      if(branch == nullptr)
      {
          std::cout << " non ";
          return;
      }

//      std::cout << branch->OriginalId << " -> ";

//      if(branch->Children.empty())
//      {
//          std::cout << " non ";
//      }
      for(int j = 0; j < branch_SP_map[bid].size(); j+=3) //j++)
      {
          if(branch_SP_map[bid][j] != bid)
          {
              std::cout << branch_SP_map[bid][j] << " -> ";
          }
      }

      if(!branch->Children.empty())
      {
          std::cout << std::endl << "'1'Children of i=" << bid << std::endl;
      }

      for(auto child : branch->Children)
      {
          PrintBranchInformation(child, branch_SP_map, child->OriginalId);  // recursively print children
      }
  }


private:
  // Private default constructore to ensure that branch decomposition can only be created from a contour tree or loaded from storate (via static methods)
  Branch()
    : Extremum((vtkm::Id)vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT)
    , ExtremumVal(0)
    , Saddle((vtkm::Id)vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT)
    , SaddleVal(0)
    , Volume(0)
    , Parent(nullptr)
    , Children()
  {
  }

  // Remove symbolic perturbation, i.e., branches with zero persistence
  void removeSymbolicPerturbation();
}; // class Branch


template <typename T>
struct PersistenceSorter
{ // PersistenceSorter()
  inline bool operator()(Branch<T>* a, Branch<T>* b) { return a->Persistence() < b->Persistence(); }
}; // PersistenceSorter()


template <typename T>
struct VolumeSorter
{ // VolumeSorter()
  inline bool operator()(Branch<T>* a, Branch<T>* b) { return a->Volume < b->Volume; }
}; // VolumeSorter()


template <typename T>
template <typename StorageType>
Branch<T>* Branch<T>::ComputeBranchDecomposition(
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
{ // C)omputeBranchDecomposition()
  auto branchMinimumPortal = branchMinimum.ReadPortal();
  auto branchMaximumPortal = branchMaximum.ReadPortal();
  auto branchSaddlePortal = branchSaddle.ReadPortal();
  auto branchParentPortal = branchParent.ReadPortal();
  auto sortOrderPortal = sortOrder.ReadPortal();
  auto supernodesPortal = contourTreeSupernodes.ReadPortal();
  auto dataFieldPortal = dataField.ReadPortal();
  vtkm::Id nBranches = branchSaddle.GetNumberOfValues();
  std::vector<Branch<T>*> branches;
  Branch<T>* root = nullptr;
  branches.reserve(static_cast<std::size_t>(nBranches));

  for (int branchID = 0; branchID < nBranches; ++branchID)
    branches.push_back(new Branch<T>);

  // Reconstruct explicit branch decomposition from array representation
  for (std::size_t branchID = 0; branchID < static_cast<std::size_t>(nBranches); ++branchID)
  {
    branches[branchID]->OriginalId = static_cast<vtkm::Id>(branchID);
    if (!NoSuchElement(branchSaddlePortal.Get(static_cast<vtkm::Id>(branchID))))
    {
      branches[branchID]->Saddle = MaskedIndex(
        supernodesPortal.Get(MaskedIndex(branchSaddlePortal.Get(static_cast<vtkm::Id>(branchID)))));
      vtkm::Id branchMin = MaskedIndex(supernodesPortal.Get(
        MaskedIndex(branchMinimumPortal.Get(static_cast<vtkm::Id>(branchID)))));
      vtkm::Id branchMax = MaskedIndex(supernodesPortal.Get(
        MaskedIndex(branchMaximumPortal.Get(static_cast<vtkm::Id>(branchID)))));
      if (branchMin < branches[branchID]->Saddle)
        branches[branchID]->Extremum = branchMin;
      else if (branchMax > branches[branchID]->Saddle)
        branches[branchID]->Extremum = branchMax;
      else
      {
        std::cerr << "Internal error";
        return 0;
      }
    }
    else
    {
      branches[branchID]->Saddle =
        supernodesPortal.Get(MaskedIndex(branchMinimumPortal.Get(static_cast<vtkm::Id>(branchID))));
      branches[branchID]->Extremum =
        supernodesPortal.Get(MaskedIndex(branchMaximumPortal.Get(static_cast<vtkm::Id>(branchID))));
    }

    if (dataFieldIsSorted)
    {
      branches[branchID]->SaddleVal = dataFieldPortal.Get(branches[branchID]->Saddle);
      branches[branchID]->ExtremumVal = dataFieldPortal.Get(branches[branchID]->Extremum);
    }
    else
    {
      branches[branchID]->SaddleVal =
        dataFieldPortal.Get(sortOrderPortal.Get(branches[branchID]->Saddle));
      branches[branchID]->ExtremumVal =
        dataFieldPortal.Get(sortOrderPortal.Get(branches[branchID]->Extremum));
    }

    branches[branchID]->Saddle = sortOrderPortal.Get(branches[branchID]->Saddle);
    branches[branchID]->Extremum = sortOrderPortal.Get(branches[branchID]->Extremum);

    if (NoSuchElement(branchParentPortal.Get(static_cast<vtkm::Id>(branchID))))
    {
      root = branches[branchID]; // No parent -> this is the root branch
    }
    else
    {
      branches[branchID]->Parent = branches[static_cast<size_t>(
        MaskedIndex(branchParentPortal.Get(static_cast<vtkm::Id>(branchID))))];
      branches[branchID]->Parent->Children.push_back(branches[branchID]);
    }
  }

  // FIXME: This is a somewhat hackish way to compute the Volume, but it works
  // It would probably be better to compute this from the already computed Volume information
  auto whichBranchPortal = whichBranch.ReadPortal();
  auto superparentsPortal = contourTreeSuperparents.ReadPortal();
  for (vtkm::Id i = 0; i < contourTreeSuperparents.GetNumberOfValues(); i++)
  {
    branches[static_cast<size_t>(
               MaskedIndex(whichBranchPortal.Get(MaskedIndex(superparentsPortal.Get(i)))))]
      ->Volume++; // Increment Volume
  }
  if (root)
  {
    root->removeSymbolicPerturbation();
  }

  return root;
} // ComputeBranchDecomposition()






















template <typename T>
template <typename StorageType>
Branch<T>* Branch<T>::ComputeBranchDecomposition(
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
  const IdArrayType& superarcIntrinsicWeight,            // NEW: passed intrincic ...
  const IdArrayType& superarcDependentWeight,            // ... and dependent weights
  const vtkm::Id& contourTreeRootnode,                      // added explicit root node because after betti augmentation the root not guaranteed to be last
  const IdArrayType& contourTreeSuperodeBetti,              // NEW: added Supernode-Betti number mappings after implementing Betti augmentation
        std::vector<Branch<T>*>& branches)                   // output
{ // ComputeBranchDecomposition()

  std::cout << "[(Branch.h) ContourTreeApp->ProcessContourTree->Branch.h::ComputeBranchDecomposition()] START" << std::endl;

  auto branchMinimumPortal = branchMinimum.ReadPortal();
  auto branchMaximumPortal = branchMaximum.ReadPortal();
  auto branchSaddlePortal = branchSaddle.ReadPortal();
  auto branchParentPortal = branchParent.ReadPortal();
  auto sortOrderPortal = sortOrder.ReadPortal();
  auto supernodesPortal = contourTreeSupernodes.ReadPortal();
  auto dataFieldPortal = dataField.ReadPortal();
  auto valueFieldPortal = valueField.ReadPortal();
  auto supernodeBettiPortal = contourTreeSuperodeBetti.ReadPortal();

  // NEW: add the read portals for branch intrinsic weights:
  auto superarcIntrinsicWeightPortal = superarcIntrinsicWeight.ReadPortal();
  // NEW: and the superarcs
  auto superarcsPortal = contourTreeSuperarcs.ReadPortal();



  // 2025-12-10 debugging of the branch decomposition after betti number augmentation



#if DEBUG_PRINT_PACTBD
  std::cout << std::endl << "(Branch.h->ComputeBranchDecomposition) Superarc Intrinsic Weight Portal (PASSED IN):" << std::endl;
  for(int i = 0; i < superarcIntrinsicWeightPortal.GetNumberOfValues(); i++)
  {
      std::cout << i << " -> " << superarcIntrinsicWeightPortal.Get(i) << std::endl;
  }
  std::cout << std::endl;
#endif

//  NOT USING DEPENDENT WEIGHTS YET
//  std::cout << std::endl << "(Branch.h->ComputeBranchDecomposition) Superarc Dependent Weight Portal:" << std::endl;
//  for(int i = 0; i < superarcDependentWeightNEWPortal.GetNumberOfValues(); i++)
//  {
//      std::cout << i << " -> " << superarcDependentWeightNEWPortal.Get(i) << std::endl;

//      superarcDependentWeightCorrectWritePortal.Set(i, realDependent[i]);

//      std::cout << indent << i << " -> " << superarcDependentWeightCorrectReadPortal.Get(i) << std::endl;
//  }







  vtkm::Id nBranches = branchSaddle.GetNumberOfValues();
//  std::vector<Branch<T>*> branches; now made an output by passed-in array
  Branch<T>* root = nullptr;
//  branches.reserve(static_cast<std::size_t>(nBranches));


  std::cout << "Number of Branches, for the Branch Decomposition:" << nBranches << std::endl;

  for (int branchID = 0; branchID < nBranches; ++branchID)
    branches.push_back(new Branch<T>);

  // Reconstruct explicit branch decomposition from array representation
  for (std::size_t branchID = 0; branchID < static_cast<std::size_t>(nBranches); ++branchID)
  {
    branches[branchID]->OriginalId = static_cast<vtkm::Id>(branchID);
    if (!NoSuchElement(branchSaddlePortal.Get(static_cast<vtkm::Id>(branchID))))
    {
      branches[branchID]->Saddle = MaskedIndex(
        supernodesPortal.Get(MaskedIndex(branchSaddlePortal.Get(static_cast<vtkm::Id>(branchID)))));
      vtkm::Id branchMin = MaskedIndex(supernodesPortal.Get(
        MaskedIndex(branchMinimumPortal.Get(static_cast<vtkm::Id>(branchID)))));
      vtkm::Id branchMax = MaskedIndex(supernodesPortal.Get(
        MaskedIndex(branchMaximumPortal.Get(static_cast<vtkm::Id>(branchID)))));
      if (branchMin < branches[branchID]->Saddle)
        branches[branchID]->Extremum = branchMin;
      else if (branchMax > branches[branchID]->Saddle)
        branches[branchID]->Extremum = branchMax;
      else
      {
        std::cerr << "Internal error";
        return 0;
      }
    }
    else
    {
      branches[branchID]->Saddle =
        supernodesPortal.Get(MaskedIndex(branchMinimumPortal.Get(static_cast<vtkm::Id>(branchID))));
      branches[branchID]->Extremum =
        supernodesPortal.Get(MaskedIndex(branchMaximumPortal.Get(static_cast<vtkm::Id>(branchID))));
    }

    if (dataFieldIsSorted)
    {
      branches[branchID]->SaddleVal = dataFieldPortal.Get(branches[branchID]->Saddle);
      branches[branchID]->ExtremumVal = dataFieldPortal.Get(branches[branchID]->Extremum);
    }
    else
    {
      branches[branchID]->SaddleVal =
        dataFieldPortal.Get(sortOrderPortal.Get(branches[branchID]->Saddle));
      branches[branchID]->ExtremumVal =
        dataFieldPortal.Get(sortOrderPortal.Get(branches[branchID]->Extremum));
    }

    branches[branchID]->Saddle = sortOrderPortal.Get(branches[branchID]->Saddle);
    branches[branchID]->Extremum = sortOrderPortal.Get(branches[branchID]->Extremum);

    if (NoSuchElement(branchParentPortal.Get(static_cast<vtkm::Id>(branchID))))
    {
      root = branches[branchID]; // No parent -> this is the root branch
    }
    else
    {
      branches[branchID]->Parent = branches[static_cast<size_t>(
        MaskedIndex(branchParentPortal.Get(static_cast<vtkm::Id>(branchID))))];
      branches[branchID]->Parent->Children.push_back(branches[branchID]);
    }
  }

//  // FIXME: This is a somewhat hackish way to compute the Volume, but it works
//  // It would probably be better to compute this from the already computed Volume information
//  // (already replaced, 2025-03-06 commented out until floats)
//  std::cout << "Computing Integer Volumes" << std::endl;
  auto whichBranchPortal = whichBranch.ReadPortal();
  auto superparentsPortal = contourTreeSuperparents.ReadPortal();
  for (vtkm::Id i = 0; i < contourTreeSuperparents.GetNumberOfValues(); i++)
  {
    branches[static_cast<size_t>(
               MaskedIndex(whichBranchPortal.Get(MaskedIndex(superparentsPortal.Get(i)))))]
      ->Volume++; // Increment Volume

    //std::cout << "branch[" << static_cast<size_t>(MaskedIndex(whichBranchPortal.Get(MaskedIndex(superparentsPortal.Get(i)))))
              //<< "]" << branches[static_cast<size_t>(MaskedIndex(whichBranchPortal.Get(MaskedIndex(superparentsPortal.Get(i)))))]->Volume
              //<< std::endl;
  }


  vtkm::Id sortID; // = supernodesPortal.Get(i);
  int current_superparent;
  int supernode_tailend;
  std::vector<std::vector<vtkm::Id>> branch_SP_map(nBranches);
  std::vector<std::vector<vtkm::Id>> branch_SP_Betti_map(nBranches);

// 2025-03-10 Commented out the branch initialisation, replaced with a branch-parent-array
#if DEBUG_PRINT_PACTBD
  std::cout << "------- vvv Branch VALUE TYPE INITIAL weights vvv --------" << std::endl;
#endif
  for (vtkm::Id i = 0; i < contourTreeSuperparents.GetNumberOfValues(); i++)
  {
    current_superparent = MaskedIndex(superparentsPortal.Get(i));
//    sortID = supernodesPortal.Get(i);
    supernode_tailend = supernodesPortal.Get(MaskedIndex(superarcsPortal.Get(current_superparent)));

    size_t branchID = static_cast<size_t>(MaskedIndex(whichBranchPortal.Get(current_superparent)));
    

    vtkm::Id branchMin = MaskedIndex(supernodesPortal.Get(
      MaskedIndex(branchMinimumPortal.Get(static_cast<vtkm::Id>(branchID)))));
    vtkm::Id branchMax = MaskedIndex(supernodesPortal.Get(
      MaskedIndex(branchMaximumPortal.Get(static_cast<vtkm::Id>(branchID)))));


#if DEBUG_PRINT_PACTBD
    std::cout << i << "[" << branchID << "] " << " (" << current_superparent << ") "
              << branches[branchID]->Extremum << "->" << branches[branchID]->Saddle << " = "
              << branches[branchID]->Volume
              << std::endl;
#endif
    if ( std::find(branch_SP_map[branchID].begin(), branch_SP_map[branchID].end(), current_superparent) == branch_SP_map[branchID].end() )
    {
#if DEBUG_PRINT_PACTBD
      // 2025-03-10
      std::cout << "-> ADDED BRANCH CHAIN SP: " << current_superparent << std::endl;
#endif
      branch_SP_map[branchID].push_back(current_superparent); // supernode_tailend
      branch_SP_map[branchID].push_back(i); // NEW 2025-03-14
      branch_SP_map[branchID].push_back(supernode_tailend); // NEW 2025-03-14
    }

  }
  
#if DEBUG_PRINT_PACTBD
  std::cout << "------- ^^^ Branch VALUE TYPE INITIAL weights ^^^ --------" << std::endl << std::endl;
  std::cout << "Printing the supernode/branch mappings" << std::endl;
#endif

//  for (vtkm::Id i = 0; i < contourTreeSuperparents.GetNumberOfValues(); i++)
//  {

//      std::cout << i << ")" << MaskedIndex(whichBranchPortal.Get(MaskedIndex(superparentsPortal.Get(i)))) << std::endl;

//    branches[static_cast<size_t>(
//               MaskedIndex(whichBranchPortal.Get(MaskedIndex(superparentsPortal.Get(i)))))]
//      ->Volume++; // Increment Volume
//  }
#if DEBUG_PRINT_PACTBD
std::cout << "Printing the supernode/branch mappings" << std::endl;
#endif

//#if WRITE_FILES
    std::ofstream filebsp("ContourTreeBranches--branchSPs.txt");
    std::cout << "Contour Tree Root Node: " << contourTreeRootnode << std::endl;
//#endif

  for(int i = 0; i < nBranches; i++)
  {
#if WRITE_FILES
      std::cout << "branch[" << i << "] "  << "\t";
#endif

#if WRITE_FILES
    std::cout << i << " -> "
              << branches[i]->Extremum << "->" << branches[i]->Saddle << " = "
              << branches[i]->Volume << std::endl << "\t";
              
              std::cout << std::endl << "branch i SPs\t" << i << "\t" << std::endl;
              
#endif

//    std::cout << "branches[" << i << "]->PrintBranchDecomposition before adding volumes:" << std::endl;
//    branches[i]->PrintBranchDecomposition(std::cout);

//#if WRITE_FILES
    //std::cout << std::endl << "branch i SPs\t" << i << "\t";// << std::endl;
    filebsp << std::endl << "branch i SPs\t" << i << "\t";// << std::endl;
//#endif

    ValueType TopBettiArcVolume = 0.f; // List of volumes for arcs that have betti number changes
    vtkm::Id TopBettiChange = 0; // List of volumes for arcs that have betti number changes

	//std::cout << std::endl << "branch_SP_map[i].size() = " << branch_SP_map[i].size() << std::endl;
	
	//for(int vhog = 0; vhog < sortOrderPortal.GetNumberOfValues(); vhog++)
	//{
		//std::cout << vhog << "\t" << sortOrderPortal.Get(vhog) << std::endl;
	//}
	
	//for(int vhog = 0; vhog < supernodesPortal.GetNumberOfValues(); vhog++)
	//{
		//std::cout << vhog << "\t" << supernodesPortal.Get(vhog) << std::endl;
	//}

	//std::cout << "Num. Betti changes: " << supernodeBettiPortal.GetNumberOfValues() << std::endl;

    for(int j = 0; j < branch_SP_map[i].size(); j+=3) //j++)
    {
		//std::cout << "branch_SP_map[i].size() = " << branch_SP_map[i].size() << std::endl;
		//std::cout << "branch_SP_map[i][" << j << "] = " << branch_SP_map[i][j] << std::endl;
		//std::cout << "sortOrderPortal.Get(branch_SP_map[i][j]) = "  << sortOrderPortal.Get(branch_SP_map[i][j]) << std::endl;
		//std::cout << "supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) = "  << supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) << std::endl;
		//std::cout << "supernodesPortal.Get(branch_SP_map[i][j]) = " << supernodesPortal.Get(branch_SP_map[i][j]) << std::endl;
		
        //vtkm::Id regularIDbr = supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j]));
        //vtkm::Id regularIDbr = supernodesPortal.Get(branch_SP_map[i][j]);
        vtkm::Id regularIDbr = sortOrderPortal.Get(supernodesPortal.Get(branch_SP_map[i][j])); // get its sort order as regular
        //vtkm::Id regularIDbr = sortOrderPortal.Get(branch_SP_map[i][j]); // get its sort order as regular

//#if WRITE_FILES
        //std::cout << branch_SP_map[i][j] << "(" << supernodeBettiPortal.Get(branch_SP_map[i][j]) << ")"; //"(" << dataFieldPortal.Get(branch_SP_map[i][j]) << ")";
        if(supernodeBettiPortal.GetNumberOfValues() > 0)
        {
			filebsp   << branch_SP_map[i][j] << "(" << supernodeBettiPortal.Get(branch_SP_map[i][j]) << ")"; //"(" << dataFieldPortal.Get(branch_SP_map[i][j]) << ")";
		}
        //std::cout << "[" << regularIDbr << "]";
        filebsp   << "[" << regularIDbr << "]";
        //std::cout << "{" << valueFieldPortal.Get( supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) ) << "}\t";
        //filebsp   << "{" << valueFieldPortal.Get( supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) ) << "}\t"; // old
        filebsp   << "{" << valueFieldPortal.Get( regularIDbr ) << "}\t"; // old
//#endif

        if(branch_SP_map[i][j] > contourTreeRootnode) // 9) // 2025-12-15 hack-resolved 2025-12-20
        {
            // make note of supernode IDs where Betti Numbers change
            branches[i]->BettiChanges.push_back(branch_SP_map[i][j]);
            //branches[i]->BettiChangesDataValue.push_back(valueFieldPortal.Get( supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) ));
            branches[i]->BettiChangesDataValue.push_back(valueFieldPortal.Get( regularIDbr ));
            // ... then the volumes for potential simplification:
            //branches[i]->BettiArcVolumes.push_back(superarcIntrinsicWeightPortal.Get(branch_SP_map[i][j])); // back to integer weights
            if(supernodeBettiPortal.GetNumberOfValues() > 0)
            {
				branches[i]->Betti1Numbers.push_back(supernodeBettiPortal.Get(branch_SP_map[i][j]));
			}

            // add an additional filter just to get Betti1 changes:
//            if ((supernodeBettiPortal.Get(branch_SP_map[i][j]) == 2) && (superarcIntrinsicWeightPortal.Get(branch_SP_map[i][j]) > TopBettiArcVolume))
            if(superarcIntrinsicWeightPortal.Get(branch_SP_map[i][j]) > TopBettiArcVolume)
            {
                TopBettiChange = supernodeBettiPortal.Get(branch_SP_map[i][j]);
                TopBettiArcVolume = superarcIntrinsicWeightPortal.Get(branch_SP_map[i][j]);
                branches[i]->TopBetti1Number = supernodeBettiPortal.Get(branch_SP_map[i][j]);     // Betti value of the branch
                //branches[i]->TopBettiChangeDataValue = valueFieldPortal.Get( supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) );    // Betti value of the branch
                branches[i]->TopBettiChangeDataValue = valueFieldPortal.Get( regularIDbr );    // Betti value of the branch
            }

//            if(superarcIntrinsicWeightPortal.Get(branch_SP_map[i][j]) > TopBettiArcVolume)
//            if(supernodeBettiPortal.Get(branch_SP_map[i][j]) > TopBettiChange)
//            {
//                TopBettiChange = supernodeBettiPortal.Get(branch_SP_map[i][j]);
//                TopBettiArcVolume = superarcIntrinsicWeightPortal.Get(branch_SP_map[i][j]);
//                branches[i]->TopBetti1Number = supernodeBettiPortal.Get(branch_SP_map[i][j]);     // Betti value of the branch
//                branches[i]->TopBettiChangeDataValue = valueFieldPortal.Get( supernodesPortal.Get(sortOrderPortal.Get(branch_SP_map[i][j])) );    // Betti value of the branch
//            }

        }

    }
    
    filebsp << std::endl << "\t\t";
    
    // Betti Numbers Row:
    for(int j = 0; j < branch_SP_map[i].size(); j+=3) //j++)
    {
        vtkm::Id regularIDbr = sortOrderPortal.Get(supernodesPortal.Get(branch_SP_map[i][j])); // get its sort order as regular
        if(supernodeBettiPortal.GetNumberOfValues() > 0)
        {
			filebsp << supernodeBettiPortal.Get(branch_SP_map[i][j]) << "\t"; //"(" << dataFieldPortal.Get(branch_SP_map[i][j]) << ")";
		}
	}

	filebsp << std::endl << "\t\t";
	
	// Isovalue Row:
    for(int j = 0; j < branch_SP_map[i].size(); j+=3) //j++)
    {
        vtkm::Id regularIDbr = sortOrderPortal.Get(supernodesPortal.Get(branch_SP_map[i][j])); // get its sort order as regular
        filebsp << valueFieldPortal.Get( regularIDbr ) << "\t"; //"(" << dataFieldPortal.Get(branch_SP_map[i][j]) << ")";
	}

	filebsp << std::endl;

  }

  std::string indent = "\t";
#if DEBUG_PRINT_PACTBD
  std::cout << "num of children at branch 0: " << branches[0]->Children.size() << std::endl;
#endif


//#if WRITE_FILES
  std::ofstream filebranchvolumes("ContourTreeBranches--BranchVolumes.txt");
  for(int i = 0; i < nBranches; i++)
  {
    filebranchvolumes << i << "\t" << branches[i]->Volume << std::endl;

//#endif
  }
  
  for(int i = 0; i < nBranches; i++)
  {
	filebranchvolumes << i << "\t";
	for(int j = 0; j < branches[i]->Betti1Numbers.size(); j++)
	{
		filebranchvolumes << branches[i]->Betti1Numbers[j] << "\t";
	}
	filebranchvolumes << std::endl;

//#endif
  }


  if (root)
  {
    root->removeSymbolicPerturbation();
  }

  std::cout << "[ContourTreeApp->ProcessContourTree->Branch.h::ComputeBranchDecomposition()] END" << std::endl;

  return root;
} // ComputeBranchDecomposition()





























template <typename T>
void Branch<T>::SimplifyToSize(vtkm::Id targetSize, bool usePersistenceSorter)
{ // SimplifyToSize()
  if (targetSize <= 1)
    return;

  // Top-down simplification, starting from one branch and adding in the rest on a biggest-first basis
  std::vector<Branch<T>*> q;
  q.push_back(this);

  std::vector<Branch<T>*> active;
  while (active.size() < static_cast<std::size_t>(targetSize) && !q.empty())
  {
    if (usePersistenceSorter)
    {
      std::pop_heap(
        q.begin(),
        q.end(),
        PersistenceSorter<
          T>()); // FIXME: This should be Volume, but we were doing this wrong for the demo, so let's start with doing this wrong here, too
    }
    else
    {
      std::pop_heap(
        q.begin(),
        q.end(),
        VolumeSorter<
          T>()); // FIXME: This should be Volume, but we were doing this wrong for the demo, so let's start with doing this wrong here, too
    }
    Branch<T>* b = q.back();
    q.pop_back();

    active.push_back(b);

    for (Branch<T>* c : b->Children)
    {
      q.push_back(c);
      if (usePersistenceSorter)
      {
        std::push_heap(q.begin(), q.end(), PersistenceSorter<T>());
      }
      else
      {
        std::push_heap(q.begin(), q.end(), VolumeSorter<T>());
      }
    }
  }

  // Rest are inactive
  for (Branch<T>* b : q)
  {
    // Hackish, remove c from its parents child list
    if (b->Parent)
      b->Parent->Children.erase(
        std::remove(b->Parent->Children.begin(), b->Parent->Children.end(), b));

    delete b;
  }
} // SimplifyToSize()


//template <typename T>
//void Branch<T>::PrintBranchDecomposition(std::ostream& os, std::string::size_type indent) const
//{ // PrintBranchDecomposition()
  //os << std::string(indent, ' ') << "{" << std::endl;
  //os << std::string(indent, ' ') << "  Saddle = " << SaddleVal << " (" << Saddle << ")"
     //<< std::endl;
  //os << std::string(indent, ' ') << "  Extremum = " << ExtremumVal << " (" << Extremum << ")"
     //<< std::endl;
  //os << std::string(indent, ' ') << "  Volume = " << Volume << std::endl;
  //if (!Children.empty())
  //{
    //os << std::string(indent, ' ') << "  Children = [" << std::endl;
    //for (Branch<T>* c : Children)
      //c->PrintBranchDecomposition(os, indent + 4);
    //os << std::string(indent, ' ') << std::string(indent, ' ') << "  ]" << std::endl;
  //}
  //os << std::string(indent, ' ') << "}" << std::endl;
//} // PrintBranchDecomposition()

template <typename T>
// print the graph in python dict format:
void Branch<T>::PrintBranchDecomposition(std::ostream& os, std::string::size_type indent) const
{ // PrintBranchDecomposition()

  os << std::string(indent, ' ') << "{" << std::endl;
  os << std::string(indent, ' ') << "  'ID' : " << OriginalId << "," << std::endl;
  os << std::string(indent, ' ') << "  'Saddle' : " << SaddleVal << ","
     << std::endl;
  os << std::string(indent, ' ') << "  'Extremum' : " << ExtremumVal << ","
     << std::endl;
  os << std::string(indent, ' ') << "  'Volume' : " << Volume << ","<< std::endl;
  //os << std::string(indent, ' ') << "  'VolumeFloat' : " << VolumeFloat << ","<< std::endl;

  if (BettiChanges.size() > 0)
  {

      os << std::string(indent, ' ') << "  'TopBettiChangeDataValue' : " << TopBettiChangeDataValue << ","<< std::endl;
      os << std::string(indent, ' ') << "  'TopBetti1Number' : " << TopBetti1Number << ","<< std::endl;

      os << std::string(indent, ' ') << "  'Betti Changes (val)' : ["; // << std::endl;
      for (vtkm::FloatDefault c : BettiChangesDataValue)
      {
        os << c << " ";
      }
      os << "]," << std::endl;

      os << std::string(indent, ' ') << "  'Betti 1 Changes' : ["; // << std::endl;
      for (ValueType c : Betti1Numbers)
      {
          os << c << " ";
      }
      os << "]," << std::endl;


      //os << std::string(indent, ' ') << "  'Betti Segment Volumes' : ["; // << std::endl;
      //for (ValueType c : BettiArcVolumes)
      //{
        //os << c << " ";
      //}
      //os << "]," << std::endl;

    os << std::string(indent, ' ') << "  'Betti Changes (SPs)' : ["; // << std::endl;
    for (vtkm::Id c : BettiChanges)
    {
      os << c << " ";
    }
    os << "]," << std::endl;

  }

  if (!Children.empty())
  {
    os << std::string(indent, ' ') << "  'Children' : [" << std::endl;
    for (Branch<T>* c : Children)
    {
      c->PrintBranchDecomposition(os, indent + 4);
    }
    os << std::string(indent, ' ') << std::string(indent, ' ') << "  ]," << std::endl;
  }

  os << std::string(indent, ' ') << "}," << std::endl;
} // PrintBranchDecomposition()



template <typename T>
Branch<T>::~Branch()
{ // ~Branch()
  for (Branch<T>* c : Children)
    delete c;
  if (Parent)
    Parent->Volume += Volume;
} // ~Branch()


// TODO this recursive accumlation of values does not lend itself well to the use of VTKM data structures
template <typename T>
void Branch<T>::GetRelevantValues(int type, T eps, std::vector<T>& values) const
{ // GetRelevantValues()
  T val;

  bool isMax = false;
  if (ExtremumVal > SaddleVal)
    isMax = true;

  switch (type)
  {
    default:
    case 0:
      val = SaddleVal + (isMax ? +eps : -eps);
      break;
    case 1:
      val = T(0.5f) * (ExtremumVal + SaddleVal);
      break;
    case 2:
      val = ExtremumVal + (isMax ? -eps : +eps);
      break;
  }
  if (Parent)
    values.push_back({ val });
  for (Branch* c : Children)
    c->GetRelevantValues(type, eps, values);
} // GetRelevantValues()


template <typename T>
void Branch<T>::AccumulateIntervals(int type, T eps, PiecewiseLinearFunction<T>& plf) const
{ //AccumulateIntervals()
  bool isMax = (ExtremumVal > SaddleVal);
  T val;

  switch (type)
  {
    default:
    case 0:
      val = SaddleVal + (isMax ? +eps : -eps);
      break;
    case 1:
      val = T(0.5f) * (ExtremumVal + SaddleVal);
      break;
    case 2:
      val = ExtremumVal + (isMax ? -eps : +eps);
      break;
  }

  if (Parent)
  {
    PiecewiseLinearFunction<T> addPLF;
    addPLF.addSample(SaddleVal, 0.0);
    addPLF.addSample(ExtremumVal, 0.0);
    addPLF.addSample(val, 1.0);
    plf += addPLF;
  }
  for (Branch<T>* c : Children)
    c->AccumulateIntervals(type, eps, plf);
} // AccumulateIntervals()


template <typename T>
void Branch<T>::removeSymbolicPerturbation()
{                                      // removeSymbolicPerturbation()
  std::vector<Branch<T>*> newChildren; // Temporary list of children that are not flat

  for (Branch<T>* c : Children)
  {
    // First recursively remove symbolic perturbation (zero persistence branches) for  all children below the current child
    // Necessary to be able to detect whether we can remove the current child
    c->removeSymbolicPerturbation();

    // Does child have zero persistence (flat region)
    if (c->ExtremumVal == c->SaddleVal && c->Children.empty())
    {
      // If yes, then we get its associated Volume and delete it
      delete c; // Will add Volume to parent, i.e., us
    }
    else
    {
      // Otherwise, keep child
      newChildren.push_back(c);
    }
  }
  // Swap out new list of children
  Children.swap(newChildren);
} // removeSymbolicPerturbation()

} // process_contourtree_inc
} // namespace contourtree_augmented
} // namespace worklet
} // namespace vtkm

#endif // vtk_m_worklet_contourtree_augmented_process_contourtree_inc_branch_h
