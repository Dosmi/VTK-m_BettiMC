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

#include "interface.h"

#include "./ct.h"
#include "./mc.h"
#include "./filter.h"

#include <string>
#include <cassert>
#include <algorithm>

#include <vtkm/io/VTKDataSetWriter.h>

#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleConstant.h>
//#include <vtkm/cont/DataSetFieldAdd.h> deprecated since VTK 1.6 https://gitlab.kitware.com/vtk/vtk-m/-/releases/v1.6.0
#include <vtkm/cont/DataSetBuilderExplicit.h>

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ProcessContourTree.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/Branch.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/SetTriangleSuperarcId.h>

using BranchType = vtkm::worklet::contourtree_augmented::process_contourtree_inc::Branch<ValueType>;
using Coefficients = vtkm::worklet::contourtree_augmented::Coefficients;


using namespace std;
using namespace vtkm;

vtkm::cont::PartitionedDataSet cv1k::interface::computeMostSignificantContours(vtkm::cont::DataSet inputData,
                                                                               std::string fieldName,
                                                                               std::string inputCTFilename,
                                                                               int simplificationThreshold,
                                                                               std::string decompositionType,
                                                                               string selectionType,
                                                                               std::string branchIsovalueFlag,
                                                                               bool performanceRun)
{//computeMostSignificantContours
    // debug >= 1, print what is being currently computed with timings
    // debug >= 2, print contour tree arrays and isosurfac triangles
    int debugLevel = 1;
    
	// Red text formatting for highlighting some console output:
    const std::string RED = "\033[31m";  // Start red text
    const std::string ORANGE = "\033[38;2;255;165;0m";  // Start red text
    const std::string YELLOW = "\033[38;2;240;240;13m";  // Warm, readable yellow
    const std::string RESET = "\033[0m"; // End red text

    //
    // Compute Contour Tree
    //
    
    
    std::cout << "Computing Contour Tree ..." << std::endl;
    

    vtkm::cont::Timer timer;
    timer.Start();

    worklet::contourtree_augmented::ContourTree ct;
    worklet::contourtree_augmented::ContourTree ctBetti;
    vtkm::cont::ArrayHandle<vtkm::Id> ctSortOrder;
    vtkm::cont::ArrayHandle<vtkm::Id> ctSortIndices;
    Id ctNumIterations;

    if (true == inputCTFilename.empty())
    {
        tie(ct, ctSortOrder, ctSortIndices, ctNumIterations) = cv1k::ct::getContourTree(inputData, fieldName);
    }
    else
    {
        tie(ct, ctSortOrder, ctNumIterations) = cv1k::ct::readContourTree(inputCTFilename);
    }

    timer.Stop();

    if (debugLevel >= 1)
    {
        //std::cout << "Computing CT took " << timer.GetElapsedTime() << " seconds." << std::endl;
    }

    //
    // Compute Branch Decomposition
    //
    cont::ArrayHandle<vtkm::Id> whichBranch;
    cont::ArrayHandle<vtkm::Id> branchMinimum;
    cont::ArrayHandle<vtkm::Id> branchMaximum;
    cont::ArrayHandle<vtkm::Id> branchSaddle;
    cont::ArrayHandle<vtkm::Id> branchParent;

    cont::ArrayHandle<Id> superarcIntrinsicWeight;
    cont::ArrayHandle<Id> superarcDependentWeight;
    cont::ArrayHandle<Id> supernodeTransferWeight;
    cont::ArrayHandle<Id> hyperarcDependentWeigh;
    
	// ================================================= BettiMC (from PACT-BD) ================================================= //

    // Floating point type weights are required for PACT-BD
    FloatArrayType superarcIntrinsicWeightNEW;
    FloatArrayType superarcDependentWeightNEW;
    FloatArrayType supernodeTransferWeightNEW;
    FloatArrayType hyperarcDependentWeightNEW;

    BranchType* branchDecompostionRoot; // for traversing additional Betti number information
    std::vector<BranchType*> branches;
    std::vector<BranchType*> branchesBetti;

    if ("volume" == decompositionType)
    {
		
		using ValueType = vtkm::Float32;
		
		
		
		
		
		
		
		
		
		
	int compute_betti = 1;
	ctBetti = ct;
	
	const vtkm::Id originalRoot = ct.Rootnode;

	if (compute_betti)
	{
	  // Time branch decompostion: Volume Weight Computation:
	  vtkm::cont::Timer computeVolumeWeightsTimeDisplay;
	  computeVolumeWeightsTimeDisplay.Start();

	  // ---------------------------- FLOAT WEIGHTS ---------------------------- //

	  vtkm::cont::ArrayHandle<vtkm::Float32> superarcIntrinsicBetti;
	  vtkm::cont::ArrayHandle<vtkm::Float32> superarcDependentBetti;
	  vtkm::cont::ArrayHandle<vtkm::Float32> supernodeTransferBetti;
	  vtkm::cont::ArrayHandle<vtkm::Float32> hyperarcDependentBetti;

	  vtkm::cont::ArrayHandle<Coefficients> superarcIntrinsicWeightBetti;
	  vtkm::cont::ArrayHandle<Coefficients> superarcDependentWeightBetti;
	  vtkm::cont::ArrayHandle<Coefficients> supernodeTransferWeightBetti;
	  vtkm::cont::ArrayHandle<Coefficients> hyperarcDependentWeightBetti;


	  std::cout << YELLOW << "\n[STAGE 3.5 Start - Coeff. Weights (IDThD)] ContourTreeApp.cxx:ComputeVolumeWeightsSerialStructCoefficients START ..." << RESET << std::endl;
	  
	  using ValueType = vtkm::Float32;

	  //    CALLGRIND_START_INSTRUMENTATION;
	  worklet::contourtree_augmented::ProcessContourTree::ComputeBettiNumbersForRegularArcs(inputData,
																	  //ct,
																	  ctBetti, // compute Betti numbers on a duplicate tree, then reuse the ct branch decomp
																	  ctNumIterations,
																	  // The following four outputs are the coefficient tuples
																	  // (such as h1, h2, h3, h4 pairs)
																	  superarcIntrinsicWeightBetti,  // (output)
																	  superarcDependentWeightBetti,  // (output)
																	  supernodeTransferWeightBetti,  // (output)
																	  hyperarcDependentWeightBetti,
																	  // 2025-01-30 added additional output ...
																	  // ... to have access to "collapsed" TODO termdefine
																	  // ("collapsed" = computed single value weight, ...
																	  //  ... instead of N-length coefficient tuples)
																	  // These "collapsed" weights are used for ...
																	  // ... computing branch weights without relying on ...
																	  // ... the node count on the branches
																	  superarcIntrinsicBetti,  // (output)
																	  superarcDependentBetti,  // (output)
																	  supernodeTransferBetti,  // (output)
																	  hyperarcDependentBetti); // (output)

	  //    CALLGRIND_STOP_INSTRUMENTATION;
	  //    CALLGRIND_DUMP_STATS;

	  std::cout << YELLOW << "[STAGE 3.5 End - Coeff. Weights (IDThD)] ContourTreeApp.cxx:ComputeVolumeWeightsSerialStructCoefficients ... END\n" << RESET << std::endl;

	}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
        std::cout << "Branch Weights Chosen: VOLUME" << std::endl;
        worklet::contourtree_augmented::ProcessContourTree::ComputeVolumeWeightsSerial(
                ct,
                ctNumIterations,
                superarcIntrinsicWeight,
                superarcDependentWeight,
                supernodeTransferWeight,
                hyperarcDependentWeigh
                );

        worklet::contourtree_augmented::ProcessContourTree::ComputeVolumeBranchDecompositionSerial(
                ct,
                superarcDependentWeight,
                superarcIntrinsicWeight,
                whichBranch,
                branchMinimum,
                branchMaximum,
                branchSaddle,
                branchParent
                );
                
		auto valueField = inputData.GetPointField("values")
							//.GetDataAsDefaultFloat()
							.GetDataAsDefaultFloat()
							.AsArrayHandle<vtkm::cont::ArrayHandle<float>>();
							
		vtkm::cont::ArrayHandle<float> dataField; //= useDataSet.GetField("values").GetData();//.AsArrayHandle();
		inputData.GetField("values").GetDataAsDefaultFloat().AsArrayHandle(dataField); 
		
		bool dataFieldIsSorted = false;
		
        vtkm::Id nBranches = branchSaddle.GetNumberOfValues();
        // branches array defined above
        branches.reserve(static_cast<std::size_t>(nBranches));
                
		//BranchType* branchDecompostionRoot = defined above
		branchDecompostionRoot =
			  worklet::contourtree_augmented::ProcessContourTree::ComputeBranchDecomposition<ValueType>(
				ct.Superparents,
				ct.Supernodes,
				ct.Superarcs,
				whichBranch,
				branchMinimum,
				branchMaximum,
				branchSaddle,
				branchParent,
				ctSortOrder,
					  //useDataSet.GetPointField("values").GetDataAsDefaultFloat().AsArrayHandle<vtkm::cont::ArrayHandle<vtkm::FloatDefault>>(),
					  valueField,
				dataField, //, use sort indices
				dataFieldIsSorted,
				//superarcIntrinsicWeightNEW,   // used to use manually set values for BD: superarcIntrinsicWeightCorrect,
				//superarcDependentWeightNEW,   // used to use manually set values for BD: superarcDependentWeightCorrect );
				superarcIntrinsicWeight,
				superarcDependentWeight,
					ct.Rootnode,
					ct.SupernodeBetti, // used to get the augmented betti nodes (which are past the root node in index)
					  branches); // output
					  
					  
		        //std::vector<ValueType> originalBranchVolumeFloats;
		        std::vector<vtkm::Id> originalBranchVolumeFloats;
        originalBranchVolumeFloats.resize(branches.size());

        std::ofstream filebranchvolumes("ContourTreeBranches--BranchVolumes-NoBetti.txt");
        for(int i = 0; i < branches.size(); i++)
        {
            originalBranchVolumeFloats[i] = branches[i]->Volume;
            filebranchvolumes << i << "\t" << branches[i]->Volume << std::endl;
        }


        std::cout << "ct.root = " << ct.Rootnode << " ctBetti.root = " << ctBetti.Rootnode << " original.root = " << originalRoot << std::endl;
        std::cout << "whichBranch.size() = " << whichBranch.GetNumberOfValues()
                  << " branchSaddle.size() = " << branchSaddle.GetNumberOfValues() << std::endl;
        std::cout << "ctBetti.SupernodeBetti.size() = " << ctBetti.SupernodeBetti.GetNumberOfValues() << std::endl;
//        std::this_thread::sleep_for(std::chrono::seconds(3));

        if (compute_betti)
        {
            cont::ArrayHandle<Id> bt_superarcIntrinsicWeightNEW;
            cont::ArrayHandle<Id> bt_superarcDependentWeightNEW;
            cont::ArrayHandle<Id> bt_supernodeTransferWeightNEW;
            cont::ArrayHandle<Id> bt_hyperarcDependentWeightNEW;

            vtkm::cont::ArrayHandle<Coefficients> bt_superarcIntrinsicWeightCoeffs;
            vtkm::cont::ArrayHandle<Coefficients> bt_superarcDependentWeightCoeffs;
            vtkm::cont::ArrayHandle<Coefficients> bt_supernodeTransferWeightCoeffs;
            vtkm::cont::ArrayHandle<Coefficients> bt_hyperarcDependentWeightCoeffs;
            
			worklet::contourtree_augmented::ProcessContourTree::ComputeVolumeWeightsSerial(ctBetti,
																							ctNumIterations,
                                                                                      bt_superarcIntrinsicWeightNEW,  // (output)
                                                                                      bt_superarcDependentWeightNEW,  // (output)
                                                                                      bt_supernodeTransferWeightNEW,  // (output)
                                                                                      bt_hyperarcDependentWeightNEW); // (output)

            //worklet::contourtree_augmented::ProcessContourTree::ComputeVolumeWeightsSerialStructCoefficients(inputData,
                                                                                      //ctBetti,
                                                                                      //ctNumIterations,
                                                                                      //// The following four outputs are the coefficient tuples
                                                                                      //// (such as h1, h2, h3, h4 pairs)
                                                                                      //bt_superarcIntrinsicWeightCoeffs,  // (output)
                                                                                      //bt_superarcDependentWeightCoeffs,  // (output)
                                                                                      //bt_supernodeTransferWeightCoeffs,  // (output)
                                                                                      //bt_hyperarcDependentWeightCoeffs,
                                                                                      //// 2025-01-30 added additional output ...
                                                                                      //// ... to have access to "collapsed" TODO termdefine
                                                                                      //// ("collapsed" = computed single value weight, ...
                                                                                      ////  ... instead of N-length coefficient tuples)
                                                                                      //// These "collapsed" weights are used for ...
                                                                                      //// ... computing branch weights without relying on ...
                                                                                      //// ... the node count on the branches
                                                                                      //bt_superarcIntrinsicWeightNEW,  // (output)
                                                                                      //bt_superarcDependentWeightNEW,  // (output)
                                                                                      //bt_supernodeTransferWeightNEW,  // (output)
                                                                                      //bt_hyperarcDependentWeightNEW); // (output)



            cont::ArrayHandle<vtkm::Id> bt_whichBranch;
            cont::ArrayHandle<vtkm::Id> bt_branchMinimum;
            cont::ArrayHandle<vtkm::Id> bt_branchMaximum;
            cont::ArrayHandle<vtkm::Id> bt_branchSaddle;
            cont::ArrayHandle<vtkm::Id> bt_branchParent;

            //worklet::contourtree_augmented::ProcessContourTree::ComputeVolumeBranchDecompositionSerialFloat(ctBetti,
                                                                                      //bt_superarcDependentWeightNEW,
                                                                                      //bt_superarcIntrinsicWeightNEW,
                                                                                      //bt_whichBranch,   // (output)
                                                                                      //bt_branchMinimum, // (output)
                                                                                      //bt_branchMaximum, // (output)
                                                                                      //bt_branchSaddle,  // (output)
                                                                                      //bt_branchParent); // (output)
                                                                                      
			worklet::contourtree_augmented::ProcessContourTree::ComputeVolumeBranchDecompositionSerial(ctBetti,
                                                                                      bt_superarcDependentWeightNEW,
                                                                                      bt_superarcIntrinsicWeightNEW,
                                                                                      bt_whichBranch,   // (output)
                                                                                      bt_branchMinimum, // (output)
                                                                                      bt_branchMaximum, // (output)
                                                                                      bt_branchSaddle,  // (output)
                                                                                      bt_branchParent); // (output)

            std::cout << "bt_whichBranch.size() = " << bt_whichBranch.GetNumberOfValues() << std::endl;
//            std::this_thread::sleep_for(std::chrono::seconds(3));







            auto ctBetti_BettiOriginalSuperparents_portal = ctBetti.BettiOriginalSuperparents.ReadPortal();

            auto bt_whichBranchPortal = bt_whichBranch.WritePortal();
            auto whichBranchPortal = whichBranch.WritePortal();

//#if WRITE_FILES
            std::ofstream file1("ContourTreeGraph--original-fullCT-SN-TO-BRANCH-COLLAPSED.txt");

            for (vtkm::Id branchID = 0; branchID < whichBranch.GetNumberOfValues(); branchID++)
            {
                // std::cout << branchID << " = " << whichBranchPortal.Get(branchID) << std::endl;
                file1 << branchID << "," << whichBranchPortal.Get(branchID) << std::endl;
            }
            file1.close();
//#endif

//            for(int i = 0; i < whichBranch.GetNumberOfValues(); i++)
            for(int i = 0; i < bt_whichBranch.GetNumberOfValues(); i++)
            {
                if(i < originalRoot)
                {
//                    bt_whichBranch[i] = whichBranch[i];
                    bt_whichBranchPortal.Set(i, whichBranchPortal.Get(i));

                }
                else
                {
//                    bt_whichBranch[i] = whichBranch[ctBetti_BettiOriginalSuperparents_portal[i]];
                    bt_whichBranchPortal.Set(i, whichBranchPortal.Get(ctBetti_BettiOriginalSuperparents_portal.Get(i)));
                }
            }

            whichBranch.Allocate(bt_whichBranch.GetNumberOfValues());
            auto whichBranchPortalModified = whichBranch.WritePortal();

            for(int i = 0; i < bt_whichBranch.GetNumberOfValues(); i++)
            {
                whichBranchPortalModified.Set(i, bt_whichBranchPortal.Get(i));
            }

//#if WRITE_FILES
            std::ofstream file("ContourTreeGraph--Betti-fullCT-SN-TO-BRANCH-COLLAPSED.txt");

            for (vtkm::Id branchID = 0; branchID < whichBranch.GetNumberOfValues(); branchID++)
            {
                // std::cout << branchID << " = " << whichBranchPortal.Get(branchID) << std::endl;
                file << branchID << "," << whichBranchPortalModified.Get(branchID) << std::endl;
            }
            file.close();
//#endif










            branches.clear();

            branchDecompostionRoot =
                worklet::contourtree_augmented::ProcessContourTree::ComputeBranchDecomposition<ValueType>(
                        ctBetti.Superparents,   // contourTreeSuperparents
                        ctBetti.Supernodes,     // contourTreeSupernodes
                        ctBetti.Superarcs,      // contourTreeSuperarcs
                  whichBranch,               // whichBranch
//                        bt_whichBranch,               // whichBranch
                  branchMinimum,             // branchMinimum
                  branchMaximum,             // branchMaximum
                  branchSaddle,              // branchSaddle
                  branchParent,              // branchParent
                  ctSortOrder,                  // sortOrder
                  /* valueField */      inputData.GetPointField("values").GetDataAsDefaultFloat().AsArrayHandle<vtkm::cont::ArrayHandle<vtkm::FloatDefault>>(),
    //                    input_var,
                  dataField, //, use sort indices
                  dataFieldIsSorted,
                  bt_superarcIntrinsicWeightNEW,   // used to use manually set values for BD: superarcIntrinsicWeightCorrect,
                  bt_superarcDependentWeightNEW,   // used to use manually set values for BD: superarcDependentWeightCorrect );
                      ctBetti.Rootnode,
    //                  ct.SupernodeBetti,       // used to get the augmented betti nodes (which are past the root node in index)
                      ctBetti.SupernodeBetti,       // used to get the augmented betti nodes (which are past the root node in index)
                        branches); // output
//            branchesBetti); // output

            // overwrite branch volumes, since those can be nonsensical thanks to high floating point sensitivity
            //for (int i = 0; i < branchesBetti.size(); i++) // bug from PACT?
            for (int i = 0; i < branches.size(); i++)
            {
                branches[i]->Volume = originalBranchVolumeFloats[i];
            }



            ct = ctBetti;
//            whichBranch = bt_whichBranch;

        }			  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  
					  

    }
    else if("height" == decompositionType)
    {
        std::cout << "Branch Weights Chosen: HEIGHT" << std::endl;
        vtkm::cont::Timer timer;
        timer.Start();

        worklet::contourtree_augmented::ProcessContourTree::ComputeHeightBranchDecomposition(
                ct,
                inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<Float64>>(),
                //inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<FloatDefault>>(),
                ctSortOrder,
                ctNumIterations,
                whichBranch,
                branchMinimum,
                branchMaximum,
                branchSaddle,
                branchParent
                );
        timer.Stop();
        //std::cout << "Height branch decomposition took " << timer.GetElapsedTime() << " seconds." << std::endl;
        if (true == performanceRun)
        {
            std::cout << timer.GetElapsedTime() << std::endl;
        }
    }
    else
    {
        std::cout << "Branch Weights Chosen: NOT IMPLEMENTED" << std::endl;
    }

    // Bit hacky, but no need to do anything else if this is a performance run
    if (true == performanceRun)
    {
        return vtkm::cont::PartitionedDataSet();
    }

    //
    // Compute Additioanl Data for the Branches
    //

    // Which branch do regular vertices belong to
    vector<Id> whichBranchRegular(ct.Nodes.GetNumberOfValues());

    // Endpoints of path in the contour tree
    cont::ArrayHandle<Vec<Id, 2>> branchEndpoints;
    branchEndpoints.Allocate(branchMinimum.GetNumberOfValues());

    cont::ArrayHandle<Vec<Id, 2>> branchEndpointsRegular;
    branchEndpointsRegular.Allocate(branchMinimum.GetNumberOfValues());

    // Endpoints of path in the contour tree
    cont::ArrayHandle<Float64> branchIsovalueHeightArray;
    branchIsovalueHeightArray.Allocate(branchMinimum.GetNumberOfValues());

    // Each point has it's own isovalue
    cont::ArrayHandle<Float64> branchIsovalueArray;
    branchIsovalueArray.Allocate(branchMinimum.GetNumberOfValues());

    // Each point has it's own isovalue
    cont::ArrayHandle<Float64> branchPointVolumeArray;
    branchPointVolumeArray.Allocate(branchMinimum.GetNumberOfValues());

    // Endpoints of path in the contour tree
    cont::ArrayHandle<Id> branchHeightArray;
    branchHeightArray.Allocate(branchMinimum.GetNumberOfValues());


    // Populate the previously innitialsed arrays
    computeAdditionalBranchData(
            inputData,
            fieldName,
            ct, 
            ctSortOrder,
            whichBranch, 
            branchMinimum, 
            branchSaddle, 
            branchMaximum, 
            whichBranchRegular,
            branchEndpointsRegular,
            branchEndpoints,
            branchHeightArray,
            branchIsovalueHeightArray,
            branchIsovalueArray,
            branchPointVolumeArray,
            superarcIntrinsicWeight,
            superarcDependentWeight,
            branchIsovalueFlag
            );

    // Prevent getting more branches than we have available
    int numberOfBranches = min(static_cast<unsigned long long>(simplificationThreshold), static_cast<unsigned long long>(branchMaximum.GetNumberOfValues()));

    std::cout << "Extracting " << numberOfBranches << " most significant contours" << std::endl;

    // 
    // We also need the mesh and extrema for the extraction of IDs (for marching cubes). This only works for 3D data.
    //
    if (debugLevel >= 1)
    {
        cout << "Mesh and extrema ..." << endl;
    }
    timer.Reset();
    timer.Start();

    // Build the mesh
    vtkm::Id3 pointDimensions = inputData.GetCellSet().AsCellSet<vtkm::cont::CellSetStructured<3>>().GetPointDimensions();
    worklet::contourtree_augmented::DataSetMeshTriangulation3DMarchingCubes mesh(vtkm::Id3(pointDimensions[0], pointDimensions[1], pointDimensions[2]));
    //mesh.SortData(inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<Float64>>());
    mesh.SortData(inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<FloatDefault>>());

    // Set up extremal chains
    worklet::contourtree_augmented::MeshExtrema extrema(mesh.NumVertices);
    extrema.SetStarts(mesh, true);
    extrema.BuildRegularChains(true);
    extrema.SetStarts(mesh, false);
    extrema.BuildRegularChains(false);

    timer.Stop();

    if (debugLevel >= 1)
    {
        std::cout << "Computing Mesh and Extrema took " << timer.GetElapsedTime() << " seconds." << std::endl;
        cout << "Branch Decomposition ..." << endl;
    }


    vtkm::cont::ArrayHandle<Float64> regularValueSecondaryImportance;
    regularValueSecondaryImportance.Allocate(branchMinimum.GetNumberOfValues());

    for (int i = 0 ; i < regularValueSecondaryImportance.GetNumberOfValues() ; i++)
    {
        // We've set B to be the bigger one in computeAdditionalBranchData
        //vtkm::Id endpointBRegularID = static_cast<vtkm::Float64>(ctSortIndices.ReadPortal().Get(branchEndpoints.ReadPortal().Get(i)[1]));
        vtkm::Id endpointBRegularID = 11;
        regularValueSecondaryImportance.WritePortal().Set(i, endpointBRegularID);
    }


    // 
    // Pick the importance metric. That is either height or volume (which is independent from the simplification type)
    //
    vtkm::cont::ArrayHandle<Float64> branchImportance;
    vtkm::cont::ArrayHandle<Float64> branchImportanceSecondary;
    if ("volume" == decompositionType)
    {
        branchImportance = branchPointVolumeArray;
        branchImportanceSecondary = branchIsovalueHeightArray;
    }
    else if("height" == decompositionType)
    {
        branchImportance = branchIsovalueHeightArray;
        branchImportanceSecondary = regularValueSecondaryImportance;
    }
    else
    {
        branchImportance = branchIsovalueHeightArray;
        branchImportanceSecondary = regularValueSecondaryImportance;
    }



    // Get branch decomposition & simplify and get relevant isovalues.
    vector<std::pair<vtkm::Float64, vtkm::Id>> vals;

    if ("sort" == selectionType)
    {
        vector<Id> branchOrder = getBranchesSortedOrder(branchImportance, branchImportanceSecondary);

        for (int i = 0 ; i  < branchOrder.size() ; i++)
        {
            vals.push_back({branchIsovalueArray.ReadPortal().Get(branchOrder[i]), branchOrder[i]});
        }
    }
    else
    {
        assert(false);
    }

    //
    // Extract one contour per branch
    //

    // We could have less branches than we specified.
    numberOfBranches = std::min(static_cast<unsigned long long>(numberOfBranches), static_cast<unsigned long long>(vals.size()));

    // Prepare output

    vtkm::cont::PartitionedDataSet outputContours;
    
    
    // ==================================================================================================================================== //
    // ==================================================================================================================================== //
    // ==================================================================================================================================== //

    // SET UP (creating the CT, branch decomposition, simplification, and choosing N most important) DONE. Now actualy extracting contours  //

    // ==================================================================================================================================== //
    // ==================================================================================================================================== //
    // ==================================================================================================================================== //

    std::cout << "Branches to extract: " << numberOfBranches << " vs requested: " << vals.size()<< std::endl;

    std::cout << "extracting branches: [";
    for (int k = 0 ; k < numberOfBranches; k++)
    {
        //
        // Compute MC Triangles for each isovalue and filter out the triangles that do not belong to the current branch
        //
        std::cout << "'" << vals[k].second << "',";
    }
    std::cout << "]?" << std::endl;
    std::cout << "a" << std::endl;
    std::cout << "b" << std::endl;
    std::cout << "at isovalues: [";

    // fill in how many isosurfaces to generate (with betti changes as well)
    //               branchID  isosurface     bettiNum
    vector<std::tuple<vtkm::Id, vtkm::Float64, vtkm::Id>> flexIsosurfaces;


    //flexIsosurfaces.emplace_back(2233, 0.00101293, 48); // manual inspection 128K
    //flexIsosurfaces.emplace_back(2233, 0.00107853, 27); // manual inspection 128K
    //flexIsosurfaces.emplace_back(2233, 0.00107855, 8); // manual inspection 128K
    //flexIsosurfaces.emplace_back(2233, 0.0010836, 2); // manual inspection 128K
    flexIsosurfaces.emplace_back(2233, 0.00108369, 1); // manual inspection 128K
    flexIsosurfaces.emplace_back(2233, 0.00109757, 0); // manual inspection 128K


    vtkm::Id ring_branch = vals[1].second;//2260;
    //    flexIsosurfaces.emplace_back(ring_branch, 0.000920763, 41); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00101375, 10); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00101581, 6); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00102285, 1); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00103525, 1); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.0010549, 1); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00117022, 41); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00118316, 31); // manual inspection 200K
//    flexIsosurfaces.emplace_back(ring_branch, 0.00119409, 6); // manual inspection 200K
    
    
    
    
        for (int k = 0 ; k < numberOfBranches; k++)
    {
        std::cout << "k=" << k << std::endl;
        //
        // Compute MC Triangles for each isovalue and filter out the triangles that do not belong to the current branch
        //
        const Id branchId = vals[k].second;
        const Float64 branchIsovalue = vals[k].first;

        std::cout << branches.size() << std::endl;
        std::cout << branchId << " " << branchIsovalue << "Betti changes? : " << branches[branchId]->BettiChanges.size() << std::endl;

        vtkm::Id extremum = branches[branchId]->Extremum;
        vtkm::Id saddle = branches[branchId]->Saddle;

//        if (min < saddle)
//        {
//            endpoints = {ctSortOrder.ReadPortal().Get(min), ctSortOrder.ReadPortal().Get(saddle)};
//            regularEndpoints = {min, saddle};
//            branchHeight = saddle - min;
//            branchType = 0;
//            if (0 == branchType)
//            {
//                isovalue = b - epsilon;
//            }
//        }

        if (branches[branchId]->BettiChanges.size() == 0)
        {
            // experiment - don't add unless has betti changes
//            flexIsosurfaces.emplace_back(branchId, branchIsovalue, -1);
        }
//        else // this adds ANY top Betti change (below changed so that only generates isovalues at Betti1==2
//        {
//            flexIsosurfaces.emplace_back(branchId, branchIsovalue, -1);
//            flexIsosurfaces.emplace_back(branchId, branches[branchId]->TopBettiChangeDataValue, branches[branchId]->TopBetti1Number);
//        }

        // ALL Betti Changes
        else
        {
            // experiment - don't add unless has betti changes
//            flexIsosurfaces.emplace_back(branchId, branchIsovalue, -1);
//            if (k > 0) // skip the main branch
            {
//                vtkm::Id ring_branch = 12661; //6084; //1578; //70;
                if(branchId == ring_branch)
                {
//                    std::string filename = "";
                    std::ofstream file("ContourTreeBetti--branch-"+std::to_string(ring_branch)+"plot.txt");

                    for (vtkm::Id betti_i = 0; betti_i < branches[branchId]->BettiChanges.size(); betti_i++)
                    {
                        // std::cout << branchID << " = " << whichBranchPortal.Get(branchID) << std::endl;
                        file << betti_i << "\t" << branches[ring_branch]->Betti1Numbers[betti_i]
                                        << "\t" << branches[ring_branch]->BettiChangesDataValue[betti_i] << std::endl;
                    }
                    file.close();
                }

                for(int i = 0; i < branches[branchId]->BettiChanges.size(); i++)
                {
                    if(2 == branches[branchId]->Betti1Numbers[i])
                    {
                        flexIsosurfaces.emplace_back(branchId, branches[branchId]->BettiChangesDataValue[i], branches[branchId]->Betti1Numbers[i]);
                        break; // experiment - just get the first genus 1 surface per such branch
                    }
                }
            }
        }

//        else
//        {
//            flexIsosurfaces.emplace_back(branchId, branchIsovalue, -1);
//            for(int i = 0; i < branches[branchId]->BettiChanges.size(); i++)
//            {
////                if(branches[branchId]->Betti1Numbers[i] == 2)
//                {
//                    flexIsosurfaces.emplace_back(branchId, branches[branchId]->BettiChangesDataValue[i], branches[branchId]->Betti1Numbers[i]);
////                    vtkm::Float64 epsilon   = 0.00000005f;
////                    if ((extremum < saddle) && (saddle > 0))
////                    {
////                        flexIsosurfaces.emplace_back(branchId, branches[branchId]->BettiChangesDataValue[i]-epsilon, branches[branchId]->Betti1Numbers[i]);
////                    }
////                    else
////                    {
////                        flexIsosurfaces.emplace_back(branchId, branches[branchId]->BettiChangesDataValue[i]+epsilon, branches[branchId]->Betti1Numbers[i]);
////                    }
//                }
//            }
//        }

//        else// if (2 == branches[branchId]->TopBetti1Number)
//        {
//            flexIsosurfaces.emplace_back(branchId, branchIsovalue, -1);
//            vtkm::Float64 epsilon   = 0.000000005f;
//            if (extremum < saddle)
//            {
//                flexIsosurfaces.emplace_back(branchId, branches[branchId]->TopBettiChangeDataValue-epsilon, branches[branchId]->TopBetti1Number);
//            }
//            else
//            {
//                flexIsosurfaces.emplace_back(branchId, branches[branchId]->TopBettiChangeDataValue+epsilon, branches[branchId]->TopBetti1Number);
//            }

//        }
////        else // if no Betti1 == 2 topology changes, don't bother generating an isosurface ...
////        {
////            flexIsosurfaces.emplace_back(branchId, branchIsovalue, -1);
////        }
    }
    
    
    
    
    
    
    
    

//    for (int k = 0 ; k < numberOfBranches; k++) old version - we had one isosurface per branch, now can have betti changes within the branch
    for (int k = 0 ; k < flexIsosurfaces.size(); k++)
    {
        //
        // Compute MC Triangles for each isovalue and filter out the triangles that do not belong to the current branch
        //
        /*
        const Id branchId = vals[k].second;
        const Float64 branchIsovalue = vals[k].first;*/
        const Id branchId = std::get<0>(flexIsosurfaces[k]); //.first;
        const Float64 branchIsovalue = std::get<1>(flexIsosurfaces[k]); //flexIsosurfaces[k].second;
        const Id branchBetti = std::get<2>(flexIsosurfaces[k]); //flexIsosurfaces[k].third;

        // 
        // Print some debug info
        //
        if (debugLevel >= 1)
        {
            cout << endl << endl << "=======================================================" << std::endl;
            printf("Getting the triangles for isovalue %lf, index %i and branch ID %llu and importance %f and height %f volume %f.\n",
                                           branchIsovalue,        k,          branchId,
                                                            branchImportance.ReadPortal().Get(branchId),
                                                                  branchIsovalueHeightArray.ReadPortal().Get(branchId),
                                                                              branchPointVolumeArray.ReadPortal().Get(branchId));
        }

        // Compute an isosurface for the whole data set
        cont::ArrayHandle<cv1k::Triangle> mcTriangles = cv1k::mc::getMarchingCubeTriangles(inputData, {branchIsovalue}, fieldName);
        
        
		std::cout << "cheesing it" << std::endl;
        
        
        
// 1. Get your float data
auto floatData = inputData.GetField(fieldName).GetData();

// 2. Create a physical double array
vtkm::cont::ArrayHandle<vtkm::Float64> materializedDoubles;

// 3. Copy/Convert the data into the new buffer
vtkm::cont::ArrayCopy(floatData, materializedDoubles);

// 4. Pass the materialized array to your function
cv1k::filter::computeTriangleIds(ct, mesh, extrema, materializedDoubles, mcTriangles, branchIsovalue);
		
		
		
		std::cout << "cheesed it" << std::endl;
		
		
		
		

        // Compute the superarc ID of all the triangles
        //cv1k::filter::computeTriangleIds(ct, mesh, extrema, inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<Float64>>(), mcTriangles, branchIsovalue);
        //cv1k::filter::computeTriangleIds(ct, mesh, extrema, castArray, mcTriangles, branchIsovalue);
        //cv1k::filter::computeTriangleIds(ct, mesh, extrema, inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<FloatDefault>>(), mcTriangles, branchIsovalue);

        //
        // Compute the superarc from the current branch sits at that isovalue 
        //
        cont::ArrayHandle<Vec<Id, 2>> endpoints;
        cont::ArrayHandle<Id> superarcIds;
        cont::ArrayHandle<Float64> isovalueArray;
        //cont::ArrayHandle<FloatDefault> isovalueArray;
        endpoints.Allocate(1);
        superarcIds.Allocate(1);
        isovalueArray.Allocate(1);


        isovalueArray.WritePortal().Set(0, branchIsovalue);
        //endpoints.WritePortal().Set(0, branchEndpoints.ReadPortal().Get(branchId));
        endpoints.WritePortal().Set(0, {
                ctSortOrder.ReadPortal().Get(branchEndpointsRegular.ReadPortal().Get(branchId)[0]),
                ctSortOrder.ReadPortal().Get(branchEndpointsRegular.ReadPortal().Get(branchId)[1]),
                });

        // Set up the worklet
        vtkm::worklet::contourtree_augmented::process_contourtree_inc::SetTriangleSuperarcId setTrianglesId(ct.Hypernodes.GetNumberOfValues(), ct.Supernodes.GetNumberOfValues());
        cont::Invoker Invoke;

        //// Run the worklet
        Invoke(
                setTrianglesId,
                endpoints,
                //inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<Float64>>(),
                materializedDoubles, // not using Float:64 input data anymore
                isovalueArray,
                mesh.SortOrder, // (input)
                mesh.SortIndices, // (input)
                ct.Superparents, // (input)
                ct.WhenTransferred, // (input)
                ct.Hyperparents, // (input)
                ct.Hyperarcs, // (input)
                ct.Hypernodes, // (input)
                ct.Supernodes, // (input)
                extrema.Peaks, // (input)
                extrema.Pits,
                superarcIds
              ); // (input)

        const vtkm::Id branchSuperarcID = superarcIds.ReadPortal().Get(0);

        //
        // Filter ouf the triangles that do not belong to the isosurface of the current branch
        //
        vector<cv1k::Triangle> branchTriangles;
        for (int j = 0 ; j < mcTriangles.GetNumberOfValues() ; j++)
        {
            Triangle currentTriangle = mcTriangles.ReadPortal().Get(j);

            if(branchSuperarcID == 55)
            {
                std::cout << "currentTriangle.superarcId " << currentTriangle.superarcId << " vs "
                          << "branchSuperarcID " << branchSuperarcID << std::endl;
            }

            if (currentTriangle.superarcId == branchSuperarcID)
            {
                currentTriangle.superarcId = k;
                currentTriangle.isovalue = branchIsovalue;
                currentTriangle.importance = branchImportance.ReadPortal().Get(branchId);
                branchTriangles.push_back(currentTriangle);
            }
        }

        if (debugLevel >= 1)
        {
            cout << "The size of the contour is " << (branchTriangles.size()) << " from " << mcTriangles.GetNumberOfValues()<< " that is " <<  100.0 * (static_cast<double>(branchTriangles.size()) / static_cast<double>(mcTriangles.GetNumberOfValues())) << "%" << endl;
        }

        std::vector<vtkm::Vec3f_32> pointCoordinates;
        std::vector<vtkm::UInt8> shapes;
        std::vector<vtkm::IdComponent> numIndices;
        std::vector<vtkm::Id> connectivity;

        for (size_t i = 0 ; i < branchTriangles.size() ; i++)
        {
            for (int j = 0 ; j < 3 ; j++)
            {
                pointCoordinates.push_back({branchTriangles[i].points[j][0], branchTriangles[i].points[j][1], branchTriangles[i].points[j][2]});
            }
            shapes.push_back(vtkm::CELL_SHAPE_TRIANGLE);
            numIndices.push_back(3);
            connectivity.push_back(i * 3);
            connectivity.push_back(i * 3 + 1);
            connectivity.push_back(i * 3 + 2);
        }



        vtkm::cont::DataSetBuilderExplicit  dataSetBuilder;
        vtkm::cont::DataSet contourDataSet = dataSetBuilder.Create(pointCoordinates, shapes, numIndices, connectivity);


        // ----------------- Write Cell Data ----------------- //

        cont::ArrayHandle<int> branchIDCellField;
        branchIDCellField.Allocate(contourDataSet.GetNumberOfCells());
        auto branchIDCellFieldWritePortal = branchIDCellField.WritePortal();
        for (int i = 0 ; i < branchIDCellField.GetNumberOfValues() ; i++)
        {
            branchIDCellFieldWritePortal.Set(i, branchId);
        }
        contourDataSet.AddCellField("branchId", branchIDCellField);

        cont::ArrayHandle<int> importanceCellField;
        importanceCellField.Allocate(contourDataSet.GetNumberOfCells());
        auto importanceCellFieldWritePortal = importanceCellField.WritePortal();
        for (int i = 0 ; i < importanceCellField.GetNumberOfValues() ; i++)
        {
            importanceCellFieldWritePortal.Set(i, k);
        }
        contourDataSet.AddCellField("importance", importanceCellField);


        cont::ArrayHandle<vtkm::Float64> isovalueCellField;
        isovalueCellField.Allocate(contourDataSet.GetNumberOfCells());
        auto isovalueCellFieldWritePortal = isovalueCellField.WritePortal();
        for (int i = 0 ; i < isovalueCellField.GetNumberOfValues() ; i++)
        {
            isovalueCellFieldWritePortal.Set(i, branchIsovalue);
        }
        contourDataSet.AddCellField("branchIsovalue",    isovalueCellField);

        cont::ArrayHandle<int> volumeCellField;
        volumeCellField.Allocate(contourDataSet.GetNumberOfCells());
        auto volumeCellFieldWritePortal = volumeCellField.WritePortal();
        for (int i = 0 ; i < volumeCellField.GetNumberOfValues() ; i++)
        {
            volumeCellFieldWritePortal.Set(i, branchPointVolumeArray.ReadPortal().Get(branchId));
        }
        contourDataSet.AddCellField("branchVolume",volumeCellField);




        // ----------------- Write Point Data (shown first) ----------------- //

        cont::ArrayHandle<vtkm::Float64> branchIDPointField;
        branchIDPointField.Allocate(contourDataSet.GetNumberOfPoints());
        auto branchIDPointFieldWritePortal = branchIDPointField.WritePortal();
        for (int i = 0 ; i < branchIDPointField.GetNumberOfValues() ; i++)
        {
            branchIDPointFieldWritePortal.Set(i, branchId);//0.5);
        }

        contourDataSet.AddPointField("branchIDpt", branchIDPointField);

        cont::ArrayHandle<vtkm::Float64> isovaluePointField;
        isovaluePointField.Allocate(contourDataSet.GetNumberOfPoints());
        auto isovaluePointFieldWritePortal = isovaluePointField.WritePortal();
        for (int i = 0 ; i < isovaluePointField.GetNumberOfValues() ; i++)
        {
            isovaluePointFieldWritePortal.Set(i, branchIsovalue);//0.5);
        }

        contourDataSet.AddPointField("isovaluePoints", isovaluePointField);


        //        contourDataSet.AddCellField("branchImportance",  branchImportance.ReadPortal().Get(branchId));
        //        contourDataSet.AddCellField("branchHeight",      branchIsovalueHeightArray.ReadPortal().Get(branchId));
        outputContours.AppendPartition(contourDataSet);
    }

    return outputContours;
}//computeMostSignificantContours


vector<Id> cv1k::interface::getBranchesSortedOrder(cont::ArrayHandle<Float64> branchImportance, cont::ArrayHandle<Float64> secondaryBranchImportance)
{
    // Add only the branches which have non zero height
    vector<tuple<Id, Float64, Float64>> sortedBranches;
    for (int i = 0 ; i < branchImportance.GetNumberOfValues() ; i++)
    {
        if (branchImportance.ReadPortal().Get(i) > 0 && secondaryBranchImportance.ReadPortal().Get(i) > 0)
        {
            sortedBranches.push_back({i, branchImportance.ReadPortal().Get(i), secondaryBranchImportance.ReadPortal().Get(i)});
        }
        else
        {
        }
    }

    // Sort by first criteria, than volume, when the first criteria is volume this is redundant.
    std::sort(sortedBranches.begin(), sortedBranches.end(), [](const tuple<Id, Float64, Float64> a, const tuple<Id, Float64, Float64> b){ 
            if (std::get<1>(a) == std::get<1>(b))
            {
                return std::get<2>(a) > std::get<2>(b);
            }
            return std::get<1>(a) > std::get<1>(b);
            });

    vector<Id> sortedBranchesMap(sortedBranches.size());
    for (int i = 0 ; i < sortedBranchesMap.size() ; i++)
    {
        sortedBranchesMap[i] = std::get<0>(sortedBranches[i]);
    }

    return sortedBranchesMap;
}


void cv1k::interface::computeAdditionalBranchData(
        const cont::DataSet inputData,
        const string fieldName,
        const worklet::contourtree_augmented::ContourTree ct, 
        const vtkm::cont::ArrayHandle<vtkm::Id> ctSortOrder,
        const cont::ArrayHandle<Id> whichBranch, 
        const cont::ArrayHandle<Id> branchMinimum, 
        const cont::ArrayHandle<Id> branchSaddle, 
        const cont::ArrayHandle<Id> branchMaximum, 
        vector<Id> &whichBranchRegular,
        cont::ArrayHandle<Vec<Id, 2>> &branchEndpointsRegular,
        cont::ArrayHandle<Vec<Id, 2>> &branchEndpoints,
        cont::ArrayHandle<Id> &branchHeightArray,
        cont::ArrayHandle<Float64> &branchIsovalueHeightArray,
        cont::ArrayHandle<Float64> &branchIsovalueArray,
        cont::ArrayHandle<Float64> &branchPointVolumeArray,
        const cont::ArrayHandle<Id> superarcIntrinsicWeight, 
        const cont::ArrayHandle<Id> superarcDependentWeight,
        const std::string branchIsovalueFlag
        )
{
    // Compute the volume of every branch.
    for (int i = 0 ; i < branchPointVolumeArray.GetNumberOfValues() ; i++)
    {
        branchPointVolumeArray.WritePortal().Set(i, 1);
    }

    using vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT;
    // @TODO Should I use intrinsic of dependent weight?
    std::cout << "SuperArc to Branch ID mappings (num. mappings: "
              << superarcIntrinsicWeight.GetNumberOfValues() << "=num. of superarcs):" << std::endl;
    for (int i = 0 ; i < superarcIntrinsicWeight.GetNumberOfValues() ; i++)
    {
        Id branchId = whichBranch.ReadPortal().Get(i);

#ifdef DEBUG_PRINT
        std::cout << "SA " << i << " -> BID: " << branchId << std::endl;
#endif

        if (branchId != NO_SUCH_ELEMENT)
        {
            branchPointVolumeArray.WritePortal().Set(branchId, 1);
            Id currentWeight = branchPointVolumeArray.ReadPortal().Get(branchId);
            //Id weight = superarcIntrinsicWeight.ReadPortal().Get(i);
            Id weight = superarcDependentWeight.ReadPortal().Get(i);

            branchPointVolumeArray.WritePortal().Set(branchId, currentWeight + weight);
        }
    }

    //std::vector<std::tuple<vtkm::Id, vtkm::Id, vtkm::Id, vtkm::Float64>> branchEndpointsStd;
    std::vector<std::tuple<vtkm::Id, vtkm::Id, vtkm::Id, vtkm::FloatDefault>> branchEndpointsStd;

    // Converts the branches to mesh vertex endpoints and compute isovalue
    std::cout << "Computing isovalues for each branch (total num. isovalues: "
              << branchMinimum.GetNumberOfValues() << "=num. of branches):" << std::endl;

    for (int i = 0 ; i < branchMinimum.GetNumberOfValues() ; i++)
    {
        using vtkm::worklet::contourtree_augmented::MaskedIndex;

        Id branchHeight = 0;
        Vec<Id, 2> endpoints;
        Vec<Id, 2> regularEndpoints;

        Id max = MaskedIndex(ct.Supernodes.ReadPortal().Get(MaskedIndex(branchMaximum.ReadPortal().Get(i))));
        Id min = MaskedIndex(ct.Supernodes.ReadPortal().Get(MaskedIndex(branchMinimum.ReadPortal().Get(i))));
        Id saddle = MaskedIndex(ct.Supernodes.ReadPortal().Get(MaskedIndex(branchSaddle.ReadPortal().Get(i))));

        // 0 for descending, 1 for ascending, 2 for Master,
        int branchType = -1;

        // Determine the type of the branch and set its endpoints
        using vtkm::worklet::contourtree_augmented::NoSuchElement;
        if (false == NoSuchElement(branchSaddle.ReadPortal().Get(static_cast<vtkm::Id>(i))))
        {
            if (min < saddle)
            {
                endpoints = {ctSortOrder.ReadPortal().Get(min), ctSortOrder.ReadPortal().Get(saddle)};
                regularEndpoints = {min, saddle};
                branchHeight = saddle - min;
                branchType = 0;
            }
            else if (saddle < max)
            {
                endpoints = {ctSortOrder.ReadPortal().Get(saddle), ctSortOrder.ReadPortal().Get(max)};
                regularEndpoints = {saddle, max};
                branchHeight = max - saddle;
                branchType = 1;

            }
            else
            {
                assert(false);
            }
        }
        else
        {
            endpoints = {ctSortOrder.ReadPortal().Get(min), ctSortOrder.ReadPortal().Get(max)};
            regularEndpoints = {min, max};
            branchHeight = max - min;
            branchType = 2;
        }

        //Float64 a = inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<Float64>>().ReadPortal().Get(endpoints[0]);
        FloatDefault a = inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<FloatDefault>>().ReadPortal().Get(endpoints[0]);
        FloatDefault b = inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<FloatDefault>>().ReadPortal().Get(endpoints[1]);
        //Float64 b = inputData.GetField(fieldName).GetData().AsArrayHandle<cont::ArrayHandle<Float64>>().ReadPortal().Get(endpoints[1]);

        //Float64 isovalue = 0.0;
        FloatDefault isovalue = 0.0;

        if ("half" == branchIsovalueFlag)
        {
            isovalue = (a + b) / 2.0;
#ifdef DEBUG_PRINT
            std::cout << "Extracting at half the branch isovalue: " << isovalue
                      << ", where its end values are: " << a << " and " << b << std::endl;
#endif
        }
        else
        {
            //vtkm::Float64 epsilon = 0.00000001f;
            vtkm::FloatDefault epsilon = 0.00000001f;
            if (0 == branchType)
            {
                isovalue = b - epsilon;
#ifdef DEBUG_PRINT
                std::cout << "Extracting at branch isovalue - epsilon: " << isovalue
                          << ", where its top end value is: " << b << std::endl;
#endif
            }
            else if (1 == branchType)
            {
                isovalue = a + epsilon;
#ifdef DEBUG_PRINT
                std::cout << "Extracting at branch isovalue + epsilon: " << isovalue
                          << ", where its bottom end value is: " << a << std::endl;
#endif
            }
            else if (2 == branchType)
            {
                isovalue = a + (b - a) / 2.0;
#ifdef DEBUG_PRINT
                std::cout << "Extracting at branch isovalue + (b-a)/2: " << isovalue
                          << ", where its end values are: " << a << " and " << b << std::endl;
#endif
            }
            else
            {   // Print even without debug mode because something weird is going on
                std::cout << "Unexpected branch type encountered" << std::endl;
                assert(false);
            }

        }

#ifdef DEBUG_PRINT
        std::cout << "BID " << i << "\t range = [" << std::setw(8) << std::fixed << std::setprecision(8) << a
                  << " -> " << std::setw(8) << std::fixed <<std::setprecision(8) << b
                  << "], isovalue = " << isovalue << " (volume = " << branchPointVolumeArray.ReadPortal().Get(i)
                  << ")" << std::endl;
#endif

        branchIsovalueArray.WritePortal().Set(i, isovalue);
        branchHeightArray.WritePortal().Set(i, branchHeight);
        branchIsovalueHeightArray.WritePortal().Set(i, abs(a - b));

        branchEndpoints.WritePortal().Set(i, {endpoints[0], endpoints[1]});
        branchEndpointsRegular.WritePortal().Set(i, {regularEndpoints[0], regularEndpoints[1]});
    }
}
