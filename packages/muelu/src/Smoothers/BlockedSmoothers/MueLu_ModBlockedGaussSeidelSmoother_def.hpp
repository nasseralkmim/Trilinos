// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

#ifndef MUELU_MODBLOCKEDGAUSSSEIDELSMOOTHER_DEF_HPP_
#define MUELU_MODBLOCKEDGAUSSSEIDELSMOOTHER_DEF_HPP_

#include "MueLu_CoupledRBMFactory_decl.hpp"
#include "Teuchos_ArrayViewDecl.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include "MueLu_ConfigDefs.hpp"

#include <Xpetra_BlockReorderManager.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_ReorderedBlockedCrsMatrix.hpp>
#include <Xpetra_ReorderedBlockedMultiVector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <iostream>
#include <ostream>
#include <sstream>

#include "MueLu_ModBlockedGaussSeidelSmoother_decl.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_Utilities.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_HierarchyUtils.hpp"
#include "MueLu_SmootherBase.hpp"

namespace MueLu {

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::ModBlockedGaussSeidelSmoother()
    : type_("modified blocked GaussSeidel"), A_(Teuchos::null)
  {
    FactManager_.reserve(10); // TODO fix me!
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::~ModBlockedGaussSeidelSmoother() {}

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set< RCP<const FactoryBase> >("A",                  Teuchos::null, "Generating factory of the matrix A");
    validParamList->set< Scalar >                ("Damping factor",     1.0, "Damping/Scaling factor in BGS");
    validParamList->set< LocalOrdinal >          ("Sweeps",             1, "Number of BGS sweeps (default = 1)");
    validParamList->set<bool>("UseSIMPLE", false, "Use SIMPLE to correct displacement field (default = false)");
    validParamList->set<bool>("UseEigenDamping", false, "Use max eingevalue estimate to damp SIMPLE correction (default = false)");
    validParamList->set<bool>("UseSIMPLEC", false, "Use SIMPLEC to correct displacement field (default = false)");
    validParamList->set<bool>("UpperTriangular", false, "Use upper triangular instead of lower (default = false)");
    validParamList->set<bool>("UseDiagInverse", false, "Use diagonal inverse in the SIMPLE (default = false)");
    validParamList->set<bool>("UseSIMPLEUL", false, "Use SIMPLE to correct damage field (default = false)");

    return validParamList;
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::AddFactoryManager(RCP<const FactoryManagerBase> FactManager, int pos) {
    TEUCHOS_TEST_FOR_EXCEPTION(pos < 0, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::AddFactoryManager: parameter \'pos\' must not be negative! error.");

    size_t myPos = Teuchos::as<size_t>(pos);

    if (myPos < FactManager_.size()) {
      // replace existing entries in FactManager_ vector
      FactManager_.at(myPos) = FactManager;
    } else if(myPos == FactManager_.size()) {
      // append new Factory manager at the end of the vector
      FactManager_.push_back(FactManager);
    } else { // if(myPos > FactManager_.size())
      RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      *out << "Warning: cannot add new FactoryManager at proper position " << pos << ". The FactoryManager is just appended to the end. Check this!" << std::endl;

      // add new Factory manager in the end of the vector
      FactManager_.push_back(FactManager);
    }
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &currentLevel) const {
    // only works for 3x3 block
    // TEUCHOS_TEST_FOR_EXCEPTION(FactManager_.size() != 3, Exceptions::RuntimeError,"MueLu::ModBlockedGaussSeidelSmoother::DeclareInput: You have to declare two FactoryManagers with a \"Smoother\" object: One for predicting the primary variable and one for the SchurComplement system. The smoother for the SchurComplement system needs a SchurComplementFactory as input for variable \"A\". make sure that you use the same proper damping factors for omega both in the SchurComplementFactory and in the SIMPLE smoother!");
    
    //this->Input(currentLevel, "A");
    // TODO: check me: why is this->Input not freeing properly A in release mode?
    currentLevel.DeclareInput("A",this->GetFactory("A").get());

    // loop over all factory managers for the subblocks of blocked operator A
    std::vector<Teuchos::RCP<const FactoryManagerBase> >::const_iterator it;
    for(it = FactManager_.begin(); it!=FactManager_.end(); ++it) {
      SetFactoryManager currentSFM  (rcpFromRef(currentLevel),   *it);

      // request "Smoother" for current subblock row.
      currentLevel.DeclareInput("PreSmoother",(*it)->GetFactory("Smoother").get());

      // request "A" for current subblock row (only needed for Thyra mode)
      currentLevel.DeclareInput("A",(*it)->GetFactory("A").get());
    }
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Setup(Level &currentLevel) {

    RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

    FactoryMonitor m(*this, "Setup blocked Gauss-Seidel Smoother", currentLevel);
    if (SmootherPrototype::IsSetup() == true) this->GetOStream(Warnings0) << "MueLu::ModBlockedGaussSeidelSmoother::Setup(): Setup() has already been called";

    // extract blocked operator A from current level
    A_ = Factory::Get< RCP<Matrix> >(currentLevel, "A"); // A needed for extracting map extractors
    RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A_);
    TEUCHOS_TEST_FOR_EXCEPTION(bA==Teuchos::null, Exceptions::BadCast, "MueLu::BlockedPFactory::Build: input matrix A is not of type BlockedCrsMatrix! error.");

    // plausibility check
    int blockSize_ = FactManager_.size();
    TEUCHOS_TEST_FOR_EXCEPTION(bA->Rows() != blockSize_, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Setup: number of block rows of A is " << bA->Rows() << " and does not match number of SubFactoryManagers " << blockSize_ << ". error.");
    TEUCHOS_TEST_FOR_EXCEPTION(bA->Cols() != blockSize_, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Setup: number of block cols of A is " << bA->Cols() << " and does not match number of SubFactoryManagers " << blockSize_ << ". error.");

    // store map extractors
    rangeMapExtractor_  = bA->getRangeMapExtractor();
    domainMapExtractor_ = bA->getDomainMapExtractor();

    // loop over all factory managers for the subblocks of blocked operator A
    std::vector<Teuchos::RCP<const FactoryManagerBase> >::const_iterator it;
    for(it = FactManager_.begin(); it!=FactManager_.end(); ++it) {
      SetFactoryManager currentSFM  (rcpFromRef(currentLevel), *it);

      // extract Smoother for current block row (BGS ordering)
      RCP<const SmootherBase> Smoo = currentLevel.Get< RCP<SmootherBase> >("PreSmoother",(*it)->GetFactory("Smoother").get());
      Inverse_.push_back(Smoo);

      // store whether subblock matrix is blocked or not!
      RCP<Matrix> Aii = currentLevel.Get< RCP<Matrix> >("A",(*it)->GetFactory("A").get());
      bIsBlockedOperator_.push_back(Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(Aii)!=Teuchos::null);
    }

    // Resize according to the number of blocks
    diagAInvVector_.resize(blockSize_);
    
    const ParameterList & pL = Factory::GetParameterList();
    bool useSIMPLE = pL.get<bool>("UseSIMPLE");
    if (useSIMPLE) {
      *out << "Using modBGS with SIMPLE-like algorithm for: " << blockSize_ << " blocks"  << std::endl;
      for (int i = 0; i < blockSize_; i++) {
        Teuchos::RCP<Vector> AiiDiag = VectorFactory::Build(bA->getMatrix(i, i)->getRowMap());
        bA->getMatrix(i, i)->getLocalDiagCopy(*AiiDiag);
        diagAInvVector_[i] = Utilities::GetInverse(AiiDiag);
      }
    }
    bool useSIMPLEC = pL.get<bool>("UseSIMPLEC");
    if (useSIMPLEC) {
      *out << "Using modBGS with SIMPLEC-like algorithm" << std::endl;
      for (int i = 0; i < blockSize_; i++) {
        Teuchos::RCP<Vector> AiiDiag = Utilities::GetLumpedMatrixDiagonal(*bA->getMatrix(i, i));
        bA->getMatrix(i, i)->getLocalDiagCopy(*AiiDiag);
        diagAInvVector_[i] = Utilities::GetInverse(AiiDiag);
      }
    }
    bool useSIMPLEUL = pL.get<bool>("UseSIMPLEUL");
    if (useSIMPLEUL) {
      *out << "Using modBGS with SIMPLEUL-like algorithm for: " << blockSize_ << " blocks"  << std::endl;
      for (int i = 0; i < blockSize_; i++) {
        Teuchos::RCP<Vector> AiiDiag = VectorFactory::Build(bA->getMatrix(i, i)->getRowMap());
        bA->getMatrix(i, i)->getLocalDiagCopy(*AiiDiag);
        diagAInvVector_[i] = Utilities::GetInverse(AiiDiag);
      }
    }
    // use eigenvalue damping only with "SIMPLE" approaches
    bool useEigenDamping = pL.get<bool>("UseEigenDamping");
    if ((useSIMPLE || useSIMPLEC) && useEigenDamping) {
      Scalar AlambdaMax = bA->getMatrix(0, 0)->GetMaxEigenvalueEstimate();
      Scalar DlambdaMax = bA->getMatrix(1, 1)->GetMaxEigenvalueEstimate();
      *out << "Using eigenvalue damping in SIMPLE-like algorithm" << std::endl;
      *out << "A lambdaMax: " << AlambdaMax << std::endl;
      *out << "D lambdaMax: " << DlambdaMax << std::endl;
    }

    bool useDiagInv = pL.get<bool>("UseDiagInverse");
    if ((useSIMPLE || useSIMPLEC) && useDiagInv) {
      *out << "Using same diagonal inverse approximation in the SIMPLE-like algorithm" << std::endl;
    }

    SmootherPrototype::IsSetup(true);
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Apply(MultiVector &X, const MultiVector& B, bool InitialGuessIsZero) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(SmootherPrototype::IsSetup() == false, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): Setup() has not been called");

#if 0 // def HAVE_MUELU_DEBUG
    // TODO simplify this debug check
    RCP<MultiVector> rcpDebugX = Teuchos::rcpFromRef(X);
    RCP<const MultiVector> rcpDebugB = Teuchos::rcpFromRef(B);
    RCP<BlockedMultiVector> rcpBDebugX = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(rcpDebugX);
    RCP<const BlockedMultiVector> rcpBDebugB = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(rcpDebugB);
    //RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A_);
    if(rcpBDebugB.is_null() == false) {
      //this->GetOStream(Runtime1) << "BlockedGaussSeidel: B is a BlockedMultiVector of size " << B.getMap()->getGlobalNumElements() << " with " << rcpBDebugB->getBlockedMap()->getNumMaps() << " blocks." << std::endl;
      //TEUCHOS_TEST_FOR_EXCEPTION(A_->getRangeMap()->isSameAs(*(B.getMap())) == false, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): The map of RHS vector B is not the same as range map of the blocked operator A. Please check the map of B and A.");
    } else {
      //this->GetOStream(Runtime1) << "BlockedGaussSeidel: B is a MultiVector of size " << B.getMap()->getGlobalNumElements() << std::endl;
      //TEUCHOS_TEST_FOR_EXCEPTION(bA->getFullRangeMap()->isSameAs(*(B.getMap())) == false, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): The map of RHS vector B is not the same as range map of the blocked operator A. Please check the map of B and A.");
    }
    if(rcpBDebugX.is_null() == false) {
      //this->GetOStream(Runtime1) << "BlockedGaussSeidel: X is a BlockedMultiVector of size " << X.getMap()->getGlobalNumElements() << " with " << rcpBDebugX->getBlockedMap()->getNumMaps() << " blocks." << std::endl;
      //TEUCHOS_TEST_FOR_EXCEPTION(A_->getDomainMap()->isSameAs(*(X.getMap())) == false, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): The map of the solution vector X is not the same as domain map of the blocked operator A. Please check the map of X and A.");
    } else {
      //this->GetOStream(Runtime1) << "BlockedGaussSeidel: X is a MultiVector of size " << X.getMap()->getGlobalNumElements() << std::endl;
      //TEUCHOS_TEST_FOR_EXCEPTION(bA->getFullDomainMap()->isSameAs(*(X.getMap())) == false, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): The map of the solution vector X is not the same as domain map of the blocked operator A. Please check the map of X and A.");
    }
#endif
    SC zero = Teuchos::ScalarTraits<SC>::zero(), one = Teuchos::ScalarTraits<SC>::one();
    
    // extract parameters from internal parameter list
    const ParameterList & pL = Factory::GetParameterList();
    LocalOrdinal nSweeps = pL.get<LocalOrdinal>("Sweeps");
    Scalar omega = pL.get<Scalar>("Damping factor");
    
    // The boolean flags check whether we use Thyra or Xpetra style GIDs
    bool bRangeThyraMode = rangeMapExtractor_->getThyraMode(); //  && (Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(F_) == Teuchos::null);
    bool bDomainThyraMode = domainMapExtractor_->getThyraMode(); // && (Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(F_) == Teuchos::null);

    // Input variables used for the rest of the algorithm
    RCP<MultiVector> rcpX = Teuchos::rcpFromRef(X);
    RCP<const MultiVector> rcpB = Teuchos::rcpFromRef(B);

    // make sure that both rcpX and rcpB are BlockedMultiVector objects
    bool bCopyResultX = false;
    RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A_);
    MUELU_TEST_FOR_EXCEPTION(bA.is_null() == true, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): A_ must be a BlockedCrsMatrix");
    RCP<BlockedMultiVector> bX = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(rcpX);
    RCP<const BlockedMultiVector> bB = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(rcpB);

    if(bX.is_null() == true) {
      RCP<MultiVector> test = Teuchos::rcp(new BlockedMultiVector(bA->getBlockedDomainMap(),rcpX));
      rcpX.swap(test);
      bCopyResultX = true;
    }

    if(bB.is_null() == true) {
      RCP<const MultiVector> test = Teuchos::rcp(new BlockedMultiVector(bA->getBlockedRangeMap(),rcpB));
      rcpB.swap(test);
    }

    // we now can guarantee that X and B are blocked multi vectors
    bX = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(rcpX);
    bB = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(rcpB);

    // check the type of operator
    RCP<Xpetra::ReorderedBlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > rbA = Teuchos::rcp_dynamic_cast<Xpetra::ReorderedBlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(bA);
    if(rbA.is_null() == false) {
      // A is a ReorderedBlockedCrsMatrix
      Teuchos::RCP<const Xpetra::BlockReorderManager > brm = rbA->getBlockReorderManager();

      // check type of X vector
      if(bX->getBlockedMap()->getNumMaps() != bA->getDomainMapExtractor()->NumMaps()) {
        // X is a blocked multi vector but incompatible to the reordered blocked operator A
        Teuchos::RCP<MultiVector> test =
            buildReorderedBlockedMultiVector(brm, bX);
        rcpX.swap(test);
      }
      if(bB->getBlockedMap()->getNumMaps() != bA->getRangeMapExtractor()->NumMaps()) {
        // B is a blocked multi vector but incompatible to the reordered blocked operator A
        Teuchos::RCP<const MultiVector> test =
            buildReorderedBlockedMultiVector(brm, bB);
        rcpB.swap(test);
      }
    }


    // check the type of operator
    /*RCP<BlockedCrsMatrix> bA = Teuchos::rcp_dynamic_cast<BlockedCrsMatrix>(A_);
    MUELU_TEST_FOR_EXCEPTION(bA.is_null() == true, Exceptions::RuntimeError, "MueLu::ModBlockedGaussSeidelSmoother::Apply(): A_ must be a BlockedCrsMatrix");
    RCP<Xpetra::ReorderedBlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > rbA = Teuchos::rcp_dynamic_cast<Xpetra::ReorderedBlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(bA);
    if(rbA.is_null() == false) {
      // A is a ReorderedBlockedCrsMatrix
      Teuchos::RCP<const Xpetra::BlockReorderManager > brm = rbA->getBlockReorderManager();

      // check type of vectors
      RCP<BlockedMultiVector> bX = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(rcpX);
      RCP<const BlockedMultiVector> bB = Teuchos::rcp_dynamic_cast<const BlockedMultiVector>(rcpB);

      // check type of X vector
      if(bX.is_null() == false && bX->getBlockedMap()->getNumMaps() != bA->getDomainMapExtractor()->NumMaps()) {
      RCP<ReorderedBlockedMultiVector> rbX = Teuchos::rcp_dynamic_cast<ReorderedBlockedMultiVector>(bX);
      if(rbX.is_null() == true) {
      // X is a blocked multi vector but not reordered
      // However, A is a reordered blocked operator
      // We have to make sure, that A and X use compatible maps
      Teuchos::RCP<MultiVector> test =
      buildReorderedBlockedMultiVector(brm, bX);
      rcpX.swap(test);
      }
      }
      if(bB.is_null() == false && bB->getBlockedMap()->getNumMaps() != bA->getRangeMapExtractor()->NumMaps()) {
      RCP<const ReorderedBlockedMultiVector> rbB = Teuchos::rcp_dynamic_cast<const ReorderedBlockedMultiVector>(bB);
      if(rbB.is_null() == true) {
      // B is a blocked multi vector but not reordered
      // However, A is a reordered blocked operator
      // We have to make sure, that A and X use compatible maps
      Teuchos::RCP<const MultiVector> test =
      buildReorderedBlockedMultiVector(brm, bB);
      rcpB.swap(test);
      }
      }
      }*/

    // Throughout the rest of the algorithm rcpX and rcpB are used for solution vector and RHS

    if (FactManager_.size() == 3) {

      // Decide if use SIMPLE with Upper triangular or usual lower triangualr block Gauss-Seidel with or without simple
      bool useSIMPLEUL = pL.get<bool>("UseSIMPLEUL");
      if (useSIMPLEUL) {
        
        RCP<MultiVector> residual = MultiVectorFactory::Build(rcpB->getMap(), rcpB->getNumVectors());
        RCP<BlockedMultiVector> bresidual = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(residual);
        RCP<MultiVector> r1 = bresidual->getMultiVector(0,bRangeThyraMode);
        RCP<MultiVector> r2 = bresidual->getMultiVector(1,bRangeThyraMode);
        RCP<MultiVector> r3 = bresidual->getMultiVector(2,bRangeThyraMode);

        // helper vector 1 (preliminary, intermediary calculation)
        RCP<MultiVector> x_p = MultiVectorFactory::Build(rcpX->getMap(), rcpX->getNumVectors());
        RCP<BlockedMultiVector> bx_p = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(x_p);
        RCP<MultiVector> x_p1 = bx_p->getMultiVector(0, bDomainThyraMode);
        RCP<MultiVector> x_p2 = bx_p->getMultiVector(1, bDomainThyraMode);
        RCP<MultiVector> x_p3 = bx_p->getMultiVector(2, bDomainThyraMode);

        // helper vector 2
        RCP<MultiVector> xhat = MultiVectorFactory::Build(rcpX->getMap(), rcpX->getNumVectors());
        RCP<BlockedMultiVector> bxhat = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(xhat);
        RCP<MultiVector> xhat1 = bxhat->getMultiVector(0, bDomainThyraMode);
        RCP<MultiVector> xhat2 = bxhat->getMultiVector(1, bDomainThyraMode);
        RCP<MultiVector> xhat3 = bxhat->getMultiVector(2, bDomainThyraMode);

        RCP<MultiVector> tempres = MultiVectorFactory::Build(rcpB->getMap(), rcpB->getNumVectors());

        // Clear solution from previos V cycles in case it is still stored
        if( InitialGuessIsZero==true )
          rcpX->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
    
        // incrementally improve solution vector X
        for (LocalOrdinal run = 0; run < nSweeps; ++run) {
          // calculate current residual = B - A rcpX
          residual->update(one, *rcpB, zero); // residual = B
          if(InitialGuessIsZero == false || run > 0)
            A_->apply(*rcpX, *residual, Teuchos::NO_TRANS, -one, one);

          // start from 0 
          x_p1->putScalar(zero);
          x_p2->putScalar(zero);
          x_p3->putScalar(zero);

          // FIRST step, apply the preconditioner to the damage field
          Inverse_.at(2)->Apply(*x_p3, *r3);

          // Compute the RHS for displacement and microrotation fields
          RCP<MultiVector> displ_RHS = rangeMapExtractor_->getVector(0, rcpB->getNumVectors(), bRangeThyraMode);
          RCP<MultiVector> microrotation_RHS =
            rangeMapExtractor_->getVector(1, rcpB->getNumVectors(), bRangeThyraMode);
          
          RCP<Matrix> C1 = bA->getMatrix(0, 2);
          C1->apply(*x_p3, *displ_RHS);
          displ_RHS->update(one, *r1, -one);
          
          RCP<Matrix> F1 = bA->getMatrix(1, 2);
          F1->apply(*x_p3, *microrotation_RHS);
          microrotation_RHS->update(one, *r2, -one);

          // Solve the intermediate problem for the displacement field and microrotation
          // TODO: this is a temporary solution, we need to implement a smoother for the Schur complement.
          // NOTE: using the smoother for A^{-1} instead of a smoother for the Schur
          // complement (A - C1 H^{-1} C2)^{-1}.
          Inverse_.at(0)->Apply(*x_p1, *displ_RHS);

          RCP<Matrix> B2 = bA->getMatrix(1, 0);
          RCP<Matrix> C2 = bA->getMatrix(1, 2);
          
          // Build Schur complement for block B2: Schur_B2 = B2 - F1 diagAinvVector[2] C2
          RCP<Matrix> Hinv_C2 = MatrixFactory::BuildCopy(C2, false);
          Hinv_C2->leftScale(*diagAInvVector_[2]);
          RCP<Matrix> F1_Hinv_C2 = MatrixFactory::BuildCopy(B2, false);
          Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Multiply(*F1, false, *Hinv_C2, false, *F1_Hinv_C2, true, true);
          RCP<Matrix> Schur_B2 = MatrixFactory::BuildCopy(B2, false);
          Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::TwoMatrixAdd(*F1_Hinv_C2, false, -one, *B2, false, one, Schur_B2, this->GetOStream(Statistics2));

          // Build a temporary RHS: temp_RHS = microrotation_RHS - Schur_B2 x_p1
          RCP<MultiVector> temp_RHS = rangeMapExtractor_->getVector(1, rcpB->getNumVectors(), bRangeThyraMode);
          Schur_B2->apply(*x_p1, *temp_RHS);
          temp_RHS->update(one, *microrotation_RHS, -one);
          
          // TODO: this is a temporary solution, we need to implement a smoother for the Schur complement.
          // NOTE: using the smoother for D^{-1} instead of a smoother for the Schur
          // complement (D - F1 H^{-1} F2)^{-1}.
          Inverse_.at(1)->Apply(*x_p2, *temp_RHS);

          // Store solution
          xhat1->update(one, *x_p1, zero);
          xhat2->update(one, *x_p2, zero);

          // Compute correction for damage
          RCP<MultiVector> Hinv_C2_x_p1 = domainMapExtractor_->getVector(2, rcpX->getNumVectors(), bDomainThyraMode);
          RCP<MultiVector> Hinv_F2_x_p2 = domainMapExtractor_->getVector(2, rcpX->getNumVectors(), bDomainThyraMode);
          // Compute Hinv_C2_x_p1 = diagAInvVecto_[2] C2 x_p1
          C2->apply(*x_p1, *Hinv_C2_x_p1);
          Hinv_C2_x_p1->elementWiseMultiply(1.0, *diagAInvVector_[2], *Hinv_C2_x_p1, 0.0);
          // Compute Hinv_F2_x_p2 = diagAInvVector_[2] F2 x_p2
          F1->apply(*x_p2, *Hinv_F2_x_p2);
          Hinv_F2_x_p2->elementWiseMultiply(1.0, *diagAInvVector_[2], *Hinv_F2_x_p2, 0.0);

          // Compute the correction for the damage field
          // xhat3 = x_p3 - Hinv_F2_x_p2 - Hinv_C2_x_p1
          xhat3->update(one, *x_p3, zero);
          xhat3->update(-one, *Hinv_F2_x_p2, one);
          xhat3->update(-one, *Hinv_C2_x_p1, one);
          
          // 7. Update solution vector
          rcpX->update(one, *bxhat, one); // x^{k+1} = x^{k} + xhat
        }
      
      }
      else {
      
        RCP<MultiVector> residual = MultiVectorFactory::Build(rcpB->getMap(), rcpB->getNumVectors());
        RCP<BlockedMultiVector> bresidual = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(residual);
        RCP<MultiVector> r1 = bresidual->getMultiVector(0,bRangeThyraMode);
        RCP<MultiVector> r2 = bresidual->getMultiVector(1,bRangeThyraMode);
        RCP<MultiVector> r3 = bresidual->getMultiVector(2,bRangeThyraMode);

        // helper vector 1
        RCP<MultiVector> xtilde = MultiVectorFactory::Build(rcpX->getMap(), rcpX->getNumVectors());
        RCP<BlockedMultiVector> bxtilde = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(xtilde);
        RCP<MultiVector> xtilde1 = bxtilde->getMultiVector(0, bDomainThyraMode);
        RCP<MultiVector> xtilde2 = bxtilde->getMultiVector(1, bDomainThyraMode);
        RCP<MultiVector> xtilde3 = bxtilde->getMultiVector(2, bDomainThyraMode);

        // helper vector 2
        RCP<MultiVector> xhat = MultiVectorFactory::Build(rcpX->getMap(), rcpX->getNumVectors());
        RCP<BlockedMultiVector> bxhat = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(xhat);
        RCP<MultiVector> xhat1 = bxhat->getMultiVector(0, bDomainThyraMode);
        RCP<MultiVector> xhat2 = bxhat->getMultiVector(1, bDomainThyraMode);
        RCP<MultiVector> xhat3 = bxhat->getMultiVector(2, bDomainThyraMode);

        RCP<MultiVector> tempres = MultiVectorFactory::Build(rcpB->getMap(), rcpB->getNumVectors());

        // Clear solution from previos V cycles in case it is still stored
        if( InitialGuessIsZero==true )
          rcpX->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
    
        // incrementally improve solution vector X
        for (LocalOrdinal run = 0; run < nSweeps; ++run) {
          // 1. calculate current residual = B - A rcpX
          residual->update(one, *rcpB, zero); // residual = B
          if(InitialGuessIsZero == false || run > 0)
            A_->apply(*rcpX, *residual, Teuchos::NO_TRANS, -one, one);

          // 2. Solve A_{11} \Delta \tilde{x}_1 = r_1
          xtilde1->putScalar(zero);
          xtilde2->putScalar(zero);
          xtilde3->putScalar(zero);
          Inverse_.at(0)->Apply(*xtilde1, *r1);

          // 3. Compute the RHS for the second sub-problem using the solution \tilde{x}_1.
          //     rhs2 = r_2 - B_2 \Delta \tilde{x}_1  with B_2 = A_{21}
          RCP<MultiVector> schurCompRHS = rangeMapExtractor_->getVector(1, rcpB->getNumVectors(), bRangeThyraMode);
          bA->getMatrix(1, 0)->apply(*xtilde1, *schurCompRHS);
          schurCompRHS->update(one, *r2, -one);

          // 4. Solve this second problem considering specific Schur complement approximation by S = A_{22}
          Inverse_.at(1)->Apply(*xtilde2, *schurCompRHS);

          // 5. Solve the third problem considering it independent from the others
          Inverse_.at(2)->Apply(*xtilde3, *r3);

          // 6. Store \tilde{x}
          // \hat{x}: correction after one iteration
          xhat3->update(one, *xtilde3, zero);
      
          bool useSIMPLE = pL.get<bool>("UseSIMPLE");
          bool useSIMPLEC = pL.get<bool>("UseSIMPLEC");
          if (useSIMPLE || useSIMPLEC) {
            // correct analogous to SIMPLE, using \Delta \tilde{x}_2 and \Delta \tilde{x}_3
            RCP<MultiVector> B1_xtilde2 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);
            RCP<MultiVector> C1_xtilde3 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);
            RCP<MultiVector> F1_xtilde3 = domainMapExtractor_->getVector(1, rcpX->getNumVectors(), bDomainThyraMode);
            RCP<MultiVector> Dinv_F1_xtilde3 = domainMapExtractor_->getVector(1, rcpX->getNumVectors(), bDomainThyraMode);
            RCP<MultiVector> B1_Dinv_F1_xtilde3 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);

            Scalar AdampingFactor;
            Scalar DdampingFactor;
            bool useEigenDamping = pL.get<bool>("UseEigenDamping");
            if (useEigenDamping) {
              Scalar AlambdaMax = bA->getMatrix(0, 0)->GetMaxEigenvalueEstimate();
              Scalar DlambdaMax = bA->getMatrix(1, 1)->GetMaxEigenvalueEstimate();
              AdampingFactor = omega / AlambdaMax;
              DdampingFactor = omega / DlambdaMax;
              RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
              /* *out << "AdampingFactor" << AdampingFactor << std::endl; */
              /* *out << "DdampingFactor" << DdampingFactor << std::endl; */
            } else {
              AdampingFactor = omega;
              DdampingFactor = omega;
            }

            // first update \Delta \tilde{x}_2 = \Delta \tilde{x}_2 - A22inv F1 \Delta \tilde{x}_3
            bA->getMatrix(1, 2)->apply(*xhat3, *F1_xtilde3);

            bool useDiagInv = pL.get<bool>("UseDiagInverse");
            if (useDiagInv) {
              RCP<MultiVector> xhat2_temp1 = domainMapExtractor_->getVector(1, rcpX->getNumVectors(), bDomainThyraMode);
              Inverse_.at(1)->Apply(*xhat2_temp1, *F1_xtilde3);
              xhat2->update(DdampingFactor, *xhat2_temp1, zero);
            } else {
              xhat2->elementWiseMultiply(DdampingFactor, *diagAInvVector_[1], *F1_xtilde3, zero);
              Dinv_F1_xtilde3->elementWiseMultiply(one, *diagAInvVector_[1], *F1_xtilde3, zero);
            }
            xhat2->update(one, *xtilde2, -one);
        
            // use the updated xhat2 to update \Delta \tilde{x}_1
            bA->getMatrix(0, 1)->apply(*xhat2, *B1_xtilde2);
            bA->getMatrix(0, 2)->apply(*xhat3, *C1_xtilde3);
            bA->getMatrix(0, 1)->apply(*Dinv_F1_xtilde3, *B1_Dinv_F1_xtilde3);

            // since omega was already applied to \tilde{x}_2, we use 1 here
            if (useDiagInv) {
              RCP<MultiVector> xhat1_temp1 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);
              RCP<MultiVector> xhat1_temp2 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);
              Inverse_.at(0)->Apply(*xhat1_temp1, *B1_xtilde2);
              Inverse_.at(0)->Apply(*xhat1_temp2, *C1_xtilde3);
              xhat1->update(AdampingFactor, *xhat1_temp1, zero);
              xhat1->update(AdampingFactor, *xhat1_temp2, one);
            } else {
              xhat1->elementWiseMultiply(AdampingFactor, *diagAInvVector_[0], *B1_xtilde2, zero);
              xhat1->elementWiseMultiply(AdampingFactor, *diagAInvVector_[0], *C1_xtilde3, one);
              xhat1->elementWiseMultiply(AdampingFactor, *diagAInvVector_[0], *B1_Dinv_F1_xtilde3, -one);
            }
            xhat1->update(one, *xtilde1, -one); // \Delta \tilde{x}_1 - Ainv B_1 \Delta \tilde{x}_2

          } else {
            xhat2->update(one, *xtilde2, zero);
            xhat1->update(one, *xtilde1, zero);
          }

          // 7. Update solution vector
          rcpX->update(one, *bxhat, one); // x^{k+1} = x^{k} + xhat
        }
      
      }      

    } else if (FactManager_.size() == 2) {
      
           RCP<MultiVector> residual = MultiVectorFactory::Build(rcpB->getMap(), rcpB->getNumVectors());
      RCP<BlockedMultiVector> bresidual = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(residual);
      RCP<MultiVector> r1 = bresidual->getMultiVector(0,bRangeThyraMode);
      RCP<MultiVector> r2 = bresidual->getMultiVector(1,bRangeThyraMode);

           // helper vector 1
           RCP<MultiVector> xtilde = MultiVectorFactory::Build(rcpX->getMap(), rcpX->getNumVectors());
           RCP<BlockedMultiVector> bxtilde = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(xtilde);
           RCP<MultiVector> xtilde1 = bxtilde->getMultiVector(0, bDomainThyraMode);
           RCP<MultiVector> xtilde2 = bxtilde->getMultiVector(1, bDomainThyraMode);

           // helper vector 2
           RCP<MultiVector> xhat = MultiVectorFactory::Build(rcpX->getMap(), rcpX->getNumVectors());
           RCP<BlockedMultiVector> bxhat = Teuchos::rcp_dynamic_cast<BlockedMultiVector>(xhat);
           RCP<MultiVector> xhat1 = bxhat->getMultiVector(0, bDomainThyraMode);
           RCP<MultiVector> xhat2 = bxhat->getMultiVector(1, bDomainThyraMode);

           // Clear solution from previos V cycles in case it is still stored
           if( InitialGuessIsZero==true )
             rcpX->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
    
           // incrementally improve solution vector X
           for (LocalOrdinal run = 0; run < nSweeps; ++run) {
             // 1. calculate current residual = B - A rcpX
             residual->update(one, *rcpB, zero); // residual = B
             if(InitialGuessIsZero == false || run > 0)
               A_->apply(*rcpX, *residual, Teuchos::NO_TRANS, -one, one);

             // 2. Solve A_{11} \Delta \tilde{x}_1 = r_1
             xtilde1->putScalar(zero);
             xtilde2->putScalar(zero);
             Inverse_.at(0)->Apply(*xtilde1, *r1);

             // 3. Compute the RHS for the second sub-problem using the solution \tilde{x}_1.
             //     rhs2 = r_2 - B_2 \Delta \tilde{x}_1  with B_2 = A_{21}
             RCP<MultiVector> schurCompRHS = rangeMapExtractor_->getVector(1, rcpB->getNumVectors(), bRangeThyraMode);
             bA->getMatrix(1, 0)->apply(*xtilde1, *schurCompRHS);
             schurCompRHS->update(one, *r2, -one);

             // 4. Solve this second problem considering specific Schur complement approximation by S = A_{22}
             Inverse_.at(1)->Apply(*xtilde2, *schurCompRHS);
             // Update the solution for the second problem
             xhat2->update(one, *xtilde2, zero);

             // SIMPLE-like correction for the first problem variable
             bool useSIMPLE = pL.get<bool>("UseSIMPLE");
             bool useSIMPLEC = pL.get<bool>("UseSIMPLEC");
             if (useSIMPLE || useSIMPLEC) {
               // correct analogous to SIMPLE, using \Delta \tilde{x}_2 and \Delta \tilde{x}_3
               RCP<MultiVector> B1_xtilde2 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);

               Scalar AdampingFactor;
               Scalar DdampingFactor;
               bool useEigenDamping = pL.get<bool>("UseEigenDamping");
               if (useEigenDamping) {
                 Scalar AlambdaMax = bA->getMatrix(0, 0)->GetMaxEigenvalueEstimate();
                 Scalar DlambdaMax = bA->getMatrix(1, 1)->GetMaxEigenvalueEstimate();
                 AdampingFactor = omega / AlambdaMax;
                 DdampingFactor = omega / DlambdaMax;
                 RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
                 /* *out << "AdampingFactor" << AdampingFactor << std::endl; */
                 /* *out << "DdampingFactor" << DdampingFactor << std::endl; */
               } else {
                 AdampingFactor = omega;
                 DdampingFactor = omega;
               }

               // Update based on damping factor
               xhat2->update(DdampingFactor, *xhat2, zero);
          
               // use the updated xhat2 to update \Delta \tilde{x}_1
               bA->getMatrix(0, 1)->apply(*xhat2, *B1_xtilde2);

               bool useDiagInv = pL.get<bool>("UseDiagInverse");
               if (useDiagInv) {
                 RCP<MultiVector> xhat1_temp1 = domainMapExtractor_->getVector(0, rcpX->getNumVectors(), bDomainThyraMode);
                 Inverse_.at(0)->Apply(*xhat1_temp1, *B1_xtilde2);
                 xhat1->update(AdampingFactor, *xhat1_temp1, zero);
               } else {
                 xhat1->elementWiseMultiply(AdampingFactor, *diagAInvVector_[0], *B1_xtilde2, zero);
               }
               xhat1->update(one, *xtilde1, -one);

             } else {
               xhat1->update(one, *xtilde1, zero);
             }

             // 7. Update solution vector
             rcpX->update(one, *bxhat, one); // x^{k+1} = x^{k} + xhat
           }
   
    }
    
    if (bCopyResultX == true) {
           RCP<MultiVector> Xmerged = bX->Merge();
      X.update(one, *Xmerged, zero);
    }
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<MueLu::SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node> > ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Copy() const {
    return rcp( new ModBlockedGaussSeidelSmoother(*this) );
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  std::string ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
    std::ostringstream out;
    out << SmootherPrototype::description();
    out << "{type = " << type_ << "}";
    return out.str();
  }

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  void ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::print(Teuchos::FancyOStream &out, const VerbLevel verbLevel) const {
    MUELU_DESCRIBE;

    // extract parameters from internal parameter list
    const ParameterList & pL = Factory::GetParameterList();
    LocalOrdinal nSweeps = pL.get<LocalOrdinal>("Sweeps");
    Scalar omega = pL.get<Scalar>("Damping factor");

    if (verbLevel & Parameters0) {
      out0 << "Prec. type: " << type_ << " Sweeps: " << nSweeps << " damping: " << omega << std::endl;
    }

    if (verbLevel & Debug) {
      out0 << "IsSetup: " << Teuchos::toString(SmootherPrototype::IsSetup()) << std::endl;
    }
  }
  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  size_t ModBlockedGaussSeidelSmoother<Scalar, LocalOrdinal, GlobalOrdinal, Node>::getNodeSmootherComplexity() const {
    // FIXME: This is a placeholder
    return Teuchos::OrdinalTraits<size_t>::invalid();
  }

} // namespace MueLu

#endif /* MUELU_MODBLOCKEDGAUSSSEIDELSMOOTHER_DEF_HPP_ */
