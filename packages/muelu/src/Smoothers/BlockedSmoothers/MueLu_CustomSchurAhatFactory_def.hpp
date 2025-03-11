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
#ifndef MUELU_CUSTOMSCHURAHATFACTORY_DEF_HPP
#define MUELU_CUSTOMSCHURAHATFACTORY_DEF_HPP

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include "MueLu_Level.hpp"
#include "MueLu_Monitor.hpp"
#include "MueLu_Utilities.hpp"

#include "MueLu_CustomSchurAhatFactory_decl.hpp"

namespace MueLu {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CustomSchurAhatFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::CustomSchurAhatFactory() 
{ }

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
CustomSchurAhatFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::~CustomSchurAhatFactory() 
{ }

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const ParameterList> CustomSchurAhatFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
  RCP<ParameterList> validParamList = rcp(new ParameterList());
  validParamList->set<RCP<const FactoryBase>>("A", null, "Factory for original matrix A");
  return validParamList;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void CustomSchurAhatFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
  Input(currentLevel, "A");
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void CustomSchurAhatFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
  FactoryMonitor m(*this, "Computing Ahat matrix", currentLevel);
    
  RCP<Matrix> A = Factory::Get< RCP<Matrix> >(currentLevel, "A");
  RCP<BlockedCrsMatrix> bA = rcp_dynamic_cast<BlockedCrsMatrix>(A);
  TEUCHOS_TEST_FOR_EXCEPTION(bA.is_null(), Exceptions::BadCast,
                           "Input matrix A must be a BlockedCrsMatrix");

  // Extract blocks
  RCP<Matrix> A11 = bA->getMatrix(0, 0);
  RCP<Matrix> C1 = bA->getMatrix(0, 2);
  RCP<Matrix> C2 = bA->getMatrix(2, 0);
  RCP<Matrix> H = bA->getMatrix(2, 2);

  // Get inverse of H diagonal
  RCP<Vector> diagH = VectorFactory::Build(H->getRowMap());
  H->getLocalDiagCopy(*diagH);
  RCP<Vector> diagHinv = Utilities::GetInverse(diagH);

  // Compute C1 * (1/diag(H)) * C2
  RCP<Matrix> HinvC2 = MatrixFactory::BuildCopy(C2);
  HinvC2->leftScale(*diagHinv);
    
  RCP<Matrix> C1HinvC2 = MatrixFactory::BuildCopy(A11);
  MatrixMatrix::Multiply(*C1, false, *HinvC2, false, *C1HinvC2, true);

  // Compute Ahat = A11 - C1 * (1/diag(H)) * C2
  RCP<Matrix> Ahat = MatrixFactory::BuildCopy(A11);
  Xpetra::MatrixMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::TwoMatrixAdd(*C1HinvC2, false, -1.0, *A11, false, 1.0, Ahat, GetOStream(Statistics2));

  // Store result
  Set(currentLevel, "A", Ahat);
}

} // namespace MueLu

#endif /* MUELU_CUSTOMSCHURAHATFACTORY_DEF_HPP */
