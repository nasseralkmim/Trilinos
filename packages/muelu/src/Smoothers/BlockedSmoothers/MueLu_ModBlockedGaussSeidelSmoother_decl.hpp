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

#ifndef MUELU_MODBLOCKEDGAUSSSEIDELSMOOTHER_DECL_HPP_
#define MUELU_MODBLOCKEDGAUSSSEIDELSMOOTHER_DECL_HPP_

#include "MueLu_SmootherPrototype.hpp"
#include "MueLu_FactoryManagerBase.hpp"
#include "MueLu_CustomSchurAhatFactory.hpp"
#include "MueLu_VerbosityLevel.hpp"

namespace MueLu {

  template <class Scalar,class LocalOrdinal, class GlobalOrdinal, class Node>
  class ModBlockedGaussSeidelSmoother : public SmootherPrototype<Scalar,LocalOrdinal,GlobalOrdinal,Node> {
#include "MueLu_UseShortNames.hpp"
public:

    //! @name Constructor/Destructor
    //@{

    /*! \brief Constructor */
    ModBlockedGaussSeidelSmoother();

    //! Destructor
    virtual ~ModBlockedGaussSeidelSmoother();
    //@}


    //! @name Overridden from SmootherPrototype
    //@{

    //! Return the list of valid (and required) parameters accepted by this smoother.
    RCP<const Teuchos::ParameterList> GetValidParameterList() const;

    //! Declare the data required to build the smoother
    void DeclareInput(Level &currentLevel) const;

    //! Setup the smoother
    void Setup(Level &currentLevel);

    //! Apply the smoother to a given problem.
    void Apply(MultiVector& X, const MultiVector& B, bool InitialGuessIsZero = false) const;

    // //! Return a smoother copy
    // RCP< SmootherPrototype<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Copy() const;

    //! Returns a string describing the object
    std::string description() const;

    //! Print information about this object to out
    void print(Teuchos::FancyOStream &out, const VerbLevel verbLevel = Default) const;

    //! Return estimate of smoother complexity per node
    size_t getNodeSmootherComplexity() const;
    //@}


    //! @name Specific methods for Blocked Gauss-Seidel smoother
    //@{

    /*! \brief Set FactoryManager for Ahat smoother */
    void SetAhatFactoryManager(RCP<const FactoryManagerBase> FactManager);

    /*! \brief Add FactoryManager for subblocks of blocked operator A */
    void AddFactoryManager(RCP<const FactoryManagerBase> FactManager, int pos);
    //@}

  private:
    //! @name Private methods
    //@{
    //@}


    //! @name Private data
    //@{
    std::string type_;
    RCP<Matrix> A_;

    //! @note Inverse_[i] is the inverse of the (i,i) block of A
    std::vector< RCP<const SmootherBase> > Inverse_;
    std::vector< bool > bIsBlockedOperator_;
    std::vector< RCP<Vector> > diagAInvVector_;

    //! @note FactManager_[i] is the FactoryManager for the i-th block row of blocked operator A
    std::vector< Teuchos::RCP<const FactoryManagerBase> > FactManager_;

    RCP<const SmootherBase> AhatSmoother_;
    RCP<const FactoryManagerBase> AhatFactoryManager_;

    RCP<const MapExtractor> rangeMapExtractor_;
    RCP<const MapExtractor> domainMapExtractor_;
    //@}
  };

} // namespace MueLu

#define MUELU_MODBLOCKEDGAUSSSEIDELSMOOTHER_DECL_HPP_
#endif
