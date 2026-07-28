// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef __Teko_SchurGaussSeidelPreconditionerFactory_hpp__
#define __Teko_SchurGaussSeidelPreconditionerFactory_hpp__

#include <map>

#include "Teko_GaussSeidelPreconditionerFactory.hpp"

namespace Teko {

/** A diagonal-inverse strategy identical to InvFactoryDiagStrategy except
 * that selected diagonal blocks are replaced by an approximate Schur
 * complement with respect to a designated "field" block. For a Schur
 * block \f$i\f$ with field block \f$p\f$ the operator inverted is
 *
 *    \f$ \hat S_i = A_{ii} - A_{ip} H A_{pi} \f$
 *
 * where \f$H \approx A_{pp}^{-1}\f$ is either a diagonal approximation
 * (Diagonal, Lumped, AbsRowSum) or the action of an inverse factory
 * applied to \f$A_{pp}\f$ ("Solve"). In the diagonal case \f$\hat S_i\f$
 * is assembled explicitly and any inverse factory can be used to invert
 * it. In the "Solve" case \f$\hat S_i\f$ is an implicit operator and the
 * inverse factory must only require applications of the operator (e.g.
 * a Belos solver).
 */
class SchurInvFactoryDiagStrategy : public InvFactoryDiagStrategy {
 public:
  //! Specification of one Schur-complement diagonal block
  struct SchurSpec {
    //! 0-based index of the field block to eliminate
    int fieldBlock = 0;
    //! Diagonal approximation of inv(A_pp); NotDiag indicates "Solve" mode
    DiagonalType approxType = Diagonal;
    //! Inverse factory for A_pp, used only in "Solve" mode
    Teuchos::RCP<InverseFactory> fieldInvFact;
    //! Inverse factory applied to the approximate Schur complement
    Teuchos::RCP<InverseFactory> invFact;
    //! User-facing names of the factories above, for screen output
    std::string fieldInvName;
    std::string invName;
  };

  SchurInvFactoryDiagStrategy(
      const std::vector<Teuchos::RCP<InverseFactory> > &inverseFactories,
      const std::vector<Teuchos::RCP<InverseFactory> > &preconditionerFactories,
      const Teuchos::RCP<InverseFactory> &defaultInverseFact,
      const Teuchos::RCP<InverseFactory> &defaultPreconditionerFact,
      const std::map<int, SchurSpec> &schurSpecs);

  virtual ~SchurInvFactoryDiagStrategy() {}

  /** returns an (approximate) inverse for each diagonal block of A,
   * substituting the approximate Schur complement inverse for blocks
   * with a SchurSpec.
   */
  virtual void getInvD(const BlockedLinearOp &A, BlockPreconditionerState &state,
                       std::vector<LinearOp> &invDiag) const;

 protected:
  //! Build the inverse of the approximate Schur complement for block i
  LinearOp buildSchurInverse(int block, const SchurSpec &spec, const BlockedLinearOp &A,
                             BlockPreconditionerState &state, const std::string &opPrefix) const;

  std::map<int, SchurSpec> schurSpecs_;
};

/** \brief A block Gauss-Seidel factory where selected diagonal blocks are
 *        replaced by approximate Schur complements.
 *
 * This behaves exactly like "Block Gauss-Seidel" except that a diagonal
 * block may be specified through a "Schur Type k" sublist (k is the 1-based
 * block index, following the "Inverse Type k" convention) instead of an
 * "Inverse Type k" parameter. This is intended for saddle-point-type fields
 * (e.g. Lagrange multipliers) whose diagonal block is zero, where the plain
 * sweep would need an identity substitute.
 *
    \verbatim
    <ParameterList name="BGS-Schur">
       <Parameter name="Type" type="string" value="Block Gauss-Seidel with Schur"/>
       <Parameter name="Inverse Type 1" type="string" value="AMG-Disp"/>
       <ParameterList name="Schur Type 5">
          <!-- 1-based index of the block eliminated to form the Schur
               complement (default 1) -->
          <Parameter name="Field Block" type="int" value="1"/>
          <!-- Diagonal, Lumped, AbsRowSum: explicit hatS from a diagonal
               approximation of the field block. Solve: implicit hatS using
               "Field Inverse Type" as inv(A_pp). -->
          <Parameter name="Schur Approximation" type="string" value="Diagonal"/>
          <!-- Only for "Solve" -->
          <!-- <Parameter name="Field Inverse Type" type="string" value="Amesos2"/> -->
          <!-- Inverse used for hatS itself; must be apply-only capable
               (e.g. Belos) when "Schur Approximation" is "Solve" -->
          <Parameter name="Inverse Type" type="string" value="Amesos2"/>
       </ParameterList>
    </ParameterList>
    \endverbatim
 *
 * Specifying both "Inverse Type k" and "Schur Type k" for the same k is an
 * error.
 */
class SchurGaussSeidelPreconditionerFactory : public GaussSeidelPreconditionerFactory {
 public:
  SchurGaussSeidelPreconditionerFactory();

 protected:
  //! Initialize from a parameter list
  virtual void initializeFromParameterList(const Teuchos::ParameterList &pl);
};

}  // end namespace Teko

#endif
