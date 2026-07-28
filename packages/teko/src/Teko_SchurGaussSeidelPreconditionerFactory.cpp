// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_SchurGaussSeidelPreconditionerFactory.hpp"

#include <set>

#include "Teko_InverseLibrary.hpp"
#include "Teuchos_VerboseObject.hpp"

using Teuchos::rcp;
using Teuchos::RCP;

namespace Teko {

SchurInvFactoryDiagStrategy::SchurInvFactoryDiagStrategy(
    const std::vector<RCP<InverseFactory> >& inverseFactories,
    const std::vector<RCP<InverseFactory> >& preconditionerFactories,
    const RCP<InverseFactory>& defaultInverseFact,
    const RCP<InverseFactory>& defaultPreconditionerFact,
    const std::map<int, SchurSpec>& schurSpecs)
    : InvFactoryDiagStrategy(inverseFactories, preconditionerFactories, defaultInverseFact,
                             defaultPreconditionerFact),
      schurSpecs_(schurSpecs) {}

void SchurInvFactoryDiagStrategy::getInvD(const BlockedLinearOp& A,
                                          BlockPreconditionerState& state,
                                          std::vector<LinearOp>& invDiag) const {
  Teko_DEBUG_SCOPE("SchurInvFactoryDiagStrategy::getInvD", 10);

  size_t diagCnt = A->productRange()->numBlocks();

  // sanity check the Schur specifications against this operator
  std::map<int, SchurSpec>::const_iterator sItr;
  for (sItr = schurSpecs_.begin(); sItr != schurSpecs_.end(); ++sItr) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        sItr->first < 0 || sItr->first >= (int)diagCnt, std::runtime_error,
        "SchurInvFactoryDiagStrategy: \"Schur Type " << sItr->first + 1
            << "\" is out of range, operator has " << diagCnt << " block rows");
    TEUCHOS_TEST_FOR_EXCEPTION(
        sItr->second.fieldBlock < 0 || sItr->second.fieldBlock >= (int)diagCnt ||
            sItr->second.fieldBlock == sItr->first,
        std::runtime_error, "SchurInvFactoryDiagStrategy: \"Field Block\" "
            << sItr->second.fieldBlock + 1 << " for \"Schur Type \" " << sItr->first + 1
            << " is invalid, operator has " << diagCnt << " block rows");
  }

  const std::string opPrefix = "SchurGSDiagOp";
  for (size_t i = 0; i < diagCnt; i++) {
    sItr = schurSpecs_.find((int)i);
    if (sItr != schurSpecs_.end()) {
      invDiag.push_back(buildSchurInverse((int)i, sItr->second, A, state, opPrefix));
      continue;
    }

    auto precFact = ((i < precDiagFact_.size()) && (!precDiagFact_[i].is_null()))
                        ? precDiagFact_[i]
                        : defaultPrecFact_;
    auto invFact  = (i < invDiagFact_.size()) ? invDiagFact_[i] : defaultInvFact_;
    invDiag.push_back(buildInverse(*invFact, precFact, getBlock(i, i, A), state, opPrefix, i));
  }
}

LinearOp SchurInvFactoryDiagStrategy::buildSchurInverse(int block, const SchurSpec& spec,
                                                        const BlockedLinearOp& A,
                                                        BlockPreconditionerState& state,
                                                        const std::string& opPrefix) const {
  int field = spec.fieldBlock;

  const LinearOp F  = getBlock(field, field, A);
  const LinearOp C  = getBlock(block, field, A);
  const LinearOp Ct = getBlock(field, block, A);
  const LinearOp D  = getBlock(block, block, A);

  TEUCHOS_TEST_FOR_EXCEPTION(
      C.is_null() || isZeroOp(C) || Ct.is_null() || isZeroOp(Ct), std::runtime_error,
      "SchurInvFactoryDiagStrategy: coupling blocks (" << block << "," << field << ") and ("
          << field << "," << block << ") must be nonzero to build a Schur complement");

  std::stringstream ss;
  ss << opPrefix << "_" << block;

  // build hatS = D - C * H * Ct with H ~ inv(F)
  LinearOp hatS;
  if (spec.approxType != NotDiag) {
    // H is a diagonal approximation of inv(F): hatS is assembled explicitly
    LinearOp H    = getInvDiagonalOp(F, spec.approxType);
    LinearOp CHCt = explicitMultiply(C, H, Ct);

    LinearOp Dop = D.is_null() ? zero(CHCt->range(), CHCt->domain()) : D;
    hatS         = explicitAdd(Dop, scale(-1.0, CHCt));
  } else {
    // H is the action of inv(F): hatS stays implicit
    ModifiableLinearOp& invF = state.getModifiableOp(ss.str() + "_invF");
    if (invF == Teuchos::null)
      invF = Teko::buildInverse(*spec.fieldInvFact, F);
    else
      Teko::rebuildInverse(*spec.fieldInvFact, F, invF);

    LinearOp CHCt = multiply(C, invF.getConst(), Ct);
    if (D.is_null() || isZeroOp(D))
      hatS = scale(-1.0, CHCt);
    else
      hatS = add(D, scale(-1.0, CHCt));
  }

  ModifiableLinearOp& invS = state.getModifiableOp(ss.str() + "_invS");
  if (invS == Teuchos::null) {
    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
    *out << "Teko: \"Block Gauss-Seidel with Schur\" block " << block + 1
         << ": hatS = D - C*H*C^T of size " << hatS->range()->dim() << " against field block "
         << field + 1 << "\n"
         << "   H         = "
         << (spec.approxType != NotDiag
                 ? "inv(" + getDiagonalName(spec.approxType) + "(F)), hatS assembled explicitly"
                 : "\"" + spec.fieldInvName + "\" applied to F, hatS kept matrix-free")
         << "\n"
         << "   inv(hatS) = \"" << spec.invName << "\"" << std::endl;
    invS = Teko::buildInverse(*spec.invFact, hatS);
  } else
    Teko::rebuildInverse(*spec.invFact, hatS, invS);

  return invS;
}

SchurGaussSeidelPreconditionerFactory::SchurGaussSeidelPreconditionerFactory()
    : GaussSeidelPreconditionerFactory() {}

//! Initialize from a parameter list
void SchurGaussSeidelPreconditionerFactory::initializeFromParameterList(
    const Teuchos::ParameterList& pl) {
  Teko_DEBUG_SCOPE("SchurGaussSeidelPreconditionerFactory::initializeFromParameterList", 10);
  Teko_DEBUG_MSG_BEGIN(9);
  DEBUG_STREAM << "Parameter list: " << std::endl;
  pl.print(DEBUG_STREAM);
  Teko_DEBUG_MSG_END();

  const std::string inverse_type        = "Inverse Type";
  const std::string preconditioner_type = "Preconditioner Type";
  const std::string schur_type          = "Schur Type";
  std::vector<RCP<InverseFactory> > inverses;
  std::vector<RCP<InverseFactory> > preconditioners;
  std::map<int, SchurInvFactoryDiagStrategy::SchurSpec> schurSpecs;
  std::set<int> explicitInverses;

  RCP<const InverseLibrary> invLib = getInverseLibrary();

  // get string specifying default inverse
  std::string invStr = "";
#if defined(Teko_ENABLE_Amesos)
  invStr = "Amesos";
#elif defined(Teko_ENABLE_Amesos2)
  invStr = "Amesos2";
#endif
  std::string precStr = "None";
  if (pl.isParameter(inverse_type)) invStr = pl.get<std::string>(inverse_type);
  if (pl.isParameter(preconditioner_type)) precStr = pl.get<std::string>(preconditioner_type);
  if (pl.isParameter("Use Upper Triangle"))
    solveType_ = pl.get<bool>("Use Upper Triangle") ? GS_UseUpperTriangle : GS_UseLowerTriangle;

  RCP<InverseFactory> defaultInverse = invLib->getInverseFactory(invStr);
  RCP<InverseFactory> defaultPrec;
  if (precStr != "None") defaultPrec = invLib->getInverseFactory(precStr);

  // now check individual solvers
  Teuchos::ParameterList::ConstIterator itr;
  for (itr = pl.begin(); itr != pl.end(); ++itr) {
    std::string fieldName = itr->first;

    if (itr->second.isList() && fieldName.compare(0, schur_type.length(), schur_type) == 0) {
      int position = -1;
      std::string schur, type;

      // figure out position
      std::stringstream ss(fieldName);
      ss >> schur >> type >> position;

      TEUCHOS_TEST_FOR_EXCEPTION(position <= 0, std::runtime_error,
                                 "\"Schur Type\" must be followed by a (strictly) positive "
                                 "integer, found \"" << fieldName << "\"");

      const Teuchos::ParameterList& schurList = pl.sublist(fieldName);
      SchurInvFactoryDiagStrategy::SchurSpec spec;

      // 1-based in the XML, matching the "Inverse Type <k>" convention
      spec.fieldBlock = 0;
      if (schurList.isParameter("Field Block"))
        spec.fieldBlock = schurList.get<int>("Field Block") - 1;
      TEUCHOS_TEST_FOR_EXCEPTION(spec.fieldBlock < 0, std::runtime_error,
                                 "\"Field Block\" must be a (strictly) positive integer");

      std::string approxStr = "Diagonal";
      if (schurList.isParameter("Schur Approximation"))
        approxStr = schurList.get<std::string>("Schur Approximation");
      if (approxStr == "Solve") {
        spec.approxType = NotDiag;
        TEUCHOS_TEST_FOR_EXCEPTION(
            !schurList.isParameter("Field Inverse Type"), std::runtime_error,
            "Schur Approximation \"Solve\" requires a \"Field Inverse Type\" parameter");
        spec.fieldInvName = schurList.get<std::string>("Field Inverse Type");
        spec.fieldInvFact = invLib->getInverseFactory(spec.fieldInvName);
      } else {
        spec.approxType = getDiagonalType(approxStr);
        TEUCHOS_TEST_FOR_EXCEPTION(
            spec.approxType == NotDiag || spec.approxType == BlkDiag, std::runtime_error,
            "Unknown \"Schur Approximation\" \""
                << approxStr << "\", valid values are Diagonal, Lumped, AbsRowSum and Solve");
      }

      std::string schurInvStr = invStr;
      if (schurList.isParameter(inverse_type))
        schurInvStr = schurList.get<std::string>(inverse_type);
      spec.invName = schurInvStr;
      spec.invFact = invLib->getInverseFactory(schurInvStr);

      schurSpecs[position - 1] = spec;

      Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
      *out << "Teko: \"Block Gauss-Seidel with Schur\" block " << position
           << ": Schur complement against field block " << spec.fieldBlock + 1
           << ", approximation \"" << approxStr << "\", inverse \"" << schurInvStr << "\""
           << std::endl;
    } else if (fieldName.compare(0, inverse_type.length(), inverse_type) == 0 &&
               fieldName != inverse_type) {
      int position = -1;
      std::string inverse, type;

      // figure out position
      std::stringstream ss(fieldName);
      ss >> inverse >> type >> position;

      TEUCHOS_TEST_FOR_EXCEPTION(position <= 0, std::runtime_error,
                                 "Gauss-Seidel \"Inverse Type\" must be a (strictly) positive "
                                 "integer");

      // inserting inverse factory into vector
      std::string invStr2 = pl.get<std::string>(fieldName);
      if (position > (int)inverses.size()) inverses.resize(position, defaultInverse);
      inverses[position - 1] = invLib->getInverseFactory(invStr2);
      explicitInverses.insert(position - 1);
    } else if (fieldName.compare(0, preconditioner_type.length(), preconditioner_type) == 0 &&
               fieldName != preconditioner_type) {
      int position = -1;
      std::string preconditioner, type;

      // figure out position
      std::stringstream ss(fieldName);
      ss >> preconditioner >> type >> position;

      TEUCHOS_TEST_FOR_EXCEPTION(position <= 0, std::runtime_error,
                                 "Gauss-Seidel \"Preconditioner Type\" must be a (strictly) "
                                 "positive integer");

      // inserting preconditioner factory into vector
      std::string precStr2 = pl.get<std::string>(fieldName);
      if (position > (int)preconditioners.size()) preconditioners.resize(position, defaultPrec);
      preconditioners[position - 1] = invLib->getInverseFactory(precStr2);
    }
  }

  // a block either gets an inverse of its diagonal or a Schur complement, not both
  std::map<int, SchurInvFactoryDiagStrategy::SchurSpec>::const_iterator sItr;
  for (sItr = schurSpecs.begin(); sItr != schurSpecs.end(); ++sItr)
    TEUCHOS_TEST_FOR_EXCEPTION(explicitInverses.count(sItr->first) > 0, std::runtime_error,
                               "Block " << sItr->first + 1 << " has both \"Inverse Type "
                                        << sItr->first + 1 << "\" and \"Schur Type "
                                        << sItr->first + 1 << "\", specify only one");

  // use default inverse
  if (inverses.size() == 0) inverses.push_back(defaultInverse);

  // based on parameter type build a strategy
  invOpsStrategy_ = rcp(new SchurInvFactoryDiagStrategy(inverses, preconditioners, defaultInverse,
                                                        defaultPrec, schurSpecs));
}

}  // namespace Teko
