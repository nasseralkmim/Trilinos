// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <unistd.h>

#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

// Xpetra
#include <Xpetra_Matrix.hpp>

// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>

#include <MueLu.hpp>

#include <MueLu_BaseClass.hpp>
#ifdef HAVE_MUELU_EXPLICIT_INSTANTIATION
#include <MueLu_ExplicitInstantiation.hpp>
#endif

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
int main_(Teuchos::CommandLineProcessor &clp, Xpetra::UnderlyingLib &lib, int argc, char *argv[]) {
#include <MueLu_UseShortNames.hpp>
  using Teuchos::ArrayView;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::TimeMonitor;

  // =========================================================================
  // MPI initialization using Teuchos
  // =========================================================================
  auto comm = Teuchos::DefaultComm<int>::getComm();

  // =========================================================================
  // Convenient definitions
  // =========================================================================

  // =========================================================================
  // Parameters initialization
  // =========================================================================
  GO nx = 100, ny = 100, nz = 100;
  Galeri::Xpetra::Parameters<GO> galeriParameters(clp, nx, ny, nz, "Laplace3D");  // manage parameters of the test case
  Xpetra::Parameters xpetraParameters(clp);                                       // manage parameters of Xpetra

  int loops = 10;
  clp.setOption("l", &loops, "Number of loops");

  clp.recogniseAllOptions(true);
  switch (clp.parse(argc, argv)) {
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED: return EXIT_SUCCESS;
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL: break;
  }

  // Retrieve matrix parameters (they may have been changed on the command line)
  // [for instance, if we changed matrix type from 2D to 3D we need to update nz]
  ParameterList galeriList = galeriParameters.GetParameterList();

  // =========================================================================
  // Problem construction
  // =========================================================================
  std::string matrixType = galeriParameters.GetMatrixType();

  RCP<const Map> map;
  if (matrixType == "Laplace1D")
    map = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian1D", comm, galeriList);
  else if (matrixType == "Laplace2D" || matrixType == "Star2D" ||
           matrixType == "BigStar2D" || matrixType == "Elasticity2D")
    map = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian2D", comm, galeriList);
  else if (matrixType == "Laplace3D" || matrixType == "Brick3D" || matrixType == "Elasticity3D")
    map = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian3D", comm, galeriList);

  RCP<Galeri::Xpetra::Problem<Map, CrsMatrixWrap, MultiVector> > Pr =
      Galeri::Xpetra::BuildProblem<SC, LO, GO, Map, CrsMatrixWrap, MultiVector>(matrixType, map, galeriList);
  RCP<Matrix> A = Pr->BuildMatrix();

  LO numRows = A->getLocalNumRows();

  RCP<TimeMonitor> tm;

  // Loop 1
  {
    GO validation = 0;

    ArrayView<const LocalOrdinal> indices;
    ArrayView<const Scalar> vals;

    tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Loop #1: Xpetra")));
    for (int i = 0; i < loops; i++) {
      for (LocalOrdinal row = 0; row < numRows; row++) {
        A->getLocalRowView(row, indices, vals);

        validation += indices.size();
      }
    }
    tm = Teuchos::null;

    std::cout << "validation = " << validation << std::endl;
  }

  // Loop 2
  {
    GO validation = 0;

    if (lib == Xpetra::UseTpetra) {
      typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tCrsMatrix;
      RCP<const tCrsMatrix> tA = toTpetra(A);
      TEUCHOS_TEST_FOR_EXCEPTION(tA.is_null(), MueLu::Exceptions::RuntimeError,
                                 "A is not a Tpetra CrsMatrix");

      typename Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::local_inds_host_view_type indices;
      typename Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>::values_host_view_type vals;

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Loop #2: Tpetra/Epetra")));
      for (int i = 0; i < loops; i++) {
        for (LocalOrdinal row = 0; row < numRows; row++) {
          tA->getLocalRowView(row, indices, vals);

          validation += indices.extent(0);
        }
      }
      tm = Teuchos::null;
    }
    if (lib == Xpetra::UseEpetra) {
#ifdef HAVE_MUELU_EPETRA
      typedef Epetra_CrsMatrix eCrsMatrix;
      RCP<const eCrsMatrix> eA = Utilities::Op2EpetraCrs(A);
      TEUCHOS_TEST_FOR_EXCEPTION(eA.is_null(), MueLu::Exceptions::RuntimeError,
                                 "A is not a Epetra CrsMatrix");

      tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Loop #2: Tpetra/Epetra")));

      for (int i = 0; i < loops; i++) {
        for (LocalOrdinal row = 0; row < numRows; row++) {
          int numEntries;
          double *eValues;
          int *eIndices;

          eA->ExtractMyRowView(row, numEntries, eValues, eIndices);

          validation += numEntries;
        }
      }
      tm = Teuchos::null;
#else
      TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::RuntimeError,
                                 "Epetra is not available");
#endif
    }
    std::cout << "validation = " << validation << std::endl;
  }

  // Loop 3
  {
    auto localMatrix = A->getLocalMatrixHost();

    GO validation = 0;

    tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Loop #3: Kokkos-serial")));
    for (int i = 0; i < loops; i++) {
      for (LocalOrdinal row = 0; row < numRows; row++) {
        auto rowView = localMatrix.row(row);
        auto length  = rowView.length;

        validation += length;
      }
    }
    tm = Teuchos::null;

    std::cout << "validation = " << validation << std::endl;
  }

  // Loop 4
  {
    auto localMatrix = A->getLocalMatrixDevice();

    GO validation = 0;

    typedef Kokkos::RangePolicy<typename NO::execution_space> RangePolicy;

    tm = rcp(new TimeMonitor(*TimeMonitor::getNewTimer("Loop #4: Kokkos-parallel (node)")));
    for (int i = 0; i < loops; i++)
      Kokkos::parallel_reduce(
          "Utils::DetectDirichletRows", RangePolicy(0, numRows),
          KOKKOS_LAMBDA(const LO row, GO &r) {
            auto rowView = localMatrix.row(row);
            auto length  = rowView.length;

            r += length;
          },
          validation);
    tm = Teuchos::null;

    std::cout << "validation = " << validation << std::endl;
  }

  {
    const bool alwaysWriteLocal = false;
    const bool writeGlobalStats = true;
    const bool writeZeroTimers  = false;
    const bool ignoreZeroTimers = true;
    const std::string filter    = "";
    TimeMonitor::summarize(comm.ptr(), std::cout, alwaysWriteLocal, writeGlobalStats,
                           writeZeroTimers, Teuchos::Union, filter, ignoreZeroTimers);
  }

  return EXIT_SUCCESS;
}

//- -- --------------------------------------------------------
#define MUELU_AUTOMATIC_TEST_ETI_NAME main_
#include "MueLu_Test_ETI.hpp"

int main(int argc, char *argv[]) {
  return Automatic_Test_ETI(argc, argv);
}
