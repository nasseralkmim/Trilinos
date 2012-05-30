/** \file  example_IntrepidPoisson_Pamgen_Tpetra.cpp
    \brief Example setup of a discretization of a Poisson equation on
           a hexahedral mesh using nodal (Hgrad) elements.  The
           system is assembled but not solved.

           This example uses the following Trilinos packages:
    \li    Pamgen to generate a Hexahedral mesh.
    \li    Sacado to form the source term from user-specified manufactured solution.
    \li    Intrepid to build the discretization matrix and right-hand side.
    \li    Tpetra to handle the global sparse matrix and dense vector.

    \verbatim

     Poisson system:

            div A grad u = f in Omega
                       u = g on Gamma

      where
             A is a material tensor (typically symmetric positive definite)
             f is a given source term

     Corresponding discrete linear system for nodal coefficients(x):

                 Kx = b

            K - HGrad stiffness matrix
            b - right hand side vector

    \endverbatim

    \author Created by P. Bochev, D. Ridzal, K. Peterson,
            D. Hensinger, C. Siefert.  Converted to Tpetra by
            I. Kalashnikova (ikalash@sandia.gov).  Modified by Mark
            Hoemmen (mhoemme@sandia.gov).

    \remark Use the "--help" command-line argument for usage info.

    \remark Example driver has an option to use an input file (XML
            serialization of a Teuchos::ParameterList) containing a
            Pamgen mesh description.  A version, Poisson.xml, is
            included in the same directory as this driver.  If not
            included, this program will use a default mesh
            description.

    \remark The exact solution (u) and material tensor (A) are set in
            the functions "exactSolution" and "materialTensor" and may
            be modified by the user.  We compute the source term f
            from u using Sacado automatic differentiation, so that u
            really is the exact solution.  The current implementation
            of exactSolution() has notes to guide your choice of
            solution.  For example, you might want to pick a solution
            in the finite element space, so that the discrete solution
            is exact if the solution of the linear system is exact
            (modulo rounding error when constructing the linear
            system).
*/

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "TrilinosCouplings_TpetraIntrepidPoissonExample.hpp"
#include "TrilinosCouplings_IntrepidPoissonExampleHelpers.hpp"


int
main (int argc, char *argv[])
{
  using namespace TrilinosCouplings; // Yes, this means I'm lazy.

  using TpetraIntrepidPoissonExample::exactResidualNorm;
  using TpetraIntrepidPoissonExample::makeMatrixAndRightHandSide;
  using IntrepidPoissonExample::makeMeshInput;
  using IntrepidPoissonExample::setCommandLineArgumentDefaults;
  using IntrepidPoissonExample::setUpCommandLineArguments;
  using IntrepidPoissonExample::parseCommandLineArguments;
  using Tpetra::DefaultPlatform;
  using Teuchos::Comm;
  using Teuchos::outArg;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcpFromRef;
  using Teuchos::getFancyOStream;
  using Teuchos::FancyOStream;
  using std::endl;
  // Pull in typedefs from the IntrepidExample namespace.
  typedef TpetraIntrepidPoissonExample::ST ST;
  typedef TpetraIntrepidPoissonExample::Node Node;
  typedef Teuchos::ScalarTraits<ST> STS;
  typedef STS::magnitudeType MT;
  typedef TpetraIntrepidPoissonExample::sparse_matrix_type sparse_matrix_type;
  typedef TpetraIntrepidPoissonExample::vector_type vector_type;

  Teuchos::oblackholestream blackHole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackHole);
  const int myRank = mpiSession.getRank ();
  //const int numProcs = mpiSession.getNProc ();

  // Get the default communicator and Kokkos Node instance
  RCP<const Comm<int> > comm =
    DefaultPlatform::getDefaultPlatform ().getComm ();
  RCP<Node> node = DefaultPlatform::getDefaultPlatform ().getNode ();

  // Did the user specify --help at the command line to print help
  // with command-line arguments?
  bool printedHelp = false;
  // Values of command-line arguments.
  int nx, ny, nz;
  std::string xmlInputParamsFile;
  bool verbose, debug;

  // Set default values of command-line arguments.
  setCommandLineArgumentDefaults (nx, ny, nz, xmlInputParamsFile,
                                  verbose, debug);
  // Parse and validate command-line arguments.
  Teuchos::CommandLineProcessor cmdp (false, true);
  setUpCommandLineArguments (cmdp, nx, ny, nz, xmlInputParamsFile,
                             verbose, debug);
  parseCommandLineArguments (cmdp, printedHelp, argc, argv, nx, ny, nz,
                             xmlInputParamsFile, verbose, debug);
  if (printedHelp) {
    // The user specified --help at the command line to print help
    // with command-line arguments.  We printed help already, so quit
    // with a happy return code.
    return EXIT_SUCCESS;
  }

  // Both streams only print on MPI Rank 0.  "out" only prints if the
  // user specified --verbose.
  RCP<FancyOStream> out =
    getFancyOStream (rcpFromRef ((myRank == 0 && verbose) ? std::cout : blackHole));
  RCP<FancyOStream> err =
    getFancyOStream (rcpFromRef ((myRank == 0 && debug) ? std::cerr : blackHole));

#ifdef HAVE_MPI
  *out << "PARALLEL executable" << endl;
#else
  *out << "SERIAL executable" << endl;
#endif

/**********************************************************************************/
/********************************** GET XML INPUTS ********************************/
/**********************************************************************************/

  // Get Pamgen mesh definition string, either from the input
  // ParameterList or from our function that makes a cube and fills in
  // the number of cells along each dimension.
  std::string meshInput;
  if (xmlInputParamsFile == "") {
    meshInput = makeMeshInput (nx, ny, nz);
  }
  else {
    // Read ParameterList from XML file.
    ParameterList inputMeshList;
    *out << "Reading mesh parameters from XML file \""
         << xmlInputParamsFile << "\"..." << endl;
    // FIXME (mfh 24 May 2012) If this only reads parameters on Proc
    // 0, the check for the "meshInput" parameter below will be
    // broken!
    Teuchos::updateParametersFromXmlFile (xmlInputParamsFile, outArg (inputMeshList));
    if (myRank == 0) {
      inputMeshList.print (*out, 2, true, true);
      *out << endl;
    }
    // If the input ParameterList has a "meshInput" parameter, use
    // that, otherwise use our function.
    if (inputMeshList.isParameter ("meshInput")) {
      // Get Pamgen mesh definition string from the input ParameterList.
      // There's nothing special about the "meshInput" parameter.
      // Pamgen itself doesn't read ParameterLists, but takes a string
      // of commands as input.  We've just chosen to make Pamgen's input
      // string a parameter in the input ParameterList.
      meshInput = inputMeshList.get<std::string> ("meshInput");
    }
    else {
      meshInput = makeMeshInput (nx, ny, nz);
    }
  }

  RCP<sparse_matrix_type> A;
  RCP<vector_type> B, X_exact, X;
  makeMatrixAndRightHandSide (A, B, X_exact, X, comm, node, meshInput,
                              out, err, verbose, debug);

  const std::vector<MT> norms = exactResidualNorm (A, B, X_exact);
  // X_exact is the exact solution of the PDE, projected onto the
  // discrete mesh.  It may not necessarily equal the exact solution
  // of the linear system.
  *out << "||B - A*X_exact||_2 = " << norms[0] << endl
       << "||B||_2 = " << norms[1] << endl
       << "||A||_F = " << norms[2] << endl;

  // Summarize timings
  Teuchos::TimeMonitor::summarize (*out);
  return EXIT_SUCCESS;
}
