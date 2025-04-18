// @HEADER
// *****************************************************************************
//            NOX: An Object-Oriented Nonlinear Solver Package
//
// Copyright 2002 NTESS and the NOX contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "NOX_Direction_Newton.H" // class definition
#include "NOX_Common.H"
#include "NOX_Abstract_Vector.H"
#include "NOX_Abstract_Group.H"
#include "NOX_Solver_Generic.H"
#include "NOX_Solver_LineSearchBased.H"
#include "NOX_Utils.H"
#include "NOX_GlobalData.H"
#include "Teuchos_StandardParameterEntryValidators.hpp"

NOX::Direction::Newton::
Newton(const Teuchos::RCP<NOX::GlobalData>& gd,
       Teuchos::ParameterList& p)
{
  reset(gd, p);
}

NOX::Direction::Newton::~Newton()
{
}

bool NOX::Direction::Newton::
reset(const Teuchos::RCP<NOX::GlobalData>& gd,
      Teuchos::ParameterList& params)
{
  globalDataPtr = gd;
  utils = gd->getUtils();

  paramsPtr = &params;

  Teuchos::ParameterList& p = params.sublist("Newton");

  // Validate and set defaults
  {
    Teuchos::ParameterList validParams;
    validParams.sublist("Linear Solver").disableRecursiveValidation();
    validParams.sublist("Stratimikos Linear Solver").disableRecursiveValidation();
    validParams.set<std::string>("Forcing Term Method", "Constant",
      "Choice of forcing term used by the linear solver",
      Teuchos::rcp(new Teuchos::StringValidator(Teuchos::tuple<std::string>("Constant", "Type 1", "Type 2"))));
    validParams.set("Forcing Term Minimum Tolerance", 1.0e-4);
    validParams.set("Forcing Term Maximum Tolerance", 0.9);
    validParams.set("Forcing Term Initial Tolerance", 0.01);
    validParams.set("Forcing Term Alpha", 1.5);
    validParams.set("Forcing Term Gamma", 0.9);
    validParams.set("Rescue Bad Newton Solve", true);

    // Not used in this direction object, but needed in the Inexact
    // Newton Utils which uses the same parameter list. Need to
    // consolidate the two objects eventually.
    validParams.set("Set Tolerance in Parameter List",true);

    p.validateParametersAndSetDefaults(validParams);
  }

  doRescue = p.get<bool>("Rescue Bad Newton Solve");

  if (!p.sublist("Linear Solver").isType<double>("Tolerance"))
    p.sublist("Linear Solver").get("Tolerance", 1.0e-10);

  method = p.get<std::string>("Forcing Term Method");

  if ( method == "Constant" ) {
    useAdjustableForcingTerm = false;
    eta_k = p.sublist("Linear Solver").get<double>("Tolerance");
  }
  else {
    useAdjustableForcingTerm = true;
    eta_min = p.get<double>("Forcing Term Minimum Tolerance");
    eta_max = p.get<double>("Forcing Term Maximum Tolerance");
    eta_initial = p.get<double>("Forcing Term Initial Tolerance");
    alpha = p.get<double>("Forcing Term Alpha");
    gamma = p.get<double>("Forcing Term Gamma");
    eta_k = eta_min;
  }

  return true;
}

bool NOX::Direction::Newton::compute(NOX::Abstract::Vector& dir,
                     NOX::Abstract::Group& soln,
                     const NOX::Solver::Generic& solver)
{
  NOX::Abstract::Group::ReturnType status;

  // Compute F at current solution.
  status = soln.computeF();
  if (status != NOX::Abstract::Group::Ok)
    NOX::Direction::Newton::throwError("compute", "Unable to compute F");

  // Reset the linear solver tolerance.
  if (useAdjustableForcingTerm) {
    resetForcingTerm(soln, solver.getPreviousSolutionGroup(),
                     solver.getNumIterations(), solver);
  }
  else {
    if (utils->isPrintType(Utils::Details)) {
      utils->out() << "       CALCULATING FORCING TERM" << std::endl;
      utils->out() << "       Method: Constant" << std::endl;
      utils->out() << "       Forcing Term: " << eta_k << std::endl;
    }
  }

  // Compute Jacobian at current solution.
  status = soln.computeJacobian();
  if (status != NOX::Abstract::Group::Ok)
    NOX::Direction::Newton::throwError("compute", "Unable to compute Jacobian");

  // Compute the Newton direction
  status = soln.computeNewton(paramsPtr->sublist("Newton").sublist("Linear Solver"));

  soln.logLastLinearSolveStats(*globalDataPtr->getNonConstSolverStatistics());

  // It didn't converge, but maybe we can recover.
  if ((status != NOX::Abstract::Group::Ok) &&
      (doRescue == false)) {
    NOX::Direction::Newton::throwError("compute",
                       "Unable to solve Newton system");
  }
  else if ((status != NOX::Abstract::Group::Ok) &&
       (doRescue == true)) {
    if (utils->isPrintType(NOX::Utils::Warning))
      utils->out() << "WARNING: NOX::Direction::Newton::compute() - Linear solve "
       << "failed to achieve convergence - using the step anyway "
       << "since \"Rescue Bad Newton Solve\" is true " << std::endl;
  }

  // Set search direction.
  dir = soln.getNewton();

  return true;
}

bool NOX::Direction::Newton::compute(NOX::Abstract::Vector& dir,
                     NOX::Abstract::Group& soln,
                     const NOX::Solver::LineSearchBased& solver)
{
  return NOX::Direction::Generic::compute( dir, soln, solver );
}

// protected
bool NOX::Direction::Newton::resetForcingTerm(const NOX::Abstract::Group& soln,
                     const NOX::Abstract::Group& oldsoln,
                     int niter,
                     const NOX::Solver::Generic& solver)
{
  // Get linear solver current tolerance.
  // NOTE: These values are changing at each nonlinear iteration and
  // must be updated from the parameter list each time a reset is called!
  double eta_km1 = paramsPtr->sublist("Newton").sublist("Linear Solver")
                   .get("Tolerance", 0.0);

  // Tolerance may have been adjusted in a line search algorithm so we
  // have to account for this.
  const NOX::Solver::LineSearchBased* solverPtr = 0;
  solverPtr = dynamic_cast<const NOX::Solver::LineSearchBased*>(&solver);
  if (solverPtr != 0) {
    eta_km1 = 1.0 - solverPtr->getStepSize() * (1.0 - eta_km1);
  }

  const std::string indent = "       ";

  if (utils->isPrintType(Utils::Details)) {
    utils->out() << indent << "CALCULATING FORCING TERM" << std::endl;
    utils->out() << indent << "Method: " << method << std::endl;
  }


  if (method == "Type 1") {

    if (niter == 0) {

      eta_k = eta_initial;

    }
    else {

      // Return norm of predicted F

      // do NOT use the following lines!! This does NOT account for
      // line search step length taken.
      // const double normpredf = 0.0;
      // oldsoln.getNormLastLinearSolveResidual(normpredf);

      // Create a new vector to be the predicted RHS
      if (Teuchos::is_null(predRhs)) {
    predRhs = oldsoln.getF().clone(ShapeCopy);
      }
      if (Teuchos::is_null(stepDir)) {
    stepDir = oldsoln.getF().clone(ShapeCopy);
      }

      // stepDir = X - oldX (i.e., the step times the direction)
      stepDir->update(1.0, soln.getX(), -1.0, oldsoln.getX(), 0);

      // Compute predRhs = Jacobian * step * dir
      if (!(oldsoln.isJacobian())) {
    if (utils->isPrintType(Utils::Details)) {
      utils->out() << "WARNING: NOX::Direction::Newton::resetForcingTerm() - "
           << "Jacobian is out of date! Recomputing Jacobian." << std::endl;
    }
    const_cast<NOX::Abstract::Group&>(oldsoln).computeJacobian();
      }
      oldsoln.applyJacobian(*stepDir, *predRhs);

      // Compute predRhs = RHSVector + predRhs (this is the predicted RHS)
      predRhs->update(1.0, oldsoln.getF(), 1.0);

      // Compute the norms
      double normpredf = 0.0;
      double normf = 0.0;
      double normoldf = 0.0;

      if (utils->isPrintType(Utils::Details)) {
    utils->out() << indent << "Forcing Term Norm: Using L-2 Norm."
             << std::endl;
      }
      normpredf = predRhs->norm();
      normf = soln.getNormF();
      normoldf = oldsoln.getNormF();

      // Compute forcing term
      eta_k = fabs(normf - normpredf) / normoldf;

      // Some output
      if (utils->isPrintType(Utils::Details)) {
    utils->out() << indent << "Residual Norm k-1 =             " << normoldf << "\n";
    utils->out() << indent << "Residual Norm Linear Model k =  " << normpredf << "\n";
    utils->out() << indent << "Residual Norm k =               " << normf << "\n";
    utils->out() << indent << "Calculated eta_k (pre-bounds) = " << eta_k << std::endl;
      }

      // Impose safeguard and constraints ...
      const double tmp_alpha = (1.0 + sqrt(5.0)) / 2.0;
      const double eta_km1_alpha = pow(eta_km1, tmp_alpha);
      if (eta_km1_alpha > 0.1)
    eta_k = NOX_MAX(eta_k, eta_km1_alpha);
      eta_k = NOX_MAX(eta_k, eta_min);
      eta_k = NOX_MIN(eta_max, eta_k);
    }
  }

  else if (method == "Type 2") {

    if (niter == 0) {

      eta_k = eta_initial;

    }
    else {

      double normf = 0.0;
      double normoldf = 0.0;

      if (utils->isPrintType(Utils::Details)) {
    utils->out() << indent << "Forcing Term Norm: Using L-2 Norm."
             << std::endl;
      }
      normf = soln.getNormF();
      normoldf = oldsoln.getNormF();

      const double residual_ratio = normf / normoldf;

      eta_k = gamma * pow(residual_ratio, alpha);

      // Some output
      if (utils->isPrintType(Utils::Details)) {
    utils->out() << indent << "Residual Norm k-1 =             " << normoldf << "\n";
    utils->out() << indent << "Residual Norm k =               " << normf << "\n";
    utils->out() << indent << "Calculated eta_k (pre-bounds) = " << eta_k << std::endl;
      }

      // Impose safeguard and constraints ...
      const double eta_k_alpha = gamma * pow(eta_km1, alpha);
      if (eta_k_alpha > 0.1)
    eta_k = NOX_MAX(eta_k, eta_k_alpha);
      eta_k = NOX_MAX(eta_k, eta_min);
      eta_k = NOX_MIN(eta_max, eta_k);
    }

  }

  else {

    if (utils->isPrintType(Utils::Warning))
      utils->out() << "NOX::Direction::Newton::resetForcingTerm "
       << "- invalid forcing term method (" << method << ")" << std::endl;

    return false;
  }

  // Set the new linear solver tolerance
  paramsPtr->sublist("Newton").sublist("Linear Solver")
    .set("Tolerance", eta_k);

  if (utils->isPrintType(Utils::Details))
    utils->out() << indent << "Forcing Term: " << eta_k << std::endl;

  return true;
}

void NOX::Direction::Newton::throwError(const std::string& functionName, const std::string& errorMsg)
{
    if (utils->isPrintType(Utils::Error))
      utils->err() << "NOX::Direction::Newton::" << functionName << " - " << errorMsg << std::endl;
    throw std::runtime_error("NOX Error");
}
