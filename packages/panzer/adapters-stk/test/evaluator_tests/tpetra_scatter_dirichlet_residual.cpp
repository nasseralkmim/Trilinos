// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_DOFManager.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_PureBasis.hpp"
#include "Panzer_BasisIRLayout.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_GatherOrientation.hpp"
#include "Panzer_ScatterDirichletResidual_Tpetra.hpp"
#include "Panzer_GlobalEvaluationDataContainer.hpp"
#include "Panzer_LOCPair_GlobalEvaluationData.hpp"
#include "Panzer_ParameterList_GlobalEvaluationData.hpp"

#include "Panzer_STK_Version.hpp"
#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STKConnManager.hpp"

#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_OpaqueWrapper.hpp"

#include "Thyra_VectorStdOps.hpp"
#include "Thyra_ProductVectorBase.hpp"
#include "Thyra_SpmdVectorBase.hpp"

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"

#include "user_app_EquationSetFactory.hpp"

#include <cstdio> // for get char
#include <vector>
#include <string>

#include "Panzer_Evaluator_WithBaseImpl.hpp"

namespace panzer
{

   using TpetraLinObjFactoryType = panzer::TpetraLinearObjFactory<panzer::Traits, double, panzer::LocalOrdinal, panzer::GlobalOrdinal>;
   using TpetraLinObjContainerType = panzer::TpetraLinearObjContainer<double, panzer::LocalOrdinal, panzer::GlobalOrdinal>;
   using Tpetra_CrsMatrix = Tpetra::CrsMatrix<double, panzer::LocalOrdinal, panzer::GlobalOrdinal>;
   using Thyra_TpetraLinearOp = Thyra::TpetraLinearOp<double, LocalOrdinal, GlobalOrdinal>;

   Teuchos::RCP<panzer::PureBasis> buildBasis(std::size_t worksetSize, const std::string &basisName);
   void testInitialization(const Teuchos::RCP<Teuchos::ParameterList> &ipb);
   Teuchos::RCP<panzer_stk::STK_Interface> buildMesh(int elemX, int elemY);

   TEUCHOS_UNIT_TEST(assembly, scatter_dirichlet_residual)
   {

#ifdef HAVE_MPI
      Teuchos::RCP<Teuchos::MpiComm<int>> tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
#else
      NOPE_PANZER_DOESNT_SUPPORT_SERIAL
#endif

      int myRank = tComm->getRank();

      const std::size_t workset_size = 4;
      const std::string fieldName1_q1 = "U";
      const std::string fieldName2_q1 = "V";
      const std::string fieldName_qedge1 = "B";

      Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh(2, 2);

      // build input physics block
      Teuchos::RCP<panzer::PureBasis> basis_q1 = buildBasis(workset_size, "Q1");
      Teuchos::RCP<panzer::PureBasis> basis_qedge1 = buildBasis(workset_size, "QEdge1");

      Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList();
      testInitialization(ipb);

      const int default_int_order = 1;
      std::string eBlockID = "eblock-0_0";
      Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
      panzer::CellData cellData(workset_size, mesh->getCellTopology("eblock-0_0"));
      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
      Teuchos::RCP<panzer::PhysicsBlock> physicsBlock =
          Teuchos::rcp(new PhysicsBlock(ipb, eBlockID, default_int_order, cellData, eqset_factory, gd, false));

      Teuchos::RCP<std::vector<panzer::Workset>> work_sets = panzer_stk::buildWorksets(*mesh, physicsBlock->elementBlockID(),
                                                                                       physicsBlock->getWorksetNeeds());
      TEST_EQUALITY(work_sets->size(), 1);

      std::vector<bool> scatter_IC_vec = {false,true};

      for (const bool scatter_IC : scatter_IC_vec) {
         // build connection manager and field manager
         const Teuchos::RCP<panzer::ConnManager> conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
         RCP<panzer::DOFManager> dofManager = Teuchos::rcp(new panzer::DOFManager(conn_manager, MPI_COMM_WORLD));

         dofManager->addField(fieldName1_q1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_q1->getIntrepid2Basis())));
         dofManager->addField(fieldName2_q1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_q1->getIntrepid2Basis())));
         dofManager->addField(fieldName_qedge1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_qedge1->getIntrepid2Basis())));

         std::vector<std::string> fieldOrder;
         fieldOrder.push_back(fieldName1_q1);
         fieldOrder.push_back(fieldName_qedge1);
         fieldOrder.push_back(fieldName2_q1);
         dofManager->setFieldOrder(fieldOrder);

         dofManager->buildGlobalUnknowns();

         // setup linear object factory
         /////////////////////////////////////////////////////////////
         Teuchos::RCP<TpetraLinObjFactoryType> t_lof = Teuchos::rcp(new TpetraLinObjFactoryType(tComm.getConst(), dofManager));
         Teuchos::RCP<LinearObjFactory<panzer::Traits>> lof = t_lof;
         Teuchos::RCP<LinearObjContainer> loc = t_lof->buildGhostedLinearObjContainer();
         Teuchos::RCP<LinearObjContainer> dc_loc = t_lof->buildGhostedLinearObjContainer();

         t_lof->initializeGhostedContainer(LinearObjContainer::X | LinearObjContainer::F, *loc);
         loc->initialize();

         t_lof->initializeGhostedContainer(LinearObjContainer::F, *dc_loc);
         dc_loc->initialize();
         Teuchos::RCP<TpetraLinObjContainerType> t_dc_loc = Teuchos::rcp_dynamic_cast<TpetraLinObjContainerType>(dc_loc);

         Teuchos::RCP<TpetraLinObjContainerType> t_loc = Teuchos::rcp_dynamic_cast<TpetraLinObjContainerType>(loc);

         Teuchos::RCP<Thyra::VectorBase<double>> x_vec = t_loc->get_x_th();
         Thyra::assign(x_vec.ptr(), 123.0 + myRank);

         // setup field manager, add evaluator under test
         /////////////////////////////////////////////////////////////

         PHX::FieldManager<panzer::Traits> fm;

         std::string resName = "";
         Teuchos::RCP<std::map<std::string, std::string>> names_map =
             Teuchos::rcp(new std::map<std::string, std::string>);
         names_map->insert(std::make_pair(fieldName1_q1, resName + fieldName1_q1));
         names_map->insert(std::make_pair(fieldName2_q1, resName + fieldName2_q1));
         names_map->insert(std::make_pair(fieldName_qedge1, resName + fieldName_qedge1));

         // evaluators under test
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(resName + fieldName1_q1);
            names->push_back(resName + fieldName2_q1);

            Teuchos::ParameterList pl;
            pl.set("Scatter Name", "ScatterQ1");
            if (scatter_IC) {
               pl.set("Basis", basis_q1.getConst());
            } else {
               pl.set("Basis", basis_q1);
            }
            pl.set("Dependent Names", names);
            pl.set("Dependent Map", names_map);
            pl.set("Side Subcell Dimension", 1);
            pl.set("Local Side ID", 2);
            pl.set("Check Apply BC", false);
            pl.set("Scatter Initial Condition", scatter_IC);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildScatterDirichlet<panzer::Traits::Residual>(pl);

            TEST_EQUALITY(evaluator->evaluatedFields().size(), 1);

            fm.registerEvaluator<panzer::Traits::Residual>(evaluator);
            fm.requireField<panzer::Traits::Residual>(*evaluator->evaluatedFields()[0]);
         }
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(resName + fieldName_qedge1);

            Teuchos::ParameterList pl;
            pl.set("Scatter Name", "ScatterQEdge1");
            if (scatter_IC) {
               pl.set("Basis", basis_qedge1.getConst());
            } else {
               pl.set("Basis", basis_qedge1);
            }
            pl.set("Dependent Names", names);
            pl.set("Dependent Map", names_map);
            pl.set("Side Subcell Dimension", 1);
            pl.set("Local Side ID", 2);
            pl.set("Check Apply BC", false);
            pl.set("Scatter Initial Condition", scatter_IC);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildScatterDirichlet<panzer::Traits::Residual>(pl);

            TEST_EQUALITY(evaluator->evaluatedFields().size(), 1);

            fm.registerEvaluator<panzer::Traits::Residual>(evaluator);
            fm.requireField<panzer::Traits::Residual>(*evaluator->evaluatedFields()[0]);
         }

         // support evaluators
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(fieldName1_q1);
            names->push_back(fieldName2_q1);

            Teuchos::ParameterList pl;
            pl.set("Basis", basis_q1);
            pl.set("DOF Names", names);
            pl.set("Indexer Names", names);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildGather<panzer::Traits::Residual>(pl);

            fm.registerEvaluator<panzer::Traits::Residual>(evaluator);
         }
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(fieldName_qedge1);

            Teuchos::ParameterList pl;
            pl.set("Basis", basis_qedge1);
            pl.set("DOF Names", names);
            pl.set("Indexer Names", names);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildGather<panzer::Traits::Residual>(pl);

            fm.registerEvaluator<panzer::Traits::Residual>(evaluator);
         }

         panzer::Traits::SD sd;
         sd.worksets_ = work_sets;

         fm.postRegistrationSetup(sd);

         panzer::Traits::PED ped;
         ped.gedc->addDataObject("Dirichlet Counter", dc_loc);
         ped.gedc->addDataObject("Solution Gather Container", loc);
         ped.gedc->addDataObject("Residual Scatter Container", loc);
         fm.preEvaluate<panzer::Traits::Residual>(ped);

         // run tests
         /////////////////////////////////////////////////////////////

         panzer::Workset &workset = (*work_sets)[0];
         workset.alpha = 0.0;
         workset.beta = 2.0; // derivatives multiplied by 2
         workset.time = 0.0;
         workset.evaluate_transient_terms = false;

         fm.evaluateFields<panzer::Traits::Residual>(workset);
         fm.postEvaluate<panzer::Traits::Residual>(0);

         // test Residual fields
         panzer::index_t dc_count(0);
         Teuchos::ArrayRCP<const double> data, dc_data;
         Teuchos::RCP<const Thyra::VectorBase<double>> f_vec = t_loc->get_f_th();
         Teuchos::RCP<const Thyra::VectorBase<double>> dc_vec = t_dc_loc->get_f_th();

         // check all the residual values and the count

         Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(dc_vec)->getLocalData(Teuchos::ptrFromRef(dc_data));
         if (scatter_IC) {
            Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(x_vec)->getLocalData(Teuchos::ptrFromRef(data));

            TEST_EQUALITY(static_cast<size_t>(data.size()), t_lof->getGhostedMap()->getLocalNumElements());
            TEST_EQUALITY(data.size(), dc_data.size());
            dc_count = 0;
            for (int i = 0; i < data.size(); i++)
            {
               double target = 123.0 + myRank;
               if (dc_data[i] == 0.0)
               {
                  TEST_EQUALITY(data[i], 0.0);
               }
               else
               {
                  TEST_EQUALITY(data[i], target);
                  dc_count++;
               }
            }
            // Filled everywhere
            TEST_EQUALITY(dc_count, data.size()); 
         } else {
            Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(f_vec)->getLocalData(Teuchos::ptrFromRef(data));
            TEST_EQUALITY(static_cast<size_t>(data.size()), t_lof->getGhostedMap()->getLocalNumElements());
            TEST_EQUALITY(data.size(), dc_data.size());
            dc_count = 0;
            for (int i = 0; i < data.size(); i++)
            {
               double target = 123.0 + myRank;
               if (dc_data[i] == 0.0)
               {
                  TEST_EQUALITY(data[i], 0.0);
               }
               else
               {
                  TEST_EQUALITY(data[i], target);
                  dc_count++;
               }
            }
            // there are 2 nodes or 1 edge on the side and the sides are not shared.
            // 2 nodal functions, 1 edge function 
            TEST_EQUALITY(dc_count, 5 * workset.num_cells); 
         }
      }
   }

   TEUCHOS_UNIT_TEST(assembly, scatter_dirichlet_tangent)
   {

#ifdef HAVE_MPI
      Teuchos::RCP<Teuchos::MpiComm<int>> tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
#else
      NOPE_PANZER_DOESNT_SUPPORT_SERIAL
#endif

      int myRank = tComm->getRank();

      const std::size_t workset_size = 4;
      const std::string fieldName1_q1 = "U";
      const std::string fieldName2_q1 = "V";
      const std::string fieldName_qedge1 = "B";
      const std::size_t numParams = 3;

      Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh(2, 2);

      // build input physics block
      Teuchos::RCP<panzer::PureBasis> basis_q1 = buildBasis(workset_size, "Q1");
      Teuchos::RCP<panzer::PureBasis> basis_qedge1 = buildBasis(workset_size, "QEdge1");

      Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList();
      testInitialization(ipb);

      const int default_int_order = 1;
      std::string eBlockID = "eblock-0_0";
      Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
      panzer::CellData cellData(workset_size, mesh->getCellTopology("eblock-0_0"));
      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
      Teuchos::RCP<panzer::PhysicsBlock> physicsBlock =
          Teuchos::rcp(new PhysicsBlock(ipb, eBlockID, default_int_order, cellData, eqset_factory, gd, false));

      Teuchos::RCP<std::vector<panzer::Workset>> work_sets = panzer_stk::buildWorksets(*mesh, physicsBlock->elementBlockID(),
                                                                                       physicsBlock->getWorksetNeeds());
      TEST_EQUALITY(work_sets->size(), 1);

      std::vector<bool> scatter_IC_vec = {false,true};

      for (const bool scatter_IC : scatter_IC_vec) {
         // build connection manager and field manager
         const Teuchos::RCP<panzer::ConnManager> conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
         RCP<panzer::DOFManager> dofManager = Teuchos::rcp(new panzer::DOFManager(conn_manager, MPI_COMM_WORLD));

         dofManager->addField(fieldName1_q1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_q1->getIntrepid2Basis())));
         dofManager->addField(fieldName2_q1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_q1->getIntrepid2Basis())));
         dofManager->addField(fieldName_qedge1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_qedge1->getIntrepid2Basis())));

         std::vector<std::string> fieldOrder;
         fieldOrder.push_back(fieldName1_q1);
         fieldOrder.push_back(fieldName_qedge1);
         fieldOrder.push_back(fieldName2_q1);
         dofManager->setFieldOrder(fieldOrder);

         dofManager->buildGlobalUnknowns();

         // setup linear object factory
         /////////////////////////////////////////////////////////////
         Teuchos::RCP<TpetraLinObjFactoryType> t_lof = Teuchos::rcp(new TpetraLinObjFactoryType(tComm.getConst(), dofManager));
         Teuchos::RCP<LinearObjFactory<panzer::Traits>> lof = t_lof;
         Teuchos::RCP<LinearObjContainer> loc = t_lof->buildGhostedLinearObjContainer();
         Teuchos::RCP<LinearObjContainer> dc_loc = t_lof->buildGhostedLinearObjContainer();

         t_lof->initializeGhostedContainer(LinearObjContainer::X | LinearObjContainer::F, *loc);
         loc->initialize();

         t_lof->initializeGhostedContainer(LinearObjContainer::F, *dc_loc);
         dc_loc->initialize();
         Teuchos::RCP<TpetraLinObjContainerType> t_dc_loc = Teuchos::rcp_dynamic_cast<TpetraLinObjContainerType>(dc_loc);

         Teuchos::RCP<TpetraLinObjContainerType> t_loc = Teuchos::rcp_dynamic_cast<TpetraLinObjContainerType>(loc);

         Teuchos::RCP<Thyra::VectorBase<double>> x_vec = t_loc->get_x_th();
         Thyra::assign(x_vec.ptr(), 123.0 + myRank);

         std::vector<Teuchos::RCP<GlobalEvaluationData>> tangentContainers;

         using LOCPair = panzer::LOCPair_GlobalEvaluationData;
         using Teuchos::rcp_dynamic_cast;

         // generate tangent data
         for (std::size_t i=0;i<numParams; ++i){
            auto locPair = Teuchos::rcp(new LOCPair(t_lof, panzer::LinearObjContainer::X));

            auto global_t_loc = rcp_dynamic_cast<TpetraLinObjContainerType>(locPair->getGlobalLOC());
            Teuchos::RCP<Thyra::VectorBase<double>> global_x_vec = global_t_loc->get_x_th();
            Thyra::assign(global_x_vec.ptr(), 0.123 + myRank + i);

            auto ghosted_t_loc = rcp_dynamic_cast<TpetraLinObjContainerType>(locPair->getGhostedLOC());
            Teuchos::RCP<Thyra::VectorBase<double>> ghosted_x_vec = ghosted_t_loc->get_x_th();
            Thyra::assign(ghosted_x_vec.ptr(), 0.123 + myRank + i);

            tangentContainers.push_back(locPair);
         }

         // setup field manager, add evaluator under test
         /////////////////////////////////////////////////////////////

         auto fm = Teuchos::rcp(new PHX::FieldManager<panzer::Traits>);

         std::string resName = "";
         Teuchos::RCP<std::map<std::string, std::string>> names_map =
             Teuchos::rcp(new std::map<std::string, std::string>);
         names_map->insert(std::make_pair(fieldName1_q1, resName + fieldName1_q1));
         names_map->insert(std::make_pair(fieldName2_q1, resName + fieldName2_q1));
         names_map->insert(std::make_pair(fieldName_qedge1, resName + fieldName_qedge1));

         std::vector<PHX::index_size_type> derivative_dimensions;
         derivative_dimensions.push_back(numParams);
         fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Tangent>(derivative_dimensions);

         // evaluators under test
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(resName + fieldName1_q1);
            names->push_back(resName + fieldName2_q1);

            Teuchos::ParameterList pl;
            pl.set("Scatter Name", "ScatterQ1");
            if (scatter_IC) {
               pl.set("Basis", basis_q1.getConst());
            } else {
               pl.set("Basis", basis_q1);
            }
            pl.set("Dependent Names", names);
            pl.set("Dependent Map", names_map);
            pl.set("Side Subcell Dimension", 1);
            pl.set("Local Side ID", 2);
            pl.set("Check Apply BC", false);
            pl.set("Scatter Initial Condition", scatter_IC);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildScatterDirichlet<panzer::Traits::Tangent>(pl);

            TEST_EQUALITY(evaluator->evaluatedFields().size(), 1);

            fm->registerEvaluator<panzer::Traits::Tangent>(evaluator);
            fm->requireField<panzer::Traits::Tangent>(*evaluator->evaluatedFields()[0]);
         }
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(resName + fieldName_qedge1);

            Teuchos::ParameterList pl;
            pl.set("Scatter Name", "ScatterQEdge1");
            if (scatter_IC) {
               pl.set("Basis", basis_qedge1.getConst());
            } else {
               pl.set("Basis", basis_qedge1);
            }
            pl.set("Dependent Names", names);
            pl.set("Dependent Map", names_map);
            pl.set("Side Subcell Dimension", 1);
            pl.set("Local Side ID", 2);
            pl.set("Check Apply BC", false);
            pl.set("Scatter Initial Condition", scatter_IC);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildScatterDirichlet<panzer::Traits::Tangent>(pl);

            TEST_EQUALITY(evaluator->evaluatedFields().size(), 1);

            fm->registerEvaluator<panzer::Traits::Tangent>(evaluator);
            fm->requireField<panzer::Traits::Tangent>(*evaluator->evaluatedFields()[0]);
         }

         // support evaluators
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(fieldName1_q1);
            names->push_back(fieldName2_q1);

            Teuchos::ParameterList pl;
            pl.set("Basis", basis_q1);
            pl.set("DOF Names", names);
            pl.set("Indexer Names", names);
            Teuchos::RCP<std::vector<std::vector<std::string>>> tangent_names = 
               Teuchos::rcp(new std::vector<std::vector<std::string>>(2));
            for (std::size_t i = 0; i < numParams; ++i)
            {
              std::stringstream ss1, ss2;
              ss1 << fieldName1_q1 << " Tangent " << i;
              ss2 << fieldName2_q1 << " Tangent " << i;
              (*tangent_names)[0].push_back(ss1.str());
              (*tangent_names)[1].push_back(ss2.str());
            }
            pl.set("Tangent Names", tangent_names);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildGather<panzer::Traits::Tangent>(pl);

            fm->registerEvaluator<panzer::Traits::Tangent>(evaluator);
         }
         for (std::size_t i = 0; i < numParams; ++i) {
           using Teuchos::RCP;
           using Teuchos::rcp;
           RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
           RCP<std::vector<std::string>> tangent_names = rcp(new std::vector<std::string>);
           names->push_back(fieldName1_q1);
           names->push_back(fieldName2_q1);
          {
             std::stringstream ss1, ss2;
             ss1 << fieldName1_q1 << " Tangent " << i;
             ss2 << fieldName2_q1 << " Tangent " << i;
             tangent_names->push_back(ss1.str());
             tangent_names->push_back(ss2.str());
           }

           Teuchos::ParameterList pl;
           pl.set("Basis", basis_q1);
           pl.set("DOF Names", tangent_names);
           pl.set("Indexer Names", names);

           std::stringstream ss;
           ss << "Tangent Container " << i;
           pl.set("Global Data Key", ss.str());

           Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator =
               lof->buildGatherTangent<panzer::Traits::Tangent>(pl);

           fm->registerEvaluator<panzer::Traits::Tangent>(evaluator);
         }
         {
            using Teuchos::RCP;
            using Teuchos::rcp;
            RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
            names->push_back(fieldName_qedge1);

            Teuchos::ParameterList pl;
            pl.set("Basis", basis_qedge1);
            pl.set("DOF Names", names);
            pl.set("Indexer Names", names);
            Teuchos::RCP<std::vector<std::vector<std::string>>> tangent_names = 
               Teuchos::rcp(new std::vector<std::vector<std::string>>(1));
            for (std::size_t i = 0; i < numParams; ++i)
            {
              std::stringstream ss;
              ss << fieldName_qedge1 << " Tangent " << i;
              (*tangent_names)[0].push_back(ss.str());
            }
            pl.set("Tangent Names", tangent_names);

            Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildGather<panzer::Traits::Tangent>(pl);

            fm->registerEvaluator<panzer::Traits::Tangent>(evaluator);
         }
         for (std::size_t i = 0; i < numParams; ++i) {
           using Teuchos::RCP;
           using Teuchos::rcp;
           RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
           RCP<std::vector<std::string>> tangent_names = rcp(new std::vector<std::string>);
           names->push_back(fieldName_qedge1);
          {
             std::stringstream ss;
             ss << fieldName_qedge1 << " Tangent " << i;
             tangent_names->push_back(ss.str());
           }

           Teuchos::ParameterList pl;
           pl.set("Basis", basis_qedge1);
           pl.set("DOF Names", tangent_names);
           pl.set("Indexer Names", names);

           std::stringstream ss;
           ss << "Tangent Container " << i;
           pl.set("Global Data Key", ss.str());

           Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator =
               lof->buildGatherTangent<panzer::Traits::Tangent>(pl);

           fm->registerEvaluator<panzer::Traits::Tangent>(evaluator);
         }

         panzer::Traits::SD sd;
         sd.worksets_ = work_sets;

         fm->postRegistrationSetup(sd);

         panzer::Traits::PED ped;
         ped.gedc->addDataObject("Dirichlet Counter", dc_loc);
         ped.gedc->addDataObject("Solution Gather Container", loc);
         ped.gedc->addDataObject("Residual Scatter Container", loc);
         for (size_t i=0; i<numParams; ++i) {
            std::stringstream ss;
            ss << "Tangent Container " << i;
            ped.gedc->addDataObject(ss.str(), tangentContainers[i]);
         }
         std::vector<std::string> params;
         std::vector<Teuchos::RCP<LOCPair>> paramContainers;
         for (std::size_t i = 0; i<numParams; ++i) {
            std::stringstream ss;
            ss << "param" << i;
            params.push_back(ss.str());
            auto paramContainer = Teuchos::rcp(new LOCPair(t_lof, panzer::LinearObjContainer::X | panzer::LinearObjContainer::F));
            ped.gedc->addDataObject(ss.str(),paramContainer->getGhostedLOC());
            paramContainers.push_back(paramContainer);
         }
         Teuchos::RCP<panzer::GlobalEvaluationData> activeParams =
            Teuchos::rcp(new panzer::ParameterList_GlobalEvaluationData(params));
         ped.gedc->addDataObject("PARAMETER_NAMES",activeParams);
         fm->preEvaluate<panzer::Traits::Tangent>(ped);

         // run tests
         /////////////////////////////////////////////////////////////

         panzer::Workset &workset = (*work_sets)[0];
         workset.alpha = 0.0;
         workset.beta = 2.0; // derivatives multiplied by 2
         workset.time = 0.0;
         workset.evaluate_transient_terms = false;

         fm->evaluateFields<panzer::Traits::Tangent>(workset);
         fm->postEvaluate<panzer::Traits::Tangent>(0);

         fm = Teuchos::null;

         // test Tangent fields
         panzer::index_t dc_count(0);
         Teuchos::ArrayRCP<const double> data, dc_data;
         Teuchos::RCP<const Thyra::VectorBase<double>> f_vec = t_loc->get_f_th();
         Teuchos::RCP<const Thyra::VectorBase<double>> dc_vec = t_dc_loc->get_f_th();

         // check all the residual values and the count

         Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(dc_vec)->getLocalData(Teuchos::ptrFromRef(dc_data));
         if (scatter_IC) {
            Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(x_vec)->getLocalData(Teuchos::ptrFromRef(data));

            TEST_EQUALITY(static_cast<size_t>(data.size()), t_lof->getGhostedMap()->getLocalNumElements());
            TEST_EQUALITY(data.size(), dc_data.size());
            dc_count = 0;
            for (int i = 0; i < data.size(); i++)
            {
               double target = 123.0 + myRank;
               if (dc_data[i] == 0.0)
               {
                  TEST_EQUALITY(data[i], 0.0);
               }
               else
               {
                  TEST_EQUALITY(data[i], target);
                  dc_count++;
               }
            }
            // Filled everywhere
            TEST_EQUALITY(dc_count, data.size()); 
         } else {
            Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(f_vec)->getLocalData(Teuchos::ptrFromRef(data));
            TEST_EQUALITY(static_cast<size_t>(data.size()), t_lof->getGhostedMap()->getLocalNumElements());
            TEST_EQUALITY(data.size(), dc_data.size());
            dc_count = 0;
            for (int i = 0; i < data.size(); i++)
            {
               double target = 123.0 + myRank;
               if (dc_data[i] == 0.0)
               {
                  TEST_EQUALITY(data[i], 0.0);
               }
               else
               {
                  TEST_EQUALITY(data[i], target);
                  dc_count++;
               }
            }
            // there are 2 nodes or 1 edge on the side and the sides are not shared.
            // 2 nodal functions, 1 edge function 
            TEST_EQUALITY(dc_count, 5 * workset.num_cells); 
         }
         for (std::size_t i=0; i<numParams; ++i)
         {
            // now check the tangent values
            Teuchos::ArrayRCP<const double> tan_data;
            Teuchos::RCP<const Thyra::VectorBase<double>> tan_vec = Teuchos::rcp_dynamic_cast<TpetraLinObjContainerType>(paramContainers[i]->getGhostedLOC())->get_f_th();
            Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(tan_vec)->getLocalData(Teuchos::ptrFromRef(tan_data));

            for (int j = 0; j < tan_data.size(); ++j)
            {
               if (dc_data[j] == 0.) {
                  TEST_EQUALITY(data[j],0.0);
               } else {
                  const double target = .123 + myRank + i;
                  TEST_EQUALITY(tan_data[j],target);
               }
            }
         }
      }
   }
//   TEUCHOS_UNIT_TEST(assembly, scatter_dirichlet_jacobian)
//   {
//
//#ifdef HAVE_MPI
//      Teuchos::RCP<Teuchos::MpiComm<int>> tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
//#else
//      NOPE_PANZER_DOESNT_SUPPORT_SERIAL
//#endif
//
//      int myRank = tComm->getRank();
//
//      const std::size_t workset_size = 4;
//      const std::string fieldName1_q1 = "U";
//      const std::string fieldName2_q1 = "V";
//      const std::string fieldName_qedge1 = "B";
//
//      Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh(2, 2);
//
//      // build input physics block
//      Teuchos::RCP<panzer::PureBasis> basis_q1 = buildBasis(workset_size, "Q1");
//      Teuchos::RCP<panzer::PureBasis> basis_qedge1 = buildBasis(workset_size, "QEdge1");
//
//      Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList();
//      testInitialization(ipb);
//
//      const int default_int_order = 1;
//      std::string eBlockID = "eblock-0_0";
//      Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
//      panzer::CellData cellData(workset_size, mesh->getCellTopology("eblock-0_0"));
//      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
//      Teuchos::RCP<panzer::PhysicsBlock> physicsBlock =
//          Teuchos::rcp(new PhysicsBlock(ipb, eBlockID, default_int_order, cellData, eqset_factory, gd, false));
//
//      Teuchos::RCP<std::vector<panzer::Workset>> work_sets = panzer_stk::buildWorksets(*mesh, physicsBlock->elementBlockID(),
//                                                                                       physicsBlock->getWorksetNeeds());
//      TEST_EQUALITY(work_sets->size(), 1);
//
//      // build connection manager and field manager
//      const Teuchos::RCP<panzer::ConnManager> conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
//      RCP<panzer::BlockedDOFManager> dofManager = Teuchos::rcp(new panzer::BlockedDOFManager(conn_manager, MPI_COMM_WORLD));
//
//      dofManager->addField(fieldName1_q1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_q1->getIntrepid2Basis())));
//      dofManager->addField(fieldName2_q1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_q1->getIntrepid2Basis())));
//      dofManager->addField(fieldName_qedge1, Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_qedge1->getIntrepid2Basis())));
//
//      std::vector<std::vector<std::string>> fieldOrder(3);
//      fieldOrder[0].push_back(fieldName1_q1);
//      fieldOrder[1].push_back(fieldName_qedge1);
//      fieldOrder[2].push_back(fieldName2_q1);
//      dofManager->setFieldOrder(fieldOrder);
//
//      // dofManager->setOrientationsRequired(true);
//      dofManager->buildGlobalUnknowns();
//
//      // setup linear object factory
//      /////////////////////////////////////////////////////////////
//
//      Teuchos::RCP<TpetraBlockedLinObjFactoryType> bt_lof = Teuchos::rcp(new TpetraBlockedLinObjFactoryType(tComm.getConst(), dofManager));
//      Teuchos::RCP<LinearObjFactory<panzer::Traits>> lof = bt_lof;
//      Teuchos::RCP<LinearObjContainer> dd_loc = bt_lof->buildGhostedLinearObjContainer();
//      Teuchos::RCP<LinearObjContainer> loc = bt_lof->buildGhostedLinearObjContainer();
//      bt_lof->initializeGhostedContainer(LinearObjContainer::F, *dd_loc);
//      dd_loc->initialize();
//
//      bt_lof->initializeGhostedContainer(LinearObjContainer::X | LinearObjContainer::F | LinearObjContainer::Mat, *loc);
//      loc->initialize();
//
//      Teuchos::RCP<TpetraBlockedLinObjContainerType> b_dd_loc = Teuchos::rcp_dynamic_cast<TpetraBlockedLinObjContainerType>(dd_loc);
//      Teuchos::RCP<TpetraBlockedLinObjContainerType> b_loc = Teuchos::rcp_dynamic_cast<TpetraBlockedLinObjContainerType>(loc);
//      Teuchos::RCP<Thyra::ProductVectorBase<double>> p_vec = Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<double>>(b_loc->get_x());
//      Thyra::assign(p_vec->getNonconstVectorBlock(0).ptr(), 123.0 + myRank);
//      Thyra::assign(p_vec->getNonconstVectorBlock(1).ptr(), 456.0 + myRank);
//      Thyra::assign(p_vec->getNonconstVectorBlock(2).ptr(), 789.0 + myRank);
//
//      auto blk_A = Teuchos::rcp_dynamic_cast<Thyra::BlockedLinearOpBase<double>>(b_loc->get_A());
//      double values[] = {123.0 + myRank, 456.0 + myRank, 789.0 + myRank};
//
//      for (int i = 0; i < 3; i++)
//         for (int j = 0; j < 3; j++)
//         {
//            auto thyraOp = Teuchos::rcp_dynamic_cast<Thyra_TpetraLinearOp>(blk_A->getNonconstBlock(i, j), false);
//            auto tpetraCrsMatrix = Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(thyraOp->getTpetraOperator(), true);
//            tpetraCrsMatrix->setAllToScalar(values[i] * values[j]);
//         }
//
//      // setup field manager, add evaluator under test
//      /////////////////////////////////////////////////////////////
//
//      PHX::FieldManager<panzer::Traits> fm;
//
//      std::string resName = "";
//      Teuchos::RCP<std::map<std::string, std::string>> names_map =
//          Teuchos::rcp(new std::map<std::string, std::string>);
//      names_map->insert(std::make_pair(fieldName1_q1, resName + fieldName1_q1));
//      names_map->insert(std::make_pair(fieldName2_q1, resName + fieldName2_q1));
//      names_map->insert(std::make_pair(fieldName_qedge1, resName + fieldName_qedge1));
//
//      // evaluators under test
//      {
//         using Teuchos::RCP;
//         using Teuchos::rcp;
//         RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
//         names->push_back(resName + fieldName1_q1);
//         names->push_back(resName + fieldName2_q1);
//
//         Teuchos::ParameterList pl;
//         pl.set("Scatter Name", "ScatterQ1");
//         pl.set("Basis", basis_q1);
//         pl.set("Dependent Names", names);
//         pl.set("Dependent Map", names_map);
//         pl.set("Side Subcell Dimension", 1);
//         pl.set("Local Side ID", 2);
//         pl.set("Check Apply BC", false);
//
//         Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildScatterDirichlet<panzer::Traits::Jacobian>(pl);
//
//         TEST_EQUALITY(evaluator->evaluatedFields().size(), 1);
//
//         fm.registerEvaluator<panzer::Traits::Jacobian>(evaluator);
//         fm.requireField<panzer::Traits::Jacobian>(*evaluator->evaluatedFields()[0]);
//      }
//      {
//         using Teuchos::RCP;
//         using Teuchos::rcp;
//         RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
//         names->push_back(resName + fieldName_qedge1);
//
//         Teuchos::ParameterList pl;
//         pl.set("Scatter Name", "ScatterQEdge1");
//         pl.set("Basis", basis_qedge1);
//         pl.set("Dependent Names", names);
//         pl.set("Dependent Map", names_map);
//         pl.set("Side Subcell Dimension", 1);
//         pl.set("Local Side ID", 2);
//         pl.set("Check Apply BC", false);
//
//         Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildScatterDirichlet<panzer::Traits::Jacobian>(pl);
//
//         TEST_EQUALITY(evaluator->evaluatedFields().size(), 1);
//
//         fm.registerEvaluator<panzer::Traits::Jacobian>(evaluator);
//         fm.requireField<panzer::Traits::Jacobian>(*evaluator->evaluatedFields()[0]);
//      }
//
//      // support evaluators
//      {
//         using Teuchos::RCP;
//         using Teuchos::rcp;
//         RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
//         names->push_back(fieldName1_q1);
//         names->push_back(fieldName2_q1);
//
//         Teuchos::ParameterList pl;
//         pl.set("Basis", basis_q1);
//         pl.set("DOF Names", names);
//         pl.set("Indexer Names", names);
//
//         Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildGather<panzer::Traits::Jacobian>(pl);
//
//         fm.registerEvaluator<panzer::Traits::Jacobian>(evaluator);
//      }
//      {
//         using Teuchos::RCP;
//         using Teuchos::rcp;
//         RCP<std::vector<std::string>> names = rcp(new std::vector<std::string>);
//         names->push_back(fieldName_qedge1);
//
//         Teuchos::ParameterList pl;
//         pl.set("Basis", basis_qedge1);
//         pl.set("DOF Names", names);
//         pl.set("Indexer Names", names);
//
//         Teuchos::RCP<PHX::Evaluator<panzer::Traits>> evaluator = lof->buildGather<panzer::Traits::Jacobian>(pl);
//
//         fm.registerEvaluator<panzer::Traits::Jacobian>(evaluator);
//      }
//
//      std::vector<PHX::index_size_type> derivative_dimensions;
//      derivative_dimensions.push_back(12);
//      fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
//
//      panzer::Traits::SD sd;
//      sd.worksets_ = work_sets;
//
//      fm.postRegistrationSetup(sd);
//
//      // panzer::Traits::PED ped;
//      // ped.dirichletData.ghostedCounter = dd_loc;
//      // fm.preEvaluate<panzer::Traits::Jacobian>(ped);
//      panzer::Traits::PED ped;
//      ped.gedc->addDataObject("Dirichlet Counter", dd_loc);
//      ped.gedc->addDataObject("Solution Gather Container", loc);
//      ped.gedc->addDataObject("Residual Scatter Container", loc);
//      fm.preEvaluate<panzer::Traits::Jacobian>(ped);
//
//      // run tests
//      /////////////////////////////////////////////////////////////
//
//      panzer::Workset &workset = (*work_sets)[0];
//      workset.alpha = 0.0;
//      workset.beta = 2.0; // derivatives multiplied by 2
//      workset.time = 0.0;
//      workset.evaluate_transient_terms = false;
//
//      fm.evaluateFields<panzer::Traits::Jacobian>(workset);
//
//      // test Residual fields
//      panzer::index_t dd_count(0);
//      Teuchos::ArrayRCP<const double> data, dd_data;
//      Teuchos::RCP<const Thyra::ProductVectorBase<double>> f_vec = Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<double>>(b_loc->get_f());
//      Teuchos::RCP<const Thyra::ProductVectorBase<double>> dd_vec = Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<double>>(b_dd_loc->get_f());
//
//      // check all the residual values. This is kind of crappy test since it simply checks twice the target
//      // value and the target. Its this way because you add two entries across elements.
//
//      Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(f_vec->getVectorBlock(0))->getLocalData(Teuchos::ptrFromRef(data));
//      Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(dd_vec->getVectorBlock(0))->getLocalData(Teuchos::ptrFromRef(dd_data));
//      TEST_EQUALITY(static_cast<size_t>(data.size()), b_loc->getMapForBlock(0)->getLocalNumElements());
//      TEST_EQUALITY(data.size(), dd_data.size());
//      dd_count = 0;
//      for (int i = 0; i < data.size(); i++)
//      {
//
//         double target = 123.0 + myRank;
//         if (dd_data[i] == 0.0)
//         {
//            TEST_EQUALITY(data[i], 0.0);
//         }
//         else
//         {
//            TEST_EQUALITY(data[i], target);
//            dd_count++;
//         }
//      }
//      TEST_EQUALITY(dd_count, 2 * workset.num_cells); // there are 2 nodes on the side and the sides are not shared
//
//      Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(f_vec->getVectorBlock(1))->getLocalData(Teuchos::ptrFromRef(data));
//      Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(dd_vec->getVectorBlock(1))->getLocalData(Teuchos::ptrFromRef(dd_data));
//      TEST_EQUALITY(static_cast<size_t>(data.size()), b_loc->getMapForBlock(1)->getLocalNumElements());
//      TEST_EQUALITY(data.size(), dd_data.size());
//      dd_count = 0;
//      for (int i = 0; i < data.size(); i++)
//      {
//
//         double target = 456.0 + myRank;
//         if (dd_data[i] == 0.0)
//         {
//            TEST_EQUALITY(data[i], 0.0);
//         }
//         else
//         {
//            TEST_EQUALITY(data[i], target);
//            dd_count++;
//         }
//      }
//      TEST_EQUALITY(dd_count, workset.num_cells); // there are 2 nodes on the side and the sides are not shared
//
//      Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(f_vec->getVectorBlock(2))->getLocalData(Teuchos::ptrFromRef(data));
//      Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(dd_vec->getVectorBlock(2))->getLocalData(Teuchos::ptrFromRef(dd_data));
//      TEST_EQUALITY(static_cast<size_t>(data.size()), b_loc->getMapForBlock(2)->getLocalNumElements());
//      TEST_EQUALITY(data.size(), dd_data.size());
//      dd_count = 0;
//      for (int i = 0; i < data.size(); i++)
//      {
//
//         double target = 789.0 + myRank;
//         if (dd_data[i] == 0.0)
//         {
//            TEST_EQUALITY(data[i], 0.0);
//         }
//         else
//         {
//            TEST_EQUALITY(data[i], target);
//            dd_count++;
//         }
//      }
//      TEST_EQUALITY(dd_count, 2 * workset.num_cells); // there are 2 nodes on the side and the sides are not shared
//   }

   Teuchos::RCP<panzer::PureBasis> buildBasis(std::size_t worksetSize, const std::string &basisName)
   {
      Teuchos::RCP<shards::CellTopology> topo =
          Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4>>()));

      panzer::CellData cellData(worksetSize, topo);
      return Teuchos::rcp(new panzer::PureBasis(basisName, 1, cellData));
   }

   Teuchos::RCP<panzer_stk::STK_Interface> buildMesh(int elemX, int elemY)
   {
      RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
      pl->set("X Blocks", 1);
      pl->set("Y Blocks", 1);
      pl->set("X Elements", elemX);
      pl->set("Y Elements", elemY);

      panzer_stk::SquareQuadMeshFactory factory;
      factory.setParameterList(pl);
      RCP<panzer_stk::STK_Interface> mesh = factory.buildUncommitedMesh(MPI_COMM_WORLD);
      factory.completeMeshConstruction(*mesh, MPI_COMM_WORLD);

      return mesh;
   }

   void testInitialization(const Teuchos::RCP<Teuchos::ParameterList> &ipb)
   {
      // Physics block
      ipb->setName("test physics");
      {
         Teuchos::ParameterList &p = ipb->sublist("a");
         p.set("Type", "Energy");
         p.set("Prefix", "");
         p.set("Model ID", "solid");
         p.set("Basis Type", "HGrad");
         p.set("Basis Order", 1);
         p.set("Integration Order", 1);
      }
      {
         Teuchos::ParameterList &p = ipb->sublist("b");
         p.set("Type", "Energy");
         p.set("Prefix", "ION_");
         p.set("Model ID", "solid");
         p.set("Basis Type", "HCurl");
         p.set("Basis Order", 1);
         p.set("Integration Order", 1);
      }
   }

}
