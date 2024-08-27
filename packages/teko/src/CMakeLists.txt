

#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_Config.h)

#
# B) Define the header and source files (and directories)
#

#
# src
#

SET(HEADERS "")
SET(SOURCES "")

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_Config.h
  )

#
# Core files
#

TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/NS)
TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/Tpetra)

IF(TEKO_HAVE_EPETRA)
   TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/Epetra)
ENDIF()

IF (ML_ENABLE_TekoSmoothers)
   TRIBITS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/mlutils)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/adapters/epetra)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/adapters/stratimikos)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/adapters/tpetra)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Headers)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Interface)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/adapters/xpetra)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/MueCentral)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Utils)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Utils/ForwardDeclaration)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Smoothers)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/UncoupledAggregation)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph)                                           
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/BrickAggregation)                          
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/Containers)                                
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/CoupledAggregation)                        
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/HybridAggregation)                         
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/MatrixTransformation)                      
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/PairwiseAggregation)                       
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/StructuredAggregation)                     
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/StructuredAggregation/coupled)             
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/StructuredAggregation/uncoupled)           
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/UncoupledAggregation)                      
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Graph/UserAggregation)                           
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Headers)                                         
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Interface)                                       
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Interface/FacadeClasses)                         
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Misc)                                            
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/MueCentral)                                      
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Rebalancing)                                     
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Smoothers)                                       
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/BaseClass)                             
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/BlackBox)                              
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Smoothers/BlockedSmoothers)                      
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/BlockedTransfers)                      
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Classical)                             
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Energy-Minimization)                   
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Energy-Minimization/Solvers)           
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/GeneralGeometric)                      
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Generic)                               
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Geometric-Interpolation)               
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Petrov-Galerkin-SA)                    
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/SemiCoarsen)                           
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/Smoothed-Aggregation)                  
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Transfers/User)                                  
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Utils)                                           
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/Utils/ForwardDeclaration)                        
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../muelu/src/../adapters/xpetra)                              
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../build/packages/muelu/src)                              
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../build/trilinos-release-13-4-0-debug/packages/muelu/src)                              

APPEND_GLOB(HEADERS
  *.hpp
  NS/*.hpp
  Tpetra/*.hpp
  )

APPEND_GLOB(SOURCES
  *.cpp
  NS/*.cpp
  Tpetra/*.cpp
  )

IF(NOT TEKO_HAVE_EPETRA)
  LIST(REMOVE_ITEM SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NS/Teko_ALOperator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Teko_DiagonalPreconditionerOp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Teko_ProbingPreconditionerFactory.cpp
  )
  LIST(REMOVE_ITEM HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NS/Teko_ALOperator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Teko_DiagonalPreconditionerOp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Teko_ProbingPreconditionerFactory.hpp
  )
ELSE()
  APPEND_GLOB(HEADERS Epetra/*.hpp)
  APPEND_GLOB(SOURCES Epetra/*.cpp)
ENDIF()

IF (ML_ENABLE_TekoSmoothers)
   APPEND_GLOB(HEADERS mlutils/*.hpp)
   APPEND_GLOB(SOURCES mlutils/*.cpp)
ENDIF()

######################################

#
# C) Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  teko
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )
