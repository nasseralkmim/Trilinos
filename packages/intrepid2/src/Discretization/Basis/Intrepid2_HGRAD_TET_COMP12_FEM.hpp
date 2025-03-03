// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_HGRAD_TET_COMP12_FEM.hpp
    \brief  Header file for the Intrepid2::Basis_HGRAD_TET_COMP12_FEM class.
    \author Created by P. Bochev, J. Ostien, K. Peterson and D. Ridzal.
            Kokkorized by Kyungjoo Kim
*/

#ifndef __INTREPID2_HGRAD_TET_COMP12_FEM_HPP__
#define __INTREPID2_HGRAD_TET_COMP12_FEM_HPP__

#include "Intrepid2_Basis.hpp"

namespace Intrepid2 {

  /** \class  Intrepid2::Basis_HGRAD_TET_COMP12_FEM
      \brief  Implementation of the default H(grad)-compatible FEM basis of degree 2 on Tetrahedron cell

      Implements Lagrangian basis of degree 2 on the reference Tetrahedron cell. The basis has
      cardinality 10 and spans a COMPLETE quadratic polynomial space. Basis functions are dual
      to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:

      \verbatim
      =================================================================================================
      |         |           degree-of-freedom-tag table                    |                           |
      |   DoF   |----------------------------------------------------------|      DoF definition       |
      | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                           |
      |=========|==============|==============|==============|=============|===========================|
      |    0    |       0      |       0      |       0      |      1      |   L_0(u) = u(0,0,0)       |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    1    |       0      |       1      |       0      |      1      |   L_1(u) = u(1,0,0)       |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    2    |       0      |       2      |       0      |      1      |   L_2(u) = u(0,1,0)       |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    3    |       0      |       3      |       0      |      1      |   L_3(u) = u(0,0,1)       |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    4    |       1      |       0      |       0      |      1      |   L_4(u) = u(1/2,0,0)     |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    5    |       1      |       1      |       0      |      1      |   L_5(u) = u(1/2,1/2,0)   |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    6    |       1      |       2      |       0      |      1      |   L_6(u) = u(0,1/2,0)     |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    7    |       1      |       3      |       0      |      1      |   L_7(u) = u(0,0,1/2)     |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    8    |       1      |       4      |       0      |      1      |   L_8(u) = u(1/2,0,1/2)   |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    9    |       1      |       5      |       0      |      1      |   L_9(u) = u(0,1/2,1/2)   |
      |=========|==============|==============|==============|=============|===========================|
      |   MAX   |  maxScDim=0  |  maxScOrd=3  |  maxDfOrd=0  |     -       |                           |
      |=========|==============|==============|==============|=============|===========================|
      \endverbatim

      \remark   Ordering of DoFs follows the node order in Tetrahedron<10> topology. Note that node order
      in this topology follows the natural order of k-subcells where the nodes are located, i.e.,
      L_0 to L_3 correspond to 0-subcells (vertices) 0 to 3 and L_4 to L_9 correspond to
      1-subcells (edges) 0 to 5.
  */

  namespace Impl {

    /**
      \brief See Intrepid2::Basis_HGRAD_TET_COMP12_FEM
    */
    class Basis_HGRAD_TET_COMP12_FEM {
    public:
      typedef struct Tetrahedron<4> cell_topology_type;

      /**
        \brief See Intrepid2::Basis_HGRAD_TET_COMP12_FEM
      */
      template<typename pointValueType>
      KOKKOS_INLINE_FUNCTION
      static ordinal_type
      getLocalSubTet( const pointValueType x,
                      const pointValueType y,
                      const pointValueType z );

      /**
        \brief See Intrepid2::Basis_HGRAD_TET_COMP12_FEM
      */
      template<EOperator opType>
      struct Serial {
        template<typename outputValueViewType,
                 typename inputPointViewType>
        KOKKOS_INLINE_FUNCTION
        static void
        getValues(       outputValueViewType outputValues,
                   const inputPointViewType  inputPoints );

      };

      template<typename DeviceType,
               typename outputValueValueType, class ...outputValueProperties,
               typename inputPointValueType,  class ...inputPointProperties>
      static void
      getValues(  const typename DeviceType::execution_space& space,
                        Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                  const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                  const EOperator operatorType );

      /**
        \brief See Intrepid2::Basis_HGRAD_TET_COMP12_FEM
      */
      template<typename outputValueViewType,
               typename inputPointViewType,
               EOperator opType>
      struct Functor {
              outputValueViewType _outputValues;
        const inputPointViewType  _inputPoints;

        KOKKOS_INLINE_FUNCTION
        Functor(      outputValueViewType outputValues_,
                      inputPointViewType  inputPoints_ )
          : _outputValues(outputValues_), _inputPoints(inputPoints_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_type pt) const {
          switch (opType) {
          case OPERATOR_VALUE : {
            auto       output = Kokkos::subview( _outputValues, Kokkos::ALL(), pt );
            const auto input  = Kokkos::subview( _inputPoints,                 pt, Kokkos::ALL() );
            Serial<opType>::getValues( output, input );
            break;
          }
          case OPERATOR_GRAD :
          case OPERATOR_MAX : {
            auto       output = Kokkos::subview( _outputValues, Kokkos::ALL(), pt, Kokkos::ALL() );
            const auto input  = Kokkos::subview( _inputPoints,                 pt, Kokkos::ALL() );
            Serial<opType>::getValues( output, input );
            break;
          }
          default: {
            INTREPID2_TEST_FOR_ABORT( opType != OPERATOR_VALUE &&
                                      opType != OPERATOR_GRAD &&
                                      opType != OPERATOR_MAX,
                                      ">>> ERROR: (Intrepid2::Basis_HGRAD_TET_COMP12_FEM::Functor::operator() operator is not supported");
          }
          }
        }
      };
    };
  }

  template<typename DeviceType = void,
           typename outputValueType = double,
           typename pointValueType = double>
  class Basis_HGRAD_TET_COMP12_FEM : public Basis<DeviceType,outputValueType,pointValueType> {
  public:
    using BasisBase = Basis<DeviceType,outputValueType,pointValueType>;
    using typename BasisBase::ExecutionSpace;

    using typename BasisBase::OrdinalTypeArray1DHost;
    using typename BasisBase::OrdinalTypeArray2DHost;
    using typename BasisBase::OrdinalTypeArray3DHost;

    using typename BasisBase::OutputViewType;
    using typename BasisBase::PointViewType ;
    using typename BasisBase::ScalarViewType;

    /** \brief  Constructor.
     */
    Basis_HGRAD_TET_COMP12_FEM();

    using BasisBase::getValues;

    /** \brief  FEM basis evaluation on a <strong>reference Tetrahedron</strong> cell.

        Returns values of <var>operatorType</var> acting on FEM basis functions for a set of
        points in the <strong>reference Tetrahedron</strong> cell. For rank and dimensions of
        I/O array arguments see Section \ref basis_md_array_sec .

        \param  outputValues      [out] - rank-2 or 3 array with the computed basis values
        \param  inputPoints       [in]  - rank-2 array with dimensions (P,D) containing reference points
        \param  operatorType      [in]  - operator applied to basis functions

        For rank and dimension specifications of <var>ArrayScalar</var> arguments see \ref basis_array_specs
    */
    virtual
    void
    getValues( const ExecutionSpace& space,
                     OutputViewType  outputValues,
               const PointViewType   inputPoints,
               const EOperator       operatorType = OPERATOR_VALUE ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      Intrepid2::getValues_HGRAD_Args(outputValues,
                                      inputPoints,
                                      operatorType,
                                      this->getBaseCellTopology(),
                                      this->getCardinality() );
#endif
      Impl::Basis_HGRAD_TET_COMP12_FEM::
        getValues<DeviceType>(space,
                              outputValues,
                              inputPoints,
                              operatorType);
    }

    virtual void 
    getScratchSpaceSize(      ordinal_type& perTeamSpaceSize,
                              ordinal_type& perThreadSpaceSize,
                        const PointViewType inputPointsconst,
                        const EOperator operatorType = OPERATOR_VALUE) const override;

    KOKKOS_INLINE_FUNCTION
    virtual void 
    getValues(       
          OutputViewType outputValues,
      const PointViewType  inputPoints,
      const EOperator operatorType,
      const typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type& team_member,
      const typename DeviceType::execution_space::scratch_memory_space & scratchStorage, 
      const ordinal_type subcellDim = -1,
      const ordinal_type subcellOrdinal = -1) const override;

    virtual
    void
    getDofCoords( ScalarViewType dofCoords ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( rank(dofCoords) != 2, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_TET_COMP12_FEM::getDofCoords) rank = 2 required for dofCoords array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_TET_COMP12_FEM::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_TET_COMP12_FEM::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    virtual
    const char*
    getName() const override {
      return "Intrepid2_HGRAD_TET_COMP12_FEM";
    }

  };
}// namespace Intrepid2

#include "Intrepid2_HGRAD_TET_COMP12_FEMDef.hpp"

#endif
