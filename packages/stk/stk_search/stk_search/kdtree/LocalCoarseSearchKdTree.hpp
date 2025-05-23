// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of NTESS nor the names of its contributors
//       may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#ifndef KDTREE_STK_LOCAL_INTERFACE_H_
#define KDTREE_STK_LOCAL_INTERFACE_H_

#ifdef _OPENMP
#include <omp.h>
#include <stk_search/CommonSearchUtil.hpp>
#endif

#include "stk_search/kdtree/KDTree.hpp"


//
//  More general search for an arbitrary range type
//
template <typename DomainIdentifier, typename RangeIdentifier, typename DomainObjType, typename RangeObjType>
inline void local_coarse_search_kdtree(std::vector< std::pair<DomainObjType, DomainIdentifier> > const & local_domain,
                                 std::vector< std::pair<RangeObjType,  RangeIdentifier > > const & local_range,
                                 std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& searchResults,
                                 [[maybe_unused]] bool enforceSearchResultSymmetry = true,
                                 bool sortSearchResults = false)
{

#ifdef _OPENMP
  std::vector<std::vector<std::pair<DomainIdentifier, RangeIdentifier> > >
      threadLocalSearchResults( omp_get_max_threads() );
#endif

  if ((local_domain.size() > 0) && (local_range.size() > 0)) {

    //
    //  Need to convert range objects to actual box type objects for proximity search
    //

    using rangeValueType = typename RangeObjType::value_type;
    using RangeBox       = stk::search::Box<rangeValueType>;

    std::vector<RangeBox> rangeBoxes;
    rangeBoxes.reserve( local_range.size() );
    for (auto& [rangeObj, rangeId] : local_range) {
      rangeBoxes.emplace_back(RangeBox(rangeObj.get_x_min(), rangeObj.get_y_min(), rangeObj.get_z_min(),
                                       rangeObj.get_x_max(), rangeObj.get_y_max(), rangeObj.get_z_max()));

    }

    const stk::search::ProximitySearchTree_T<RangeBox> proxSearch(rangeBoxes);
    const unsigned numBoxDomain = local_domain.size();

#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
    {
      //
      //  Set the known return vector sizes
      //

      std::vector<int> overlapList;
#ifdef _OPENMP
      std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& interList   = threadLocalSearchResults[omp_get_thread_num()];
#else
      std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& interList   = searchResults;
#endif
      //
      //  Create an array to store interactions returned by the recursive search routines.  There are at maximum
      //  N interactions per object when searching N objects
      //

      //
      //  Loop over all boxAs in group1 and search them against those objects in group2
      //
#ifdef _OPENMP
#pragma omp for
#endif
      for(unsigned int iboxDomain = 0; iboxDomain < numBoxDomain; ++iboxDomain) {
        proxSearch.SearchForOverlap(local_domain[iboxDomain].first, overlapList);
        for(auto&& jboxRange : overlapList) {
          if(intersects(local_domain[iboxDomain].first, local_range[jboxRange].first)) {
            interList.emplace_back( local_domain[iboxDomain].second, local_range[jboxRange].second );
          }
        }
      }
    }
  }
#ifdef _OPENMP
  stk::search::concatenate_thread_lists(threadLocalSearchResults, searchResults);
#endif

  if (sortSearchResults) {
    std::sort(searchResults.begin(), searchResults.end());
  }
}



//
//  Most optimal search specific to actual box arguments
//
template <typename DomainIdentifier, typename RangeIdentifier, typename DomainObjType, typename RBoxNumType>
inline void local_coarse_search_kdtree(std::vector< std::pair<DomainObjType, DomainIdentifier> > const & local_domain,
                                 std::vector< std::pair<stk::search::Box<RBoxNumType>,  RangeIdentifier > > const & local_range,
                                 std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& searchResults,
                                 [[maybe_unused]] bool enforceSearchResultSymmetry = true,
                                 bool sortSearchResults = false)
{
  searchResults.clear();

#ifdef _OPENMP
  std::vector<std::vector<std::pair<DomainIdentifier, RangeIdentifier> > >
      threadLocalSearchResults( omp_get_max_threads() );
#endif

  {
    std::vector<stk::search::Box<RBoxNumType> > rangeBoxes( local_range.size() );
    for (size_t i=0; i < local_range.size(); ++i) {
      rangeBoxes[i] = local_range[i].first;
    }

    if ((local_domain.size() > 0) && (rangeBoxes.size() > 0)) {

      const stk::search::ProximitySearchTree_T<stk::search::Box<RBoxNumType> > proxSearch(rangeBoxes);
      const unsigned numBoxDomain = local_domain.size();

#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
      {
        //
        //  Set the known return vector sizes
        //

        std::vector<int> overlapList;
#ifdef _OPENMP
        std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& interList   = threadLocalSearchResults[omp_get_thread_num()];
#else
        std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& interList   = searchResults;
#endif

        //
        //  Create an array to store interactions returned by the recursive search routines.  There are at maximum
        //  N interactions per object when searching N objects
        //

        //
        //  Loop over all boxAs in group1 and search them against those objects in group2
        //
        interList.reserve((numBoxDomain*3)/2);
#ifdef _OPENMP
#pragma omp for
#endif
        for(unsigned int iboxDomain = 0; iboxDomain < numBoxDomain; ++iboxDomain) {
          proxSearch.SearchForOverlap(local_domain[iboxDomain].first, overlapList);
          for(auto&& jboxRange : overlapList) {
              interList.emplace_back( local_domain[iboxDomain].second, local_range[jboxRange].second );
            }
          }
        }
      }
    }
#ifdef _OPENMP
  stk::search::concatenate_thread_lists(threadLocalSearchResults, searchResults);
#endif
 
  if (sortSearchResults) { 
    std::sort(searchResults.begin(), searchResults.end());
  }
}


template <typename DomainIdentifier, typename RangeIdentifier, typename DomainObjType, typename RangeObjType>
inline void local_coarse_search_kdtree_driver(std::vector< std::pair<DomainObjType, DomainIdentifier> > const & local_domain,
                                        std::vector< std::pair<RangeObjType,  RangeIdentifier > > const & local_range,
                                        std::vector<std::pair<DomainIdentifier, RangeIdentifier> >& searchResults,
                                        bool sortSearchResults = false)
{
  const bool domain_has_more_boxes = local_domain.size() > local_range.size();
  if(domain_has_more_boxes)
  {
    local_coarse_search_kdtree(local_domain, local_range, searchResults, sortSearchResults);
  }
  else
  {
    std::vector<std::pair<RangeIdentifier, DomainIdentifier> > tempSearchResults;
    local_coarse_search_kdtree(local_range, local_domain, tempSearchResults, sortSearchResults);
    searchResults.reserve(tempSearchResults.size());
    for (auto& [range_id, domain_id] : tempSearchResults)
    {
      searchResults.emplace_back(domain_id, range_id);
    }

    if (sortSearchResults) {
      std::sort(searchResults.begin(), searchResults.end());
    }
  }
}

#endif
