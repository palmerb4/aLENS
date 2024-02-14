/**
 * @file FusionBlock.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-11-04
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef FUSIONBLOCK_HPP_
#define FUSIONBLOCK_HPP_

#include "Util/EigenDef.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <type_traits>
#include <vector>

/**
 * @brief Fusion information block
 *
 * Each block stores the information for one fusion.
 * The blocks are collected by FusionCollector and then used to fuse sylinders
 * If two sylinders are supposed to fuse, then we need to know the parent and child sylinders.
 * The parent is the sylinder which is supposed to absorb the child (Greek god style). As such, a child
 * may fuse to either the left or right-hand sides of the parent while holding the orientation of the parent fixed.
 * This interface allows the user to specify if two sylinders are supposed to fuse and if so, which one is the
 * parent, which is the child, and which side of the parent the child is supposed to fuse to.
 */
struct FusionBlock {
    int parentGid = GEO_INVALID_INDEX; ///< parent gid unique across all processes
    int childGid = GEO_INVALID_INDEX;  ///< child gid

    int parentGlobalIndex = GEO_INVALID_INDEX; ///< parent sequential unique global index
    int childGlobalIndex = GEO_INVALID_INDEX;  ///< child sequential unique global index

    int parentRank = -1; ///< rank of parent
    int childRank = -1;  ///< rank of child

    int side = -1;          ///< side of parent to fuse to 0 for left, 1 for right
    double childLength = 0; ///< length of child sylinder
};  // struct FusionBlock

static_assert(std::is_trivially_copyable<FusionBlock>::value, "");
static_assert(std::is_default_constructible<FusionBlock>::value, "");

using FusionBlockQue = std::deque<FusionBlock>;      ///< a queue contains blocks collected by one thread
using FusionBlockPool = std::vector<FusionBlockQue>; ///< a pool contains queues on different threads

#endif