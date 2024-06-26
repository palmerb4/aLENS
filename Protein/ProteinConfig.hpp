/**
 * @file ProteinConfig.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-01-04
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef PROTEINCONFIG_HPP_
#define PROTEINCONFIG_HPP_

#include "ProteinType.hpp"

#include "KMC/lookup_table.hpp"

#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief read the proteinConfig.yaml file to read in different types of
 * proteins
 *
 * The lifetime of ProteinConfig should be the entire lifetime of the simulation
 * because types[i].LUTablePtr holds the LUT table for each protein forever
 *
 */
class ProteinConfig {
  public:
    double KBT;
    std::string tubulinBindInteractionType; ///< The type of tubulin binding interaction to use. Options are 'explicit' and 'implicit'. 
    double defaultTubulinUnbindingRate;          ///< The default unbinding rate for tubulin.
    double proteinEnhancedTubulinUnbindingRate;  ///< The unbinding rate for tubulin when a protein is present at the end of the microtubule.
    double proteinEnhancementCutoffDistance;     ///< The distance from the end of the microtubule at which a protein enhances the unbinding rate of tubulin.
    double tubulinBindingRate;         ///< The rate at which tubulin binds to the end of a microtubule.

    // Only used by explicit tubulin model
    double tubulinBindingCutoffRadius = 0; ///< The radius around the end of the microtubule at which tubulin can bind.

    // Only used by implicit tubulin model
    double tubulinLength = 0; ///< The length of a tubulin.
    int tubulinLoadBalanceFrequency = 1; ///< The frequency (in number of timesteps between load balance calls) at which the 
                                     /// local number of tubulin is load balanced synchronized with the global count and redistributed.
    int initialFreeTubulinCount = 0; ///< The initial number of free tubulin in the global pool.

    std::vector<ProteinType> types; ///< settings for different types

    std::vector<int> freeNumber; ///< free number for each type

    std::vector<std::vector<double>> fixedLocations;
    ///< fixed location for each type

    std::unordered_map<int, int> tagLookUp; ///< hash table for map and iType

    /**
     * @brief Construct a new ProteinConfig object
     * 
     * @param proteinConfigFile 
     */
    explicit ProteinConfig(std::string proteinConfigFile);

    /**
     * @brief Destroy the ProteinConfig object
     * 
     */
    ~ProteinConfig();

    /**
     * @brief display settings
     *
     */
    void echo() const;
};

#endif
