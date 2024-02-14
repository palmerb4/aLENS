/**
 * @file CalcTubulinBind.hpp
 * @author Wen Yan (wenyan4work@gmail.com) and Adam Lamson
 * @brief
 * @version 0.1
 * @date 2019-01-04
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef CALCTUBULINBIND_HPP_
#define CALCTUBULINBIND_HPP_

#include "FDPS/particle_simulator.hpp"
#include "FusionBlock.hpp"

// SimToolbox module
#include "Sylinder/SylinderNear.hpp"
#include "Util/Logger.hpp"

bool evaluateFusion(const SylinderNearEP &syJ, const std::vector<const SylinderNearEP *> &srcPtrArr, const double dt,
                    const double randU01, const double tubulinBindingRate, const double tubulinBindingCutoffRadius,
                    FusionBlock &fusionBlock) {

    // The explicit KMC steps for binding a point-like particle to a collection of other point-like particles (with equal probability and a fixed cutoff radius):
    //  1. Get the total probability of binding to any of the microtubules
    //  2. Decide if the tubulin will bind to a microtubule
    //  3. If the tubulin will bind to a microtubule, decide which microtubule it will bind to
    // 0-------------------------------------------------------------------------------------------------------------------------------------------------1
    // 0-------------------------------------------------------total binding probability-------------------------------------------------------|---------1
    // 0--probability of binding to microtubule 0--|--probability of binding to microtubule 1--| ... |--probability of binding to microtule N--|---------1

    // Step 1 illustrated:
    // 0-------------------------------------------------------total binding probability-------------------------------------------------------|---------1
    // 0-------------------------------------------------------------randU01-----------------------------------------------X-------------------------Y---1
    // if randU01 < totalBindingProbability, then binding occurs. X binds but Y does not
    //
    // Step 2 illustrated:
    // 0--probability of binding to microtubule 0--|--probability of binding to microtubule 1--| ... |--probability of binding to microtule N--|---------1
    // 0--------------------------Z--------------------------W-------scaled randU01------------------------------------------------------------|---------1
    // if randU01 * totalBindingProbability within the range of the probability of binding to microtubule 0, we bind to microtubule 0. Otherwise, if
    // its within probability of binding to microtubule 0 to probability of binding to 0 + probability of binding to microtubule 1, we bind to microtubule 1,
    // and so on. For example, Z binds to microtubule 0 and W binds to microtubule 1.

    // Step 1: Get the total probability of binding to any of the microtubules. Note, tubulin only bind to the plus end of microtubules.
    std::vector<double> bindingProbabilities(srcPtrArr.size(), 0.0);
    double totalBindingProbability = 0;
    for (int i = 0; i < srcPtrArr.size(); i++) {
        const Evec3 plusEnd = ECmap3(srcPtrArr[i]->pos) + 0.5 * srcPtrArr[i]->length * ECmap3(srcPtrArr[i]->direction);
        const double dx = syJ.pos[0] - plusEnd[0];
        const double dy = syJ.pos[1] - plusEnd[1];
        const double dz = syJ.pos[2] - plusEnd[2];
        const double distToPlusEnd = std::sqrt(dx * dx + dy * dy + dz * dz);
        bindingProbabilities[i] = tubulinBindingRate * dt * (distToPlusEnd < tubulinBindingCutoffRadius);
        totalBindingProbability += bindingProbabilities[i];
    }

    // Step 2: Decide if the tubulin will bind to a microtubule using a Poisson process.
    // Rescale probabilities to add to 1 so that roll samples all events.
    double passProb = exp(-1. * totalBindingProbability);
    const double scaledRandU01 = randU01 * (totalBindingProbability + passProb);
    const bool bindOccurs = scaledRandU01 < totalBindingProbability;
    if (bindOccurs) {
        // Step 3: Decide which microtubule to bind to.
        double cumulativeProbability = 0;
        for (int i = 0; i < srcPtrArr.size(); ++i) {
            cumulativeProbability += bindingProbabilities[i];
            if (scaledRandU01 < cumulativeProbability) {
                // We bind to microtubule i. Populate the fusion block.
                fusionBlock.parentGid = srcPtrArr[i]->gid;
                fusionBlock.childGid = syJ.gid;

                fusionBlock.parentGlobalIndex = srcPtrArr[i]->globalIndex;
                fusionBlock.childGlobalIndex = syJ.globalIndex;

                fusionBlock.parentRank = srcPtrArr[i]->rank;
                fusionBlock.childRank = syJ.rank;

                fusionBlock.side = 1; // We always bind to the plus end of the microtubule.
                fusionBlock.childLength = syJ.length;
                break;
            }
        }
    }

    return bindOccurs;
}

/**
 * @brief The functor class for KMC protein-tubule binding
 *
 */
class CalcTubulinBind {
    double dt;                                        ///< timestep size
    double tubulinBindingRate;                        ///< binding rate
    double tubulinBindingCutoffRadius;                ///< binding cutoff radius
    std::shared_ptr<TRngPool> rngPoolPtr;             ///< rng generator
    std::shared_ptr<FusionBlockPool> &fusionPoolPtr; ///< fusion block pool

  public:
    /**
     * @brief Construct a new CalcTubulinBind object
     *
     * @param dt_
     * @param KBT_ 
     * @param rngPoolPtr_
     */
    CalcTubulinBind(std::shared_ptr<FusionBlockPool> &fusionPoolPtr_, double dt_, double tubulinBindingRate_,
                    double tubulinBindingCutoffRadius_, std::shared_ptr<TRngPool> &rngPoolPtr_)
        : fusionPoolPtr(fusionPoolPtr_), dt(dt_), tubulinBindingRate(tubulinBindingRate_),
          tubulinBindingCutoffRadius(tubulinBindingCutoffRadius_), rngPoolPtr(rngPoolPtr_) {
        assert(fusionPoolPtr && "FusionBlockPool is not initialized");
    }

    /**
     * @brief Functor interface required by MixPairInteraction
     *
     * @param ep_i target
     * @param Nip number of target
     * @param ep_j source
     * @param Njp number of source
     * @param forceNear computed force
     */
    void operator()(const SylinderNearEP *const ep_i, const PS::S32 Nip, const SylinderNearEP *const ep_j,
                    const PS::S32 Njp, ForceNear *const forceNear) {
        const int myThreadId = omp_get_thread_num();
        auto &fusionQue = (*fusionPoolPtr)[myThreadId];

        // At the onset of this function, we are given a set of source sylinders and a set of target sylinders. We do not,
        // however, know which sylinders are supposed to be considered as microtubules and which are supposed to be considered as
        // tubulin. We use the sylinder's group number to distinguish the two. Microtubules have a group number of 0, while tubulin
        // has a group number of 1.

        // Collect all microtubule (source) sylinders
        std::vector<const SylinderNearEP *> srcMicrotubulePtrArr;
        for (PS::S32 i = 0; i < Nip; ++i) {
            if (ep_i[i].group == 0) {
                srcMicrotubulePtrArr.push_back(&(ep_i[i]));
            }
        }

        // For each parent (target) tubulin, collect its neighboring microtubule (source) sylinders. We then pass this information to KMC.
        for (PS::S32 j = 0; j < Njp; ++j) {
            auto &syJ = ep_j[j];
            if (syJ.group == 1) {

                const double randU01 = rngPoolPtr->getU01(myThreadId);
                FusionBlock fusionBlock;
                const bool fusionFlag = evaluateFusion(syJ, srcMicrotubulePtrArr, dt, randU01, tubulinBindingRate,
                                                       tubulinBindingCutoffRadius, fusionBlock);
                if (fusionFlag) {
                    fusionQue.push_back(fusionBlock);
                }
            }
        }
    }
}; // class CalcTubulinBind

#endif
