#include "TubuleSystem.hpp"
#include "CalcProteinBind.hpp"
#include "CalcTubulinBind.hpp"
#include "FusionBlock.hpp"
#include "KMC/kmc.hpp"
#include "Protein/ProteinConfig.hpp"

// SimToolbox module
#include "MPI/CommMPI.hpp"
#include "Trilinos/ZDD.hpp"
#include "Util/GeoUtil.hpp"

#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTypeInt32Array.h>
#include <vtkTypeUInt8Array.h>
#include <vtkXMLPPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>

#include <unordered_map>

TubuleSystem::TubuleSystem(const std::string &configFileSystem, const std::string &configFileProtein,
                           const std::string &restartFile, //
                           int argc, char **argv)
    : proteinConfig(configFileProtein) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    if (rank == 0) {
        proteinConfig.echo();
    }

    // step 1 reinitialize tubule system
    // move rods after protein bindings are reconstructed
    // snapID and stepCount both ++ in this reinitialize() function
    rodSystem.reinitialize(configFileSystem, restartFile, argc, argv, false);
    rodSystem.prepareStep();
    MPI_Barrier(MPI_COMM_WORLD);

    // step 2 initialize shared resource
    rngPoolPtr = rodSystem.getRngPoolPtr();

    // step 3 reinitialize proteins and distribute
    buildLookupTable();
    proteinContainer.initialize();
    // load protein ascii file:
    // snapID has ++ in rodSystem.reinitialize()
    auto snapID = rodSystem.getSnapID() - 1;
    std::string baseFolder = rodSystem.getResultFolderWithID(snapID);
    std::string proteinVTKFile = baseFolder + std::string("Protein_") + std::to_string(snapID) + ".pvtp";
    readProteinVTK(proteinVTKFile);

    rodSystem.stepEuler();

    // step 5 setup MixPairInteraction object
    bindInteraction.initialize();

    Teuchos::TimeMonitor::zeroOutTimers();

    return;
}

TubuleSystem::TubuleSystem(const std::string &configFileSystem, const std::string &posFileTubule,
                           const std::string &configFileProtein,
                           const std::string &posFileProtein, //
                           int argc, char **argv)
    : proteinConfig(configFileProtein) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    if (rank == 0) {
        proteinConfig.echo();
    }

    // step 1 initialize tubule system
    rodSystem.initialize(configFileSystem, posFileTubule, argc, argv);
    MPI_Barrier(MPI_COMM_WORLD);

    // step 2 initialize shared resource
    rngPoolPtr = rodSystem.getRngPoolPtr();

    // step 2.5 Initialize the global tubulin pool
    // The global tubulin pool is used to keep track of the number of unbound tubulin.
    // GlobalTubulinPool(const int &initial_global_microtubule_count, const int &initial_global_tubulin_count,
    //                   const int &num_procs, const int rank, std::shared_ptr<TRngPool> &rng_pool_ptr)
    globalTubulinPoolPtr = std::make_shared<GlobalTubulinPool>(rodSystem.getContainer().getNumberOfParticleGlobal(),
                                          proteinConfig.initialFreeTubulinCount, nProcs, rank, rngPoolPtr.get());

    // step 3 initialize proteins and distribute
    buildLookupTable();
    proteinContainer.initialize();
    if (IOHelper::fileExist(posFileProtein)) {
        // if posFileProtein file exists, ignore numbers in configFileProtein
        setInitialProteinFromFile(posFileProtein);
    } else {
        setInitialProteinFromConfig();
    }

    // step 4 output initial configuration
    // simBox and Sylinder has been written inside SylinderSystem
    // Here, write protein Info only
    outputProteinData();

    rodSystem.writeResult();

    // step 5 setup MixPairInteraction object
    bindInteraction.initialize();

    rodSystem.setTimer(true);
    Teuchos::TimeMonitor::zeroOutTimers();

    // Prepare step needs called during initialization for the first call to step to work
    prepareStep();
    return;
}

bool TubuleSystem::end() {
    const auto dt = rodSystem.runConfig.dt;
    const auto timeTotal = rodSystem.runConfig.timeTotal;
    return dt * rodSystem.getStepCount() > timeTotal;
}

void TubuleSystem::prepareStep() {
    // step 1 prepare rodSystem
    // repartitioned if necessary
    rodSystem.prepareStep();
    auto &dinfo = rodSystem.getDomainInfo();

    // protein partition follow tubule partition
    proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());
    proteinContainer.exchangeParticle(rodSystem.getDomainInfoNonConst());

    const auto &runConfig = rodSystem.runConfig;
    if (runConfig.monolayer) {
        const double monoZ = (runConfig.simBoxHigh[2] + runConfig.simBoxLow[2]) / 2;
        const int nLocalProtein = proteinContainer.getNumberOfParticleLocal();
#pragma omp parallel for
        for (int i = 0; i < nLocalProtein; i++) {
            auto &pr = proteinContainer[i];
            auto pos = pr.getPosPtr();
            double posNew[3] = {pos[0], pos[1], monoZ};
            pr.setPos(posNew);
        }
    }

    setLookupTablePtr();
}

void TubuleSystem::step() {
    const double dt = rodSystem.runConfig.dt;
    spdlog::warn("CurrentTime {:8g}", rodSystem.getStepCount() * dt);

    using Teuchos::Time;
    using Teuchos::TimeMonitor;
    TimeMonitor::zeroOutTimers();

    // Global Timers across all MPI ranks
    // Timers do not accumulate over timesteps
    Teuchos::RCP<Teuchos::Time> mainLoopTimer = Teuchos::TimeMonitor::getNewCounter("aLENS main loop");
    Teuchos::RCP<Time> calcTubulinBindInteractionTimer =
        TimeMonitor::getNewCounter("0 calcTubulinBindInteractionTimer Time");
    Teuchos::RCP<Time> prepareStepTimer = TimeMonitor::getNewCounter("1 prepareStep Time");
    Teuchos::RCP<Time> updateProteinMotionTimer = TimeMonitor::getNewCounter("2 updateProteinMotion Time");
    Teuchos::RCP<Time> calcBindInteractionTimer = TimeMonitor::getNewCounter("3 calcBindInteraction Time");
    Teuchos::RCP<Time> outputProteinDataTimer = TimeMonitor::getNewCounter("4 outputProteinData Time");
    Teuchos::RCP<Time> setProteinConstraintTimer = TimeMonitor::getNewCounter("5 setProteinConstraint Time");
    Teuchos::RCP<Time> rodSystemTimer = TimeMonitor::getNewCounter("6 rodSystem Time");

    {
        TimeMonitor mon(*mainLoopTimer);

        // step 0 ompute the tubulin bind/unbind interactions
        // Note, this MUST come before prepareStep() because the total number of particles may change
        // as a result of binding and unbinding.
        {
            TimeMonitor mon(*calcTubulinBindInteractionTimer);
            calcTubulinBindInteraction();
            spdlog::debug("calcTubulinBindInteraction");
        }

        // step 1 prepare.
        // nothing moves
        {
            TimeMonitor mon(*prepareStepTimer);
            prepareStep();
            rodSystem.calcOrderParameter();
            spdlog::debug("prepareStep");
        }

        // step 2
        // MTs have moved at the end of the last timestep
        // MT info should be updated and protein move according to this updated MT
        // configuration this move includes diffusion and walking
        {
            TimeMonitor mon(*updateProteinMotionTimer);
            updateBindWithGid();
            updateProteinMotion();
            proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());
            spdlog::debug("updateProteinMotion");
        }

        // step 3 compute bind interaction.
        // protein ends have moved inside this function
        // this move includes only KMC binding/unbinding kinetics
        {
            TimeMonitor mon(*calcBindInteractionTimer);
            calcBindInteraction();
            proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());
            spdlog::debug("calcBindInteraction");
        }

        // write protein data
        {
            TimeMonitor mon(*outputProteinDataTimer);
            outputProteinData();
            spdlog::debug("outputProteinData");
        }

        // step 4 calculate bilateral constraints with protein binding information
        {
            TimeMonitor mon(*setProteinConstraintTimer);
            setProteinConstraints();
            spdlog::debug("setProteinConstraints");
        }

        // MAJOR STEP:
        // move tubules with binding force and Brownian & collision & bilateral
        // tubule data and protein data written in this step before moving.
        {
            TimeMonitor mon(*rodSystemTimer);
            rodSystem.runStep();
            rodSystem.calcConStress();
            spdlog::debug("rodSystemStep");
        }
    }

    rodSystem.printTimingSummary();
}

void TubuleSystem::calcBindInteraction() {
    auto &dinfo = rodSystem.getDomainInfoNonConst();
    bindInteraction.updateSystem(proteinContainer, rodSystem.getContainer(), dinfo);
    spdlog::debug("mixSystemUpdated");
    bindInteraction.updateTree();
    spdlog::debug("mixTreeUpdated");
    // bindInteraction.dumpSystem();

    CalcProteinBind interactionFtr(rodSystem.runConfig.dt, proteinConfig.KBT, rngPoolPtr);
    bindInteraction.computeForce(interactionFtr, dinfo);
    spdlog::debug("forceComputed");

    auto &result = bindInteraction.getForceResult();
    const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
    assert(result.size() == nProteinLocal);
#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nProteinLocal; i++) {
            auto &p = proteinContainer[i];
            p.bind = result[i];
            p.updateGeometryWithBind();
        }
    }
}

void TubuleSystem::calcTubulinBindInteraction() {
    if (proteinConfig.tubulinBindInteractionType == "explicit") {
        calcTubulinBindInteractionExplicit();
    } else if (proteinConfig.tubulinBindInteractionType == "implicit") {
        calcTubulinBindInteractionImplicit();
    } else {
        throw std::runtime_error(
            "Unknown tubulin bind interaction type. Valiud options are 'explicit' and 'implicit'.");
    }
}

void TubuleSystem::calcTubulinBindInteractionImplicit() {
    /* Methodology: 
    Instead of representing tubulin as individual particles, we represent them as a finite sized pool of tubulin. 
    Each tubulin within this pool has an equal probability of binding to each microtubule. We rely on the GlobalTubulinPool class 
    to keep track of the number of unbound tubulin and to count of number of tubulin that bind to each microtubule.

    What are out control parameters?
     - tubulinBindInteractionType: The type of tubulin binding interaction to use. Options are 'explicit' and 'implicit'. 
     - defaultTubulinUnbindingRate: The default unbinding rate for tubulin.
     - proteinEnhancedTubulinUnbindingRate: The unbinding rate for tubulin when a protein is present at the end of the microtubule.
     - proteinEnhancementCutoffDistance: The distance from the end of the microtubule at which a protein enhances the unbinding rate of tubulin.
     - tubulinBindingRate: The rate at which tubulin binds to the end of a microtubule.
     - tubulinLength: The length of a tubulin.
     - tubulinLoadBalanceFrequency: The frequency at which the local number of tubulin is load balanced synchronized with the global count and redistributed.
     - initialFreeTubulinCount: The initial number of free tubulin in the global pool.
     - dt: The time step size.
    */

    // Because the global count of tubulin is distributed among the processes where the local count is allowed to go out of sync with the global count,
    // leading to the potential for load imbalance, we occasionally synchronize the global count of tubulin and redistribute it among the processes.
    if (rodSystem.getStepCount() % proteinConfig.tubulinLoadBalanceFrequency == 0) {
        globalTubulinPoolPtr->synchronize();
        globalTubulinPoolPtr->distribute();
    }

    ////////////////////////////////////
    // Setup tubulin bind interaction //
    ////////////////////////////////////
    // To ensure that binding and unbinding do not impact one another within the same timestep, we interleave the unbinding and binding steps.
    {
        // Count the number of tubulin that bind to each microtubule.
        globalTubulinPoolPtr->count_tubulin_binding(proteinConfig.tubulinBindingRate, rodSystem.runConfig.dt);

        // Incrementing the length of the microtubules based on this count is deferred until after the unbinding step to avoid introducing bias.
    }

    ////////////////////////////////
    // Tubulin unbind interaction //
    ////////////////////////////////
    {
        // A tubulin unbinds based on a Poisson process with a rate that depends on
        // the presence of a protein at the end of the microtubule.

        // Step 1: Calculate the unbinding rate for each microtubule based on the presence of a protein at the end.

        // Loop over each protein, determine if it's close enough to the end of a microtubule to enhance the unbinding
        // rate of tubulin or not. Store the rate per microtubule and the sequential global index of said microtubule.
        // Perform an all-to-all communication to send the rates and global indices to everyone. Then, sort the rates by
        // global index. After all of that, we can loop over the microtubules and fetch the rate using the global index.
        std::vector<int> globalIndicesToCommunicate;
        std::vector<int> hasEnhancedUnbindingFlagToCommunicate;
        std::vector<std::vector<int>> globalIndicesToCommunicatePool(omp_get_max_threads());
        std::vector<std::vector<int>> hasEnhancedUnbindingFlagToCommunicatePool(omp_get_max_threads());
        const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            globalIndicesToCommunicatePool[threadId].clear();
            hasEnhancedUnbindingFlagToCommunicatePool[threadId].clear();
#pragma omp for
            for (int i = 0; i < nProteinLocal; i++) {
                const auto &p = proteinContainer[i];
                if ((p.property.tag == 0) && (p.bind.gidBind[0] != ID_UB) &&
                    (0.5 * p.bind.lenBind[0] - p.bind.distBind[0] < proteinConfig.proteinEnhancementCutoffDistance)) {
                    // For these proteins, only the first head binds. The second shouldn't exist and is only present for consistency with the other protein types.
                    // Note, distBind ranges from [-0.5 length, 0.5 length] where length is the length of the microtubule.

                    // The protein is the correct type, and it's close enough to the end of the microtubule to cause increased unbinding of tubulin.
                    assert(p.bind.indexBind[0] != GEO_INVALID_INDEX &&
                           "The global index of the bound microtubule is invalid. Tell the developers.");
                    globalIndicesToCommunicatePool[threadId].push_back(p.bind.indexBind[0]);
                    hasEnhancedUnbindingFlagToCommunicatePool[threadId].push_back(
                        1); // MPI doesn't like bools. We'll use 0 and 1 instead.
                }
            }
        }

        // Merge the thread-local pools.
        for (int i = 0; i < omp_get_max_threads(); i++) {
            globalIndicesToCommunicate.insert(globalIndicesToCommunicate.end(),
                                              globalIndicesToCommunicatePool[i].begin(),
                                              globalIndicesToCommunicatePool[i].end());
            hasEnhancedUnbindingFlagToCommunicate.insert(hasEnhancedUnbindingFlagToCommunicate.end(),
                                                         hasEnhancedUnbindingFlagToCommunicatePool[i].begin(),
                                                         hasEnhancedUnbindingFlagToCommunicatePool[i].end());
        }

        // Perform a global all gather to collect the global indices and flags.

        // Each process needs to know how many indices it will send and receive from every other process.
        int nProcs = rodSystem.getCommRcp()->getSize();
        int localSize = globalIndicesToCommunicate.size(); // Size of local data to send
        std::vector<int> allSizes(nProcs);                 // Vector to hold the sizes of data from all processes
        MPI_Allgather(&localSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Each process calculates the displacements where each segment of received data should be placed in the receive buffer.
        std::vector<int> displs(nProcs);
        int totalSize = 0;
        for (int i = 0; i < nProcs; ++i) {
            displs[i] = totalSize;
            totalSize += allSizes[i];
        }

        // Gather the data
        std::vector<int> allGlobalIndices(totalSize);
        std::vector<int> allHasEnhancedUnbindingFlags(totalSize);
        MPI_Allgatherv(globalIndicesToCommunicate.data(), localSize, MPI_INT, allGlobalIndices.data(), allSizes.data(),
                       displs.data(), MPI_INT, MPI_COMM_WORLD);
        MPI_Allgatherv(hasEnhancedUnbindingFlagToCommunicate.data(), localSize, MPI_INT,
                       allHasEnhancedUnbindingFlags.data(), allSizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        // Now, we need to sort and unique the global indices and flag. If two global indices are the same, we take the max of the has enhanced unbinding flags.
        // To perform this sorting, we will create a single vector with size number of global particles and initialized to false. The global index can be used to
        // index this vector. We will index into this vector and store the max of the unsorted flags. As a result, if there are multiple entries
        // for the same rod, the rod should be marked with true if at least one instance of its flag is true.
        int globalNumberOfRods = 0;
        int localNumberOfRods = rodSystem.getContainer().getNumberOfParticleLocal();
        MPI_Allreduce(&localNumberOfRods, &globalNumberOfRods, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        std::vector<int> globalHasEnhancedUnbindingFlags(globalNumberOfRods, 0);

#pragma omp parallel for
        for (int i = 0; i < allHasEnhancedUnbindingFlags.size(); i++) {

#pragma omp critical
            globalHasEnhancedUnbindingFlags[allGlobalIndices[i]] =
                std::max(globalHasEnhancedUnbindingFlags[allGlobalIndices[i]], allHasEnhancedUnbindingFlags[i]);
        }

        // Step 2: Perform unbinding based on the calculated rates.
        const int nRodLocal = rodSystem.getContainer().getNumberOfParticleLocal();
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nRodLocal; i++) {
                auto &sy = rodSystem.getContainerNonConst()[i];

                // Check if the microtubule should unbind a tubulin based on a Poisson process with the calculated rate.
                const double randU01 = rngPoolPtr->getU01(threadId);
                const bool sufficientlyLong =
                    sy.length >
                    proteinConfig.tubulinLength + 1e-12; // NOTE, we only unbind tubulin if the microtubule has at least one tubulin-worth of length.

                assert(sy.globalIndex != GEO_INVALID_INDEX &&
                       "The global index of a microtubule is invalid. Tell the developers.");
                const bool enhancedUnbindingFlag = globalHasEnhancedUnbindingFlags[sy.globalIndex];
                const double tubulinUnbindingRate = enhancedUnbindingFlag
                                                        ? proteinConfig.proteinEnhancedTubulinUnbindingRate
                                                        : proteinConfig.defaultTubulinUnbindingRate;

                if ((randU01 < tubulinUnbindingRate * rodSystem.runConfig.dt) && sufficientlyLong) {
                    // Unbind a tubulin from the microtubule and add the tubulin to the global unbound pool. Note, this must be done in a thread-safe manner.
#pragma omp critical
                    globalTubulinPoolPtr->increment();

                    // Reduce the length of the microtubule by the tubulin length.
                    sy.length -= proteinConfig.tubulinLength;

                    // Shift the position to account for the change in length.
                    const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
                    Emap3(sy.pos) -= direction * proteinConfig.tubulinLength / 2.0;
                }
            }
        }

        // Step 3. Perform the unbinding of end-bound proteins that are connected to microtubules that unbound a tubulin and now lie outside the microtubule's centerline.
        // Note, the microtubules have changed length, so bind.lenBind is out of date. We must fetch the connected microtubule's
        // length from the sylinder container. We can use the ZDD for this.

        // This step requires that we create another global map from global index to sylinder length.
        // The last step was complicated because we had to from an unknown number of proteins to the global indices of the microtubules they were connected to.
        // On the other hand, this step is rather straightforward because we simply need to map from GID to length. We can use the ZDD for this.
        const auto &rodContainer = rodSystem.getContainer();
        ZDD<double> rodLengthDataDirectory(nRodLocal);
        rodLengthDataDirectory.gidOnLocal.resize(nRodLocal);
        rodLengthDataDirectory.dataOnLocal.resize(nRodLocal);
#pragma omp parallel for
        for (int t = 0; t < nRodLocal; t++) {
            rodLengthDataDirectory.gidOnLocal[t] = rodContainer[t].gid;
            rodLengthDataDirectory.dataOnLocal[t] = rodContainer[t].length;
        }
        rodLengthDataDirectory.buildIndex();

        // step 2 put id to find. two ids per protein
        rodLengthDataDirectory.gidToFind.resize(2 * nProteinLocal);
        rodLengthDataDirectory.dataToFind.resize(2 * nProteinLocal);
#pragma omp parallel for
        for (int p = 0; p < nProteinLocal; p++) {
            // for gidBind = ID_UB, ZDD fills findData with invalid data.
            rodLengthDataDirectory.gidToFind[2 * p + 0] = proteinContainer[p].bind.gidBind[0];
            rodLengthDataDirectory.gidToFind[2 * p + 1] = proteinContainer[p].bind.gidBind[1];
        }
        rodLengthDataDirectory.find();

#pragma omp parallel for
        for (int i = 0; i < nProteinLocal; i++) {
            auto &p = proteinContainer[i];

            for (int e = 0; e < 2; e++) {
                const bool endBound = (p.bind.gidBind[e] != ID_UB);
                if (endBound) {
                    const double length = rodLengthDataDirectory.dataToFind[2 * i + e];
                    const bool unbindOccurs =
                        (p.bind.distBind[e] > 0.5 * length) || (p.bind.distBind[e] < -0.5 * length);
                    if (unbindOccurs) {
                        p.bind.setUnBind(e);
                    }
                }
            }
        }
    }

    ///////////////////////////////////////
    // Finalize tubulin bind interaction //
    ///////////////////////////////////////
    {
        // Loop over each microtubule and increment its length by the number of tubulin that bind to it.
        // Shift its position accordingly.
        const int nRodLocal = rodSystem.getContainer().getNumberOfParticleLocal();
        for (int i = 0; i < nRodLocal; i++) {
            auto &sy = rodSystem.getContainerNonConst()[i];

            // Increment the length of the microtubule by the number of tubulin that bind to it.
            const int bindCount = globalTubulinPoolPtr->get_bind_count(sy.globalIndex);
            const double changeInLength = bindCount * proteinConfig.tubulinLength;
            sy.length += changeInLength;

            // Shift the position to account for the change in length.
            const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
            Emap3(sy.pos) += direction * changeInLength / 2.0;
        }
    }
}

void TubuleSystem::calcTubulinBindInteractionExplicit() {
    // TUBULIN MUST BE LENGTH ZERO (aka, spheres)
    // MICROTUBULES MUST BE IN GROUP 0 AND TUBULIN IN GROUP 1
    // PROTEINS SHOULD HAVE TAG 0

#ifndef NDEBUG
    // Test our assumptions:

    {
        // We assume that tubulin are length zero (aka, spheres).
        // We assume that the tubulinBindingCutoffRadius is larger than the radius of a tubulin.
        const int nRodLocal = rodSystem.getContainer().getNumberOfParticleLocal();
        for (int i = 0; i < nRodLocal; i++) {
            const auto &sy = rodSystem.getContainer()[i];
            const bool isTubulin = (sy.group == 1);
            if (isTubulin) {
                assert(sy.length < 1e-12 && "Tubulin must be length zero (aka, spheres).");
                assert(sy.radius < proteinConfig.tubulinBindingCutoffRadius &&
                       "The tubulin binding cutoff radius must be larger than the radius of a tubulin.");
            }
        }
    }
#endif

    ////////////////////////////////
    // Tubulin unbind interaction //
    ////////////////////////////////

    // What are out control parameters?
    //  - tubulinBindInteractionType: The type of tubulin binding interaction to use. Options are 'explicit' and 'implicit'. 
    //  - defaultTubulinUnbindingRate: The default unbinding rate for tubulin.
    //  - proteinEnhancedTubulinUnbindingRate: The unbinding rate for tubulin when a protein is present at the end of the microtubule.
    //  - proteinEnhancementCutoffDistance: The distance from the end of the microtubule at which a protein enhances the unbinding rate of tubulin.
    //  - tubulinBindingRate: The rate at which tubulin binds to the end of a microtubule.
    //  - tubulinBindingCutoffRadius: The radius around the end of the microtubule at which tubulin can bind. This should be larger than the radius of a tubulin.
    //  - dt: The time step size.

    std::vector<Sylinder> newSylinders;
    {
        // A tubulin unbinds based on a Poisson process with a rate that depends on
        // the presence of a protein at the end of the microtubule.

        // Step 1: Calculate the unbinding rate for each microtubule based on the presence of a protein at the end.

        // Loop over each protein, determine if it's close enough to the end of a microtubule to enhance the unbinding rate of tubulin or not. Store the rate per microtubule
        // and the sequential global index of said microtubule. Perform an all-to-all communication to send the rates and global indices to everyone. Then, sort the rates by global index.
        // After all of that, we can loop over the microtubules and fetch the rate using the global index.
        std::vector<int> globalIndicesToCommunicate;
        std::vector<int> hasEnhancedUnbindingFlagToCommunicate;
        std::vector<std::vector<int>> globalIndicesToCommunicatePool(omp_get_max_threads());
        std::vector<std::vector<int>> hasEnhancedUnbindingFlagToCommunicatePool(omp_get_max_threads());
        const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            globalIndicesToCommunicatePool[threadId].clear();
            hasEnhancedUnbindingFlagToCommunicatePool[threadId].clear();
#pragma omp for
            for (int i = 0; i < nProteinLocal; i++) {
                const auto &p = proteinContainer[i];
                if ((p.property.tag == 0) && (p.bind.gidBind[0] != ID_UB) &&
                    (0.5 * p.bind.lenBind[0] - p.bind.distBind[0] < proteinConfig.proteinEnhancementCutoffDistance)) {
                    // For these proteins, only the first head binds. The second shouldn't exist and is only present for consistency with the other protein types.
                    // Note, distBind ranges from [-0.5 length, 0.5 length] where length is the length of the microtubule.

                    // The protein is the correct type, and it's close enough to the end of the microtubule to cause increased unbinding of tubulin.
                    assert(p.bind.indexBind[0] != GEO_INVALID_INDEX &&
                           "The global index of the bound microtubule is invalid. Tell the developers.");
                    globalIndicesToCommunicatePool[threadId].push_back(p.bind.indexBind[0]);
                    hasEnhancedUnbindingFlagToCommunicatePool[threadId].push_back(
                        1); // MPI doesn't like bools. We'll use 0 and 1 instead.
                }
            }
        }

        // Merge the thread-local pools.
        for (int i = 0; i < omp_get_max_threads(); i++) {
            globalIndicesToCommunicate.insert(globalIndicesToCommunicate.end(),
                                              globalIndicesToCommunicatePool[i].begin(),
                                              globalIndicesToCommunicatePool[i].end());
            hasEnhancedUnbindingFlagToCommunicate.insert(hasEnhancedUnbindingFlagToCommunicate.end(),
                                                         hasEnhancedUnbindingFlagToCommunicatePool[i].begin(),
                                                         hasEnhancedUnbindingFlagToCommunicatePool[i].end());
        }

        // Perform a global all gather to collect the global indices and flags.

        // Each process needs to know how many indices it will send and receive from every other process.
        int nProcs = rodSystem.getCommRcp()->getSize();
        int localSize = globalIndicesToCommunicate.size(); // Size of local data to send
        std::vector<int> allSizes(nProcs);                 // Vector to hold the sizes of data from all processes
        MPI_Allgather(&localSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Each process calculates the displacements where each segment of received data should be placed in the receive buffer.
        std::vector<int> displs(nProcs);
        int totalSize = 0;
        for (int i = 0; i < nProcs; ++i) {
            displs[i] = totalSize;
            totalSize += allSizes[i];
        }

        // Gather the data
        std::vector<int> allGlobalIndices(totalSize);
        std::vector<int> allHasEnhancedUnbindingFlags(totalSize);
        MPI_Allgatherv(globalIndicesToCommunicate.data(), localSize, MPI_INT, allGlobalIndices.data(), allSizes.data(),
                       displs.data(), MPI_INT, MPI_COMM_WORLD);
        MPI_Allgatherv(hasEnhancedUnbindingFlagToCommunicate.data(), localSize, MPI_INT,
                       allHasEnhancedUnbindingFlags.data(), allSizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        // Now, we need to sort and unique the global indices and flag. If two global indices are the same, we take the max of the has enhanced unbinding flags.
        // To perform this sorting, we will create a single vector with size number of global particles and initialized to false. The global index can be used to
        // index this vector. We will index into this vector and store the max of the unsorted flags. As a result, if there are multiple entries
        // for the same rod, the rod should be marked with true if at least one instance of its flag is true.
        int globalNumberOfRods = 0;
        int localNumberOfRods = rodSystem.getContainer().getNumberOfParticleLocal();
        MPI_Allreduce(&localNumberOfRods, &globalNumberOfRods, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        std::vector<int> globalHasEnhancedUnbindingFlags(globalNumberOfRods, 0);

#pragma omp parallel for
        for (int i = 0; i < allHasEnhancedUnbindingFlags.size(); i++) {

#pragma omp critical
            globalHasEnhancedUnbindingFlags[allGlobalIndices[i]] =
                std::max(globalHasEnhancedUnbindingFlags[allGlobalIndices[i]], allHasEnhancedUnbindingFlags[i]);
        }

        // Step 2: Perform unbinding based on the calculated rates.
        // Create a thread-local pool of particles-to-add. Each thread will append to its own queue and we'll merge them at the end.
        std::vector<std::vector<Sylinder>> newSylindersPool(omp_get_max_threads());
        for (auto &pool : newSylindersPool) {
            pool.clear();
        }

        const int nRodLocal = rodSystem.getContainer().getNumberOfParticleLocal();
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nRodLocal; i++) {
                auto &sy = rodSystem.getContainerNonConst()[i];
                const bool isMicrotubule = (sy.group == 0);
                if (isMicrotubule) {
                    // Check if the microtubule should unbind a tubulin based on a Poisson process with the calculated rate.
                    const double randU01 = rngPoolPtr->getU01(threadId);
                    const bool tooShort =
                        sy.length < 4 * sy.radius; // NOTE, we set the minimum microtubule size to be 2 tubulin

                    assert(sy.globalIndex != GEO_INVALID_INDEX &&
                           "The global index of a microtubule is invalid. Tell the developers.");
                    const bool enhancedUnbindingFlag = globalHasEnhancedUnbindingFlags[sy.globalIndex];
                    const double tubulinUnbindingRate = enhancedUnbindingFlag
                                                            ? proteinConfig.proteinEnhancedTubulinUnbindingRate
                                                            : proteinConfig.defaultTubulinUnbindingRate;

                    if ((randU01 < tubulinUnbindingRate * rodSystem.runConfig.dt) && !tooShort) {
                        // Unbind a tubulin from the microtubule.

                        // Create a new sylinder with the same properties as the old one, but with tubulin length and shifted position.
                        // The position is chosen randomly within the sphere of radius TubulinBindingCutoffRadius around the sylinder endpoint.
                        // Using a truly random position may cause collisions with the microtubule, so we only use a random position outside one radius of the microtubule to mediate this.
                        Sylinder newSy = sy;
                        newSy.length = 0;
                        newSy.group = 1; // Group 1 is the tubulin group.

                        Equatn unitRandomOrient;
                        EquatnHelper::setUnitRandomEquatn(unitRandomOrient, rngPoolPtr->getU01(threadId),
                                                          rngPoolPtr->getU01(threadId), rngPoolPtr->getU01(threadId));
                        Evec3 randomUnitDir = unitRandomOrient * Evec3(0, 0, 1);
                        const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
                        const double randomRadius =
                            sy.radius +
                            rngPoolPtr->getU01(threadId) * (proteinConfig.tubulinBindingCutoffRadius - sy.radius);
                        Emap3(newSy.pos) =
                            Emap3(sy.pos) + 0.5 * sy.length * direction +
                            rngPoolPtr->getU01(threadId) * proteinConfig.tubulinBindingCutoffRadius * randomUnitDir;

                        // Add the new sylinder to the thread-local pool.
                        newSylindersPool[threadId].push_back(newSy);

                        // Reduce the length of the microtubule by the 2 radius of a tubulin.
                        sy.length -= 2 * sy.radius;

                        // Shift the length by the half the tubulin size in the direction of the microtubule's orientation.
                        Emap3(sy.pos) -= direction * sy.radius;
                    }
                }
            }
        }

        // Step 3. Perform the unbinding of end-bound proteins that are connected to microtubules that unbound a tubulin.
        // Note, the microtubules have changed length, so bind.lenBind is out of date. We must fetch the connected microtubule's
        // length from the sylinder container. We can use the ZDD for this.

        // This step requires that we create another global map from global index to sylinder length.
        // The last step was complicated because we had to from an unknown number of proteins to the global indices of the microtubules they were connected to.
        // On the other hand, this step is rather straight forward because we simply need to map from GID to length. We can use the ZDD for this.

        const auto &rodContainer = rodSystem.getContainer();
        ZDD<double> rodLengthDataDirectory(nRodLocal);
        rodLengthDataDirectory.gidOnLocal.resize(nRodLocal);
        rodLengthDataDirectory.dataOnLocal.resize(nRodLocal);
#pragma omp parallel for
        for (int t = 0; t < nRodLocal; t++) {
            rodLengthDataDirectory.gidOnLocal[t] = rodContainer[t].gid;
            rodLengthDataDirectory.dataOnLocal[t] = rodContainer[t].length;
        }
        rodLengthDataDirectory.buildIndex();

        // step 2 put id to find. two ids per protein
        rodLengthDataDirectory.gidToFind.resize(2 * nProteinLocal);
        rodLengthDataDirectory.dataToFind.resize(2 * nProteinLocal);
#pragma omp parallel for
        for (int p = 0; p < nProteinLocal; p++) {
            // for gidBind = ID_UB, ZDD fills findData with invalid data.
            rodLengthDataDirectory.gidToFind[2 * p + 0] = proteinContainer[p].bind.gidBind[0];
            rodLengthDataDirectory.gidToFind[2 * p + 1] = proteinContainer[p].bind.gidBind[1];
        }
        rodLengthDataDirectory.find();

#pragma omp parallel for
        for (int i = 0; i < nProteinLocal; i++) {
            auto &p = proteinContainer[i];

            for (int e = 0; e < 2; e++) {
                const bool endBound = (p.bind.gidBind[e] != ID_UB);
                if (endBound) {
                    const double length = rodLengthDataDirectory.dataToFind[2 * i + e];
                    const bool unbindOccurs =
                        (p.bind.distBind[e] > 0.5 * length) || (p.bind.distBind[e] < -0.5 * length);
                    if (unbindOccurs) {
                        p.bind.setUnBind(e);
                    }
                }
            }
        }

        // Finally, merge the pools and add the new sylinders to the system.
        // Note, we defer adding the new sylinders to the system to avoid having a tubulin unbind and then immediately rebind within the same time step.
        // This also helps keep the LIDs and GIDs of the sylinders consistent during binding.
        for (auto &pool : newSylindersPool) {
            newSylinders.insert(newSylinders.end(), pool.begin(), pool.end());
        }
    }

    //////////////////////////////
    // Tubulin bind interaction //
    //////////////////////////////
    {
        auto fusionPoolPtr = std::make_shared<FusionBlockPool>();
        fusionPoolPtr->resize(omp_get_max_threads());
        for (auto &queue : *fusionPoolPtr) {
            queue.clear();
        }

        CalcTubulinBind interactionFtr(fusionPoolPtr, rodSystem.runConfig.dt, proteinConfig.tubulinBindingRate,
                                       proteinConfig.tubulinBindingCutoffRadius, rngPoolPtr);

        auto &dinfo = rodSystem.getDomainInfoNonConst();
        auto &sylinderContainer = rodSystem.getContainerNonConst();

        rodSystem.setTreeSylinder();
        auto &treeSylinderNearPtr = rodSystem.getTreeSylinderNearPtr();
        TEUCHOS_ASSERT(treeSylinderNearPtr);
        treeSylinderNearPtr->calcForceAll(interactionFtr, sylinderContainer, dinfo);

        // At this point, fusionPoolPtr contains the fusion blocks in a thread pool.

        // Loop over each fusion block and collect the markedForDeletion flags, bind counts, and sylinder global indices.
        // We'll need to perform a similar global communication scheme as used during unbinding.
        std::vector<int> childGlobalIndicesToCommunicate;
        std::vector<int> childMarkedForDeletionFlagsToCommunicate;
        std::vector<int> parentGlobalIndicesToCommunicate;
        std::vector<int> parentBindCountsToCommunicate;
        std::vector<std::vector<int>> childGlobalIndicesToCommunicatePool(omp_get_max_threads());
        std::vector<std::vector<int>> childMarkedForDeletionFlagsToCommunicatePool(omp_get_max_threads());
        std::vector<std::vector<int>> parentGlobalIndicesToCommunicatePool(omp_get_max_threads());
        std::vector<std::vector<int>> parentBindCountsToCommunicatePool(omp_get_max_threads());
        const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
        const auto &fusionPool = *fusionPoolPtr;
        const int nThreads = fusionPool.size();
        // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
        for (int threadId = 0; threadId < nThreads; threadId++) {
            // each thread process a queue
            childGlobalIndicesToCommunicatePool[threadId].clear();
            childMarkedForDeletionFlagsToCommunicatePool[threadId].clear();
            parentGlobalIndicesToCommunicatePool[threadId].clear();
            parentBindCountsToCommunicatePool[threadId].clear();

            const auto &fusionBlockQue = fusionPool[threadId];
            const int fusionBlockNum = fusionBlockQue.size();

            for (int j = 0; j < fusionBlockNum; j++) {
                // For each fusion block, store
                // TODO(palmerb4): Is it possible to get duplicate fusion blocks? If so, we need to handle this case within CalcTubulinBind.
                auto &fusionBlock = fusionBlockQue[j];

                // Store the child's global index and the markedForDeletion flag.
                childGlobalIndicesToCommunicatePool[threadId].push_back(fusionBlock.childGlobalIndex);
                childMarkedForDeletionFlagsToCommunicatePool[threadId].push_back(
                    1); // MPI doesn't like bools. We'll use 0 and 1 instead.

                // Store the parent's global index and the bind count.
                parentGlobalIndicesToCommunicatePool[threadId].push_back(fusionBlock.parentGlobalIndex);
                parentBindCountsToCommunicatePool[threadId].push_back(
                    1); // For now, just one. We'll reduce this to get the total bind count.
            }
        }

        // Merge the thread-local pools.
        for (int i = 0; i < nThreads; i++) {
            childGlobalIndicesToCommunicate.insert(childGlobalIndicesToCommunicate.end(),
                                                   childGlobalIndicesToCommunicatePool[i].begin(),
                                                   childGlobalIndicesToCommunicatePool[i].end());
            childMarkedForDeletionFlagsToCommunicate.insert(childMarkedForDeletionFlagsToCommunicate.end(),
                                                            childMarkedForDeletionFlagsToCommunicatePool[i].begin(),
                                                            childMarkedForDeletionFlagsToCommunicatePool[i].end());
            parentGlobalIndicesToCommunicate.insert(parentGlobalIndicesToCommunicate.end(),
                                                    parentGlobalIndicesToCommunicatePool[i].begin(),
                                                    parentGlobalIndicesToCommunicatePool[i].end());
            parentBindCountsToCommunicate.insert(parentBindCountsToCommunicate.end(),
                                                 parentBindCountsToCommunicatePool[i].begin(),
                                                 parentBindCountsToCommunicatePool[i].end());
        }

        // Perform a global all gather to collect the global indices and flags.

        // Each process needs to know how many indices it will send and receive from every other process.
        int nProcs = rodSystem.getCommRcp()->getSize();
        int localSize = childGlobalIndicesToCommunicate.size(); // Size of local data to send
        std::vector<int> allSizes(nProcs);                      // Vector to hold the sizes of data from all processes
        MPI_Allgather(&localSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Each process calculates the displacements where each segment of received data should be placed in the receive buffer.
        std::vector<int> displs(nProcs);
        int totalSize = 0;
        for (int i = 0; i < nProcs; ++i) {
            displs[i] = totalSize;
            totalSize += allSizes[i];
        }

        // Gather the data
        std::vector<int> allChildGlobalIndices(totalSize);
        std::vector<int> allChildMarkedForDeletionFlags(totalSize);
        std::vector<int> allParentGlobalIndices(totalSize);
        std::vector<int> allParentBindCounts(totalSize);
        MPI_Allgatherv(childGlobalIndicesToCommunicate.data(), localSize, MPI_INT, allChildGlobalIndices.data(),
                       allSizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        MPI_Allgatherv(childMarkedForDeletionFlagsToCommunicate.data(), localSize, MPI_INT,
                       allChildMarkedForDeletionFlags.data(), allSizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        MPI_Allgatherv(parentGlobalIndicesToCommunicate.data(), localSize, MPI_INT, allParentGlobalIndices.data(),
                       allSizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        MPI_Allgatherv(parentBindCountsToCommunicate.data(), localSize, MPI_INT, allParentBindCounts.data(),
                       allSizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        // Now, we need to sort and unique the global indices and flag. If two global indices are the same, we take the max of the marked for deletion flag and the sum of the bind counts.
        // To perform this sorting, we will create a single globalChildMarkedForDeletionFlags and globalParentBindCounts vector with size number of global particles and initialized to false and 0.
        // The global index can be used to index this vector.
        int globalNumberOfRods = 0;
        int localNumberOfRods = rodSystem.getContainer().getNumberOfParticleLocal();
        MPI_Allreduce(&localNumberOfRods, &globalNumberOfRods, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        std::vector<int> globalChildMarkedForDeletionFlags(globalNumberOfRods, 0);
        std::vector<int> globalParentBindCounts(globalNumberOfRods, 0);

#pragma omp parallel for
        for (int i = 0; i < allChildMarkedForDeletionFlags.size(); i++) {
            const int childGlobalIndex = allChildGlobalIndices[i];

#pragma omp critical
            globalChildMarkedForDeletionFlags[childGlobalIndex] =
                std::max(globalChildMarkedForDeletionFlags[childGlobalIndex], allChildMarkedForDeletionFlags[i]);
        }

#pragma omp parallel for
        for (int i = 0; i < allParentBindCounts.size(); i++) {
            const int parentGlobalIndex = allParentGlobalIndices[i];

#pragma omp atomic
            globalParentBindCounts[parentGlobalIndex] += allParentBindCounts[i];
        }

        // Now, after all that, we can finally grow the microtubules and mark the tubulin for deletion.
        const int nRodLocal = rodSystem.getContainer().getNumberOfParticleLocal();
        std::vector<bool> localMarkedForDeletion(nRodLocal, false);
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nRodLocal; i++) {
                auto &sy = rodSystem.getContainerNonConst()[i];
                const bool isMicrotubule = (sy.group == 0);
                if (isMicrotubule) {
                    // Grow length by the child's total length (tubulin are sphere's, so length is zero but total length is 2 * radius)
                    // Grow by as many children as we consume.
                    const double growLength = sy.radius * 2 * globalParentBindCounts[sy.globalIndex];
                    sy.length += growLength;

                    // Shift position by the child's total length (along our axis) toward the side that the child fused to.
                    const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
                    Emap3(sy.pos) += direction * growLength / 2.0;
                } else {
                    // Mark the tubulin for deletion.
                    localMarkedForDeletion[i] = globalChildMarkedForDeletionFlags[sy.globalIndex];
                }
            }
        }

        // Collect and remove all sylinders marked for deletion.
        // Warning: This will change BOTH the LIDs and the GIDs of the sylinders.
        // As a result, we will need to update indexBind and gidBind for the proteins using prepareStep.
        std::vector<int> lidsToDelete;
        for (int i = 0; i < localMarkedForDeletion.size(); i++) {
            if (localMarkedForDeletion[i]) {
                lidsToDelete.push_back(i);
            }
        }
        sylinderContainer.removeParticle(lidsToDelete.data(), lidsToDelete.size());
    }

    // Add the new tubulin
    rodSystem.addNewSylinder(newSylinders);
}

void TubuleSystem::updateProteinMotion() {
    const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
    const double dt = rodSystem.runConfig.dt;
    const double KBT = proteinConfig.KBT;

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nProteinLocal; i++) {
            auto &protein = proteinContainer[i];
            if (protein.getWalkOrNot()) { // At least one head is bound
                protein.updatePosWalk(KBT, dt, rngPoolPtr->getU01(threadId), rngPoolPtr->getN01(threadId),
                                      rngPoolPtr->getN01(threadId));
            } else { // No heads are bound, diffuse in solution
                protein.updatePosDiffuse(dt, rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId),
                                         rngPoolPtr->getN01(threadId));
            }
        }
    }

    // process protein-boundary motion
    for (const auto &bPtr : rodSystem.runConfig.boundaryPtr) {
#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nProteinLocal; i++) {
                auto &protein = proteinContainer[i];
                if (!protein.getWalkOrNot()) {
                    // No heads are bound, diffuse in solution
                    const auto Query = protein.getPosPtr();
                    double Proj[3] = {0, 0, 0};
                    double delta[3] = {0, 0, 0};
                    bPtr->project(Query, Proj, delta);
                    if (Emap3(delta).dot(ECmap3(Query) - Emap3(Proj)) < 0) {
                        // protein outside of boundary
                        protein.setPos(Proj);
                    }
                }
            }
        }
    }
}

void TubuleSystem::outputProteinData() {
    if (rodSystem.getIfWriteResultCurrentStep()) {
        std::string baseFolder = rodSystem.getCurrentResultFolder();
        IOHelper::makeSubFolder(baseFolder);
        writeProteinAscii();
        writeProteinVTK();
    }
}

void TubuleSystem::writeProteinAscii() {
    // write a single ascii .dat file
    const int nGlobal = proteinContainer.getNumberOfParticleGlobal();
    auto snapID = rodSystem.getSnapID();
    std::string baseFolder = rodSystem.getCurrentResultFolder();
    std::string name = baseFolder + std::string("ProteinAscii_") + std::to_string(snapID) + ".dat";
    ProteinAsciiHeader header;
    header.nparticle = nGlobal;
    header.time = rodSystem.getStepCount() * rodSystem.runConfig.dt;
    proteinContainer.writeParticleAscii(name.c_str(), header);
    MPI_Barrier(MPI_COMM_WORLD);
}

void TubuleSystem::setInitialProteinFromConfig() {
    double boxEdge[3];
    auto initBoxLow = rodSystem.runConfig.initBoxLow;
    auto initBoxHigh = rodSystem.runConfig.initBoxHigh;
    for (int i = 0; i < 3; i++) {
        boxEdge[i] = initBoxHigh[i] - initBoxLow[i];
    }

    // x axis circular cross section
    Evec3 centerCrossSec = Evec3(0, (initBoxHigh[1] - initBoxLow[1]) * 0.5 + initBoxLow[1],
                                 (initBoxHigh[2] - initBoxLow[2]) * 0.5 + initBoxLow[2]);
    double radiusCrossSec = 0.5 * std::min(initBoxHigh[2] - initBoxLow[2], initBoxHigh[1] - initBoxLow[1]);

    const auto &tubuleContainer = rodSystem.getContainer();
    const int nTubuleLocal = tubuleContainer.getNumberOfParticleLocal();
    const int nTubuleGlobal = tubuleContainer.getNumberOfParticleGlobal();
    const auto &tubuleMap = getTMAPFromLocalSize(nTubuleLocal, rodSystem.getCommRcp());

    /**
     * Gid order:
     * 0 Tubule: [0,nTubuleGlobal - 1]
     * 1 Protein Type 1: [0, freeNumberType1 - 1] [0, nFixedPerMT1 *
     * (nTubuleGlobal-1)]
     * 2 Protein Type 2: [0, freeNumberType2 - 1] [0, nFixedPerMT2 *
     * (nTubuleGlobal-1)]
     * ....
     * All gids start from 0 for Tubules and each protein type
     */

    // free protein initialized on rank 0
    // fixedEnd0 protein initialized on all ranks
    const int nType = proteinConfig.types.size();
    if (rank == 0) {
        for (int iType = 0; iType < nType; iType++) {
            // free proteins initialized only on rank 0
            const int freeNumber = proteinConfig.freeNumber[iType];
            if (freeNumber > 0)
                spdlog::debug("initializing free proteins for tag {}", proteinConfig.types[iType].tag);
            for (int i = 0; i < freeNumber; i++) {
                ProteinData newProtein;
                newProtein.gid = i;
                newProtein.property = proteinConfig.types[iType];
                newProtein.property.LUTablePtr = &(LUTArr[iType]);
                newProtein.bind.clear();
                double pos[3] = {0, 0, 0};
                for (int k = 0; k < 3; k++) {
                    pos[k] = rngPoolPtr->getU01(0) * boxEdge[k] + initBoxLow[k];
                }
                if (rodSystem.runConfig.initCircularX) {
                    double y, z = 0;
                    getRandPointInCircle(radiusCrossSec, rngPoolPtr->getU01(0), rngPoolPtr->getU01(0), y, z);
                    pos[1] = y + centerCrossSec[1];
                    pos[2] = z + centerCrossSec[2];
                }
                newProtein.setPos(pos);
                newProtein.updateGeometryWithBind();
                proteinContainer.addOneParticle(newProtein);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // fixedEnd0 protein initialized on all ranks
    // this includes permanently fixed proteins and initially fixed
    // proteins
    for (int iType = 0; iType < nType; iType++) {
        const int nFixedPerMT = proteinConfig.fixedLocations[iType].size();
        const int gidFixedBase = tubuleMap->getMinGlobalIndex();
        if (nFixedPerMT > 0) {
            for (int t = 0; t < nTubuleLocal; t++) {
                for (int k = 0; k < nFixedPerMT; k++) {
                    ProteinData newProtein;
                    newProtein.property = proteinConfig.types[iType];
                    newProtein.bind.clear();

                    newProtein.gid = proteinConfig.freeNumber[iType] + nFixedPerMT * (gidFixedBase + t) + k;
                    auto &tubule = tubuleContainer[t];
                    newProtein.bind.indexBind[0] = tubule.globalIndex;
                    newProtein.bind.gidBind[0] = tubule.gid;
                    if (proteinConfig.fixedLocations[iType][k] > 1 || proteinConfig.fixedLocations[iType][k] < -1) {
                        // random location along MT
                        newProtein.bind.distBind[0] = (rngPoolPtr->getU01(0) - 0.5) * tubule.length;
                    } else {
                        std::cout << "fixed location: " << proteinConfig.fixedLocations[iType][k] << std::endl;
                        newProtein.bind.distBind[0] = proteinConfig.fixedLocations[iType][k] * tubule.length * 0.5;
                    }
                    newProtein.bind.lenBind[0] = tubule.length;
                    newProtein.bind.centerBind[0][0] = tubule.pos[0];
                    newProtein.bind.centerBind[0][1] = tubule.pos[1];
                    newProtein.bind.centerBind[0][2] = tubule.pos[2];
                    Evec3 direction = ECmapq(tubule.orientation) * Evec3(0, 0, 1);
                    newProtein.bind.directionBind[0][0] = direction[0];
                    newProtein.bind.directionBind[0][1] = direction[1];
                    newProtein.bind.directionBind[0][2] = direction[2];
                    newProtein.updateGeometryWithBind();
                    proteinContainer.addOneParticle(newProtein);
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());
    proteinContainer.exchangeParticle(rodSystem.getDomainInfoNonConst());
}

void TubuleSystem::updateBindWithGid(bool reconstruct) {
    // both tubule and protein are all distributed on all ranks
    // protein must have valid pos[], adjusted into the root domain
    // this is for reconstruction of bind information from initial
    // configuration newProtein.bind.posEndBind is valid

    // step 1  put data into tubuleDataDirectory
    const auto &tubuleContainer = rodSystem.getContainer();
    const int nTubuleLocal = tubuleContainer.getNumberOfParticleLocal();
    ZDD<SylinderNearEP> tubuleDataDirectory(nTubuleLocal);
    tubuleDataDirectory.gidOnLocal.resize(nTubuleLocal);
    tubuleDataDirectory.dataOnLocal.resize(nTubuleLocal);
#pragma omp parallel for
    for (int t = 0; t < nTubuleLocal; t++) {
        tubuleDataDirectory.gidOnLocal[t] = tubuleContainer[t].gid;
        tubuleDataDirectory.dataOnLocal[t].copyFromFP(tubuleContainer[t]);
    }
    tubuleDataDirectory.buildIndex();

    // step 2 put id to find. two ids per protein
    const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
    tubuleDataDirectory.gidToFind.resize(2 * nProteinLocal);
    tubuleDataDirectory.dataToFind.resize(2 * nProteinLocal);
#pragma omp parallel for
    for (int p = 0; p < nProteinLocal; p++) {
        // for gidBind = ID_UB, ZDD fills findData with invalid data.
        tubuleDataDirectory.gidToFind[2 * p + 0] = proteinContainer[p].bind.gidBind[0];
        tubuleDataDirectory.gidToFind[2 * p + 1] = proteinContainer[p].bind.gidBind[1];
    }
    tubuleDataDirectory.find();

    auto simBoxLow = rodSystem.runConfig.simBoxLow;
    auto simBoxHigh = rodSystem.runConfig.simBoxHigh;
    auto simBoxPBC = rodSystem.runConfig.simBoxPBC;

    // step 3 update
#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int p = 0; p < nProteinLocal; p++) {
            auto &protein = proteinContainer[p];

            // printf("%d,%lf,%lf,%lf\n", protein.gid, protein.bind.pos[0],
            // protein.bind.pos[1], protein.bind.pos[2]);

            for (int e = 0; e < 2; e++) { // check both ends
                if (protein.bind.gidBind[e] == ID_UB) {
                    protein.bind.setUnBind(e);
                } else { // get tubule data from ZDD
                    auto &tubuleBind = tubuleDataDirectory.dataToFind[2 * p + e];
                    if (protein.bind.gidBind[e] != tubuleBind.gid) {
                        printf("gid does not match\n");
                        std::exit(1);
                    }
                    // step 1 find actual bind PBC Image of tubule
                    for (int dim = 0; dim < 3; dim++) {
                        if (simBoxPBC[dim]) {
                            findPBCImage(simBoxLow[dim], simBoxHigh[dim], tubuleBind.pos[dim], protein.bind.pos[dim]);
                        }
                    }

                    // step 2 reconstruct binding information here
                    /***
                     * For each end:
                     * 1. update center/direction
                     * 2. if distBind not valid (setup initial data):
                     *      compute distBind with posEnd.
                     *      if this distBind is invalid (data error),
                     *      then generage random valid distBind
                     *    if distBind is valid (during simulation):
                     *      do nothing
                     *
                     * After both ends updated:
                     *    protein.calcPosEndWithDistBind()
                     *
                     * Result: updated data always consistent with
                     *  valid given posEnd or distBind
                     *
                     */
                    protein.bind.lenBind[e] = tubuleBind.length;
                    for (int dim = 0; dim < 3; dim++) {
                        protein.bind.directionBind[e][dim] = tubuleBind.direction[dim];
                        protein.bind.centerBind[e][dim] = tubuleBind.pos[dim];
                    }
                    if (reconstruct) {
                        // update distBind rebuild with posEndBind
                        Evec3 bindFoot = Emap3(protein.bind.posEndBind[e]);
                        double distBind = (bindFoot - Emap3(tubuleBind.pos)).dot(Emap3(tubuleBind.direction));
                        if (distBind > -tubuleBind.length * 0.5 && distBind < tubuleBind.length * 0.5) {
                            // Case 1, valid data
                            protein.bind.distBind[e] = distBind;
                        } else if ((distBind > -tubuleBind.length * 0.5 * 1.01 &&
                                    distBind < tubuleBind.length * 0.5 * 1.01) ||
                                   tubuleBind.isSphere()) {
                            // Case 2, valid data with some floating point error
                            protein.bind.distBind[e] = distBind;
                            protein.bind.updatePosEndClamp(e);
                        } else {
                            // Case 3, invalid data, set unbind
                            spdlog::error("posEnd {} invalid at distBind {} for "
                                          "protein {}",
                                          e, distBind, protein.gid);
                            spdlog::error("set end {} to unbind", e);
                            protein.bind.setUnBind(e);
                        }
                    }
                }
            }
            if (reconstruct)
                protein.updateGeometryWithBind();
        }
    }
    proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());

    if (reconstruct) {
        //buildLookupTable();
        setLookupTablePtr();
    }

#pragma omp parallel for
    for (int i = 0; i < nProteinLocal; i++) {
        // update protein force/torque of binding by actual updated binding position.
        auto &p = proteinContainer[i];
        p.updateForceTorqueBind();
    }
}

void TubuleSystem::buildLookupTable() {
    LUTArr.resize(proteinConfig.types.size());
    double D = rodSystem.runConfig.sylinderDiameter;
    for (int i = 0; i < proteinConfig.types.size(); i++) {
        LUTFiller *lut_filler_ptr = makeLUTFiller(proteinConfig.types[i]);
        LUTArr[i] = LookupTable(lut_filler_ptr, proteinConfig.types[i].useBindVol);
        if (proteinConfig.types[i].lookupType == 0) {
            proteinConfig.types[i].testKMCStepSize(rodSystem.runConfig.dt, &LUTArr[i]);
        }
        if (proteinConfig.types[i].useBindVol)
            spdlog::critical("bind volume: {}", LUTArr[i].getBindVolume());

        // New pointer was created in makeLUTFiller method. Clean it up.
        delete lut_filler_ptr;
    }
}

LUTFiller *TubuleSystem::makeLUTFiller(const ProteinType &ptype) {
    double D = rodSystem.runConfig.sylinderDiameter;
    const int grid_num = ptype.lookupGrid;
    switch (ptype.lookupType) {
    case 0: {
        // Energy dependent lookup table filler object
        LUTFillerEdep *lut_filler_ptr = new LUTFillerEdep(grid_num, grid_num);
        // Exponent pre-factor in Boltzmann factor of lookup table
        double M = .5 * (1. - ptype.lambda) * ptype.kappa / proteinConfig.KBT;
        // Add tubule diameter to freeLength to approximate binding to the
        // surface of sylinder instead of center.
        double ell0 = ptype.freeLength + D;
        lut_filler_ptr->Init(M, ell0, D);
        return lut_filler_ptr;
    }
    case 1: {
        LUTFiller2ndOrder *lut_filler_ptr = new LUTFiller2ndOrder(grid_num, grid_num);
        // Exponent pre-factor in Boltzmann factor of lookup table
        double M = .5 * (1. - ptype.lambda) * ptype.kappa / proteinConfig.KBT;
        lut_filler_ptr->Init(M, ptype.freeLength, D);
        return lut_filler_ptr;
    }
    default:
        return nullptr;
        break;
    }
}

void TubuleSystem::setLookupTablePtr() {
    // set LUT ptrs
    const int numProtein = proteinContainer.getNumberOfParticleLocal();
    const auto &tagLookup = proteinConfig.tagLookUp;
#pragma omp parallel for
    for (int i = 0; i < numProtein; i++) {
        auto &protein = proteinContainer[i];
        const int tag = protein.property.tag;
        const auto &index = tagLookup.find(tag);
        if (index == tagLookup.end()) {
            spdlog::critical("protein tagLookup error");
            std::exit(1);
        }
        protein.property.LUTablePtr = &(LUTArr[index->second]);
    }
}

void TubuleSystem::findTubuleRankWithGid() {
    // using ZDD to find distributed data
    // step 1  put data into tubuleDataDirectory
    const auto &tubuleContainer = rodSystem.getContainer();
    const int nTubuleLocal = tubuleContainer.getNumberOfParticleLocal();
    ZDD<int> tubuleDataDirectory(nTubuleLocal);
    tubuleDataDirectory.gidOnLocal.resize(nTubuleLocal);
    tubuleDataDirectory.dataOnLocal.resize(nTubuleLocal);
#pragma omp parallel for
    for (int t = 0; t < nTubuleLocal; t++) {
        tubuleDataDirectory.gidOnLocal[t] = tubuleContainer[t].gid;
        tubuleDataDirectory.dataOnLocal[t] = rank;
    }
    tubuleDataDirectory.buildIndex();

    // step 2 put id to find. two ids per protein
    const int nProteinLocal = proteinContainer.getNumberOfParticleLocal();
    tubuleDataDirectory.gidToFind.resize(2 * nProteinLocal);
    tubuleDataDirectory.dataToFind.resize(2 * nProteinLocal);
#pragma omp parallel for
    for (int p = 0; p < nProteinLocal; p++) {
        // for gidBind = ID_UB, ZDD fills findData with invalid data.
        tubuleDataDirectory.gidToFind[2 * p + 0] = proteinContainer[p].bind.gidBind[0];
        tubuleDataDirectory.gidToFind[2 * p + 1] = proteinContainer[p].bind.gidBind[1];
    }
    tubuleDataDirectory.find();

    // step 3 update
#pragma omp parallel for
    for (int p = 0; p < nProteinLocal; p++) {
        auto &protein = proteinContainer[p];
        for (int e = 0; e < 2; e++) {
            if (protein.bind.gidBind[e] != ID_UB) {
                protein.bind.rankBind[e] = tubuleDataDirectory.dataToFind[2 * p + e];
                assert(protein.bind.rankBind[e] >= 0 && protein.bind.rankBind[e] < nProcs);
            } else {
                protein.bind.rankBind[e] = -1;
            }
        }
    }
}

void TubuleSystem::setProteinConstraints() {
    const int nLocal = proteinContainer.getNumberOfParticleLocal();

    auto &conPool = rodSystem.getConstraintPoolNonConst();
    const int nThreads = conPool.size();
    const double tubuleDiameter = rodSystem.runConfig.sylinderDiameter;
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto &biQue = conPool[tid];
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            const auto &pr = proteinContainer[i];
            if (pr.getBindID(0) == ID_UB || pr.getBindID(1) == ID_UB) {
                // not doubly bound, not a constraint
                continue;
            }
            // geometry of MT I and J
            const Evec3 centerI = ECmap3(pr.bind.centerBind[0]);
            const Evec3 directionI = ECmap3(pr.bind.directionBind[0]);
            const Evec3 Ploc = ECmap3(pr.bind.posEndBind[0]);
            const Evec3 centerJ = ECmap3(pr.bind.centerBind[1]);
            const Evec3 directionJ = ECmap3(pr.bind.directionBind[1]);
            const Evec3 Qloc = ECmap3(pr.bind.posEndBind[1]);
            // information of constraint block
            const Evec3 PQvec = Qloc - Ploc;
            const double delta0 = pr.getProteinForceLength() - pr.property.freeLength;
            const double gamma = -delta0 * pr.property.kappa;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;
            biQue.emplace_back(delta0, gamma,                          // current separation, initial guess of gamma
                               pr.bind.gidBind[0], pr.bind.gidBind[1], //
                               pr.bind.indexBind[0],                   //
                               pr.bind.indexBind[1],                   //
                               normI.data(), normJ.data(),             // direction of constraint force
                               posI.data(),
                               posJ.data(),              // location relative to particle center
                               Ploc.data(), Qloc.data(), // location in lab frame
                               false, true, pr.property.kappa, 0.0);
            Emat3 stressIJ;
            CalcSylinderNearForce::collideStress(directionI, directionJ, centerI, centerJ, //
                                                 pr.bind.lenBind[0], pr.bind.lenBind[1], tubuleDiameter / 2,
                                                 tubuleDiameter / 2, 1.0, Ploc, Qloc, stressIJ);
            biQue.back().setStress(stressIJ);
        }
    }
}

void TubuleSystem::writeProteinVTK() {
    // write parallel XML VTK files from all ranks
    std::string baseFolder = rodSystem.getCurrentResultFolder();
    auto snapID = rodSystem.getSnapID();
    ProteinData::writeVTP<PS::ParticleSystem<ProteinData>>(
        proteinContainer, proteinContainer.getNumberOfParticleLocal(), baseFolder, std::to_string(snapID), rank);

    if (rank == 0) { // write parallel head
        ProteinData::writePVTP(baseFolder, std::to_string(snapID), nProcs);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void TubuleSystem::setInitialProteinFromFile(const std::string &posFilename) {
    // read file all to rank 0
    std::vector<ProteinData> proteinReadFromFile;
    if (rank == 0) {
        proteinConfig.echo();
        spdlog::warn("Read protein position from data file");
        std::ifstream myfile(posFilename);
        std::string line;
        std::getline(myfile, line); // read two header lines
        std::getline(myfile, line);

        while (std::getline(myfile, line)) {
            char typeChar;
            std::istringstream liness(line);
            liness >> typeChar;
            if (typeChar == 'P') {
                int gid, tag, gidBind[2];
                double end0[3];
                double end1[3];
                liness >> gid >> tag >> end0[0] >> end0[1] >> end0[2] >> end1[0] >> end1[1] >> end1[2] >> gidBind[0] >>
                    gidBind[1];
                int iType = proteinConfig.tagLookUp.find(tag)->second;
                ProteinData newProtein;
                newProtein.setFromFileInput(gid, tag, end0, end1, gidBind, gidBind, proteinConfig.types[iType]);
                proteinReadFromFile.push_back(newProtein);
                typeChar = 'N';
            }
        }
        myfile.close();
    } else { // other rank no protein in the beginning
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const int nProteinInFile = proteinReadFromFile.size();
    proteinContainer.setNumberOfParticleLocal(nProteinInFile);
#pragma omp parallel for
    for (int i = 0; i < nProteinInFile; i++) {
        proteinContainer[i] = proteinReadFromFile[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // protein.setPos() cannot be called before the bound MT data is updated
    // this must be called before adjustPosition and exchange
    updateBindWithGid(true);

    proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());
    proteinContainer.exchangeParticle(rodSystem.getDomainInfoNonConst());
}

void TubuleSystem::readProteinVTK(const std::string &pvtpFileName) {
    auto &commRcp = rodSystem.getCommRcp();
    if (commRcp->getRank() != 0) {
        proteinContainer.setNumberOfParticleLocal(0);
    } else {
        vtkSmartPointer<vtkXMLPPolyDataReader> reader = vtkSmartPointer<vtkXMLPPolyDataReader>::New();
        spdlog::warn("Reading " + pvtpFileName);
        reader->SetFileName(pvtpFileName.c_str());
        reader->Update();

        // Extract the polydata (At this point, the polydata is unsorted)
        vtkSmartPointer<vtkPolyData> polydata = reader->GetOutput();

        // Extract the point/cell data
        vtkSmartPointer<vtkPoints> posData = polydata->GetPoints();
        // cell data
        vtkSmartPointer<vtkTypeInt32Array> gidData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetCellData()->GetAbstractArray("gid"));
        vtkSmartPointer<vtkTypeInt32Array> tagData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetCellData()->GetAbstractArray("tag"));
        // point data
        vtkSmartPointer<vtkTypeInt32Array> idBindData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetPointData()->GetArray("gidBind"));

        // two points per protein
        const int proteinNumberInFile = posData->GetNumberOfPoints() / 2;
        std::vector<ProteinData> proteinReadFromFile(proteinNumberInFile);
        // set local
        proteinContainer.setNumberOfParticleLocal(proteinNumberInFile);
        spdlog::debug("Protein number in file: {}", proteinNumberInFile);
#pragma omp parallel for
        for (int i = 0; i < proteinNumberInFile; i++) {
            double end0[3] = {0, 0, 0};
            double end1[3] = {0, 0, 0};
            posData->GetPoint(i * 2, end0);
            posData->GetPoint(i * 2 + 1, end1);
            int gid = gidData->GetTypedComponent(i, 0);
            int tag = tagData->GetTypedComponent(i, 0);
            int gidBind[2];
            gidBind[0] = idBindData->GetTypedComponent(2 * i, 0);
            gidBind[1] = idBindData->GetTypedComponent(2 * i + 1, 0);
            auto &newProtein = proteinContainer[i];
            int iType = proteinConfig.tagLookUp.find(tag)->second;
            newProtein.setFromFileInput(gid, tag, end0, end1, gidBind, gidBind, proteinConfig.types[iType]);
            // std::cout << gid << " " << tag << " " << gidBind[0] << " "
            //           << gidBind[1] << " " << Emap3(end0).transpose() << " "
            //           << Emap3(end1).transpose() << std::endl;
        }
    }
    commRcp->barrier();
    // protein.setPos() cannot be called before the bound MT data is updated
    // this must be called before adjustPosition and exchange
    updateBindWithGid(true);

    proteinContainer.adjustPositionIntoRootDomain(rodSystem.getDomainInfo());
    proteinContainer.exchangeParticle(rodSystem.getDomainInfoNonConst());
}
