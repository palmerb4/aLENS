/* This test is designed to check the correctness of TubuleSystem::calcTubulinBindInteraction()*/

#include "TubuleSystem.hpp"
#include "GlobalTubulinPool.hpp"
#include <gtest/gtest.h>
#include <mpi.h>

// Global variables to store command-line arguments
int global_argc;
char **global_argv;

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1); // disable threading in eigen

    MPI_Init(&argc, &argv);
    PS::Initialize(argc, argv); // init FDPS system
    Logger::setup_mpi_spdlog();
    testing::InitGoogleTest(&argc, argv);

    // Store argc and argv in global variables after Google Test has been initialized
    global_argc = argc;
    global_argv = argv;
    int return_val = RUN_ALL_TESTS();

    PS::Finalize();
    MPI_Finalize();
    return return_val;
}

namespace {

//! \name Test test
//@{

TEST(AnEmptyTest, EmptyTest) {
    // This test is empty and is used to ensure that the test suite is running correctly.
    EXPECT_EQ(1, 1);
}
//@}

//! \name GlobalTubulinPool tests
//@{

TEST(GlobalTubulinPool, GlobalLocalInteractions) {
    int num_procs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int global_microtubule_count = 2 * num_procs;
    const int global_tubulin_count = 5 * num_procs;
    TRngPool rng_pool(0);

    GlobalTubulinPool myPool(global_microtubule_count, global_tubulin_count, num_procs, rank, &rng_pool);
    EXPECT_EQ(myPool.get_global_tubulin_count(), global_tubulin_count);
    EXPECT_EQ(myPool.get_local_tubulin_count(), 5);
    myPool.set_global_tubulin_count(6 * num_procs);
    EXPECT_EQ(myPool.get_global_tubulin_count(), 6 * num_procs);
    EXPECT_EQ(myPool.get_local_tubulin_count(), 6);
    myPool.set_global_microtubule_count(3 * num_procs);
    EXPECT_EQ(myPool.get_global_microtubule_count(), 3 * num_procs);

    myPool.increment();
    EXPECT_EQ(myPool.get_local_tubulin_count(), 7);
    EXPECT_EQ(myPool.get_global_tubulin_count(), 7 * num_procs) 
        << "Incrementing locally does not impact the global count until you synchronize, BUT getting the global count will synchronize";

    if (rank == 0) {
        for (int i = 0; i < 2 * num_procs; i++) {
            myPool.increment();
        }
        EXPECT_EQ(myPool.get_local_tubulin_count(), 7 + 2 * num_procs);
    } else {
        EXPECT_EQ(myPool.get_local_tubulin_count(), 7);
    }
    EXPECT_EQ(myPool.get_global_tubulin_count(), 7 * num_procs + 2 * num_procs);

    myPool.distribute();
    EXPECT_EQ(myPool.get_local_tubulin_count(), 9);
    EXPECT_EQ(myPool.get_global_tubulin_count(), 9 * num_procs); 
}
//@}

//! \name Implicit Tubulin
//@{

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, SingleUnbindEvent) {
    // 1 microtubule with no proteins and a large enough default tubulin unbind rate to force a tubulin to unbind.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/SingleUnbindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ImplicitTubulin/SingleUnbindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ImplicitTubulin/SingleUnbindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ImplicitTubulin/SingleUnbindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "Only one microtubule should be present";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";

    // Save some information for later
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin unbind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    EXPECT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 1) << "One unbound tubulin should be present";

    // Check the microtubule center and direction
    const Evec3 expectedPos =
        posOld - mySystem.proteinConfig.tubulinLength / 2.0 * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, SingleBindEvent) {
    // 1 microtubule and 1 tubulin, a large enough bind rate to force a tubulin to bind, and a zero unbind rate.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/SingleBindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ImplicitTubulin/SingleBindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ImplicitTubulin/SingleBindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ImplicitTubulin/SingleBindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);
    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 1) << "One unbound tubulin should be present";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "The microtubule should remain";
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength + mySystem.proteinConfig.tubulinLength, 1e-12)
        << "The microtubule length should increase by the tubulin length";

    // Check the microtubule center and direction
    const Evec3 expectedPos =
        posOld + mySystem.proteinConfig.tubulinLength / 2.0 * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, DoubleBindEvent) {
    // 1 microtubule and 2 tubulin, a near zero unbind rate, and a large enough bind rate to force both tubulin to bind.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/DoubleBindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ImplicitTubulin/DoubleBindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ImplicitTubulin/DoubleBindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ImplicitTubulin/DoubleBindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 2) << "Two unbound tubulin should be present";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    EXPECT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength + 2 * mySystem.proteinConfig.tubulinLength, 1e-12)
        << "The microtubule length should increase by two tubulin lengths";

    // Check the microtubule center and direction
    const Evec3 expectedPos =
        posOld + mySystem.proteinConfig.tubulinLength * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, DoubleBindUnbindEvent) {
    // 1 microtubule and 1 free tubulin, a large enough bind rate to force a tubulin to bind, and a large enough unbind rate to force a tubulin to unbind.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/DoubleBindUnbindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ImplicitTubulin/DoubleBindUnbindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ImplicitTubulin/DoubleBindUnbindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ImplicitTubulin/DoubleBindUnbindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 1) << "One unbound tubulin should be present";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    EXPECT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 1) << "One unbound tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength, 1e-12)
        << "The microtubule length should have increased and then decreased by the same amount";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld;
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, SingleUnBindEventWithProteinEnhancementAndPopoff) {
    // 1 microtubule with 1 protein near its end (within the protein enhancement cutoff distance). The default tubulin unbinding
    // rate should be zero and the enhanced rate should be large enough to force a tubulin to unbind. If the protein enhancement
    // functionality is correct, a tubulin should unbind. Furthermore, the protein should fall within one radius of the microtubule
    // end such that it will unbind during the timestep.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/RunConfig.yaml";
    std::string configFileProtein =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/ProteinConfig.yaml";
    std::string posFileTubule =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/TubuleInitial.dat";
    std::string posFileProtein =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // The protein should be within the enhancement cutoff radius and near the microtubule end
    const double proteinEnhancementCutoffDistance = mySystem.proteinConfig.proteinEnhancementCutoffDistance;
    const double distanceToMicrotubuleEnd =
        0.5 * proteinContainer[0].bind.lenBind[0] - proteinContainer[0].bind.distBind[0];
    ASSERT_LT(distanceToMicrotubuleEnd, proteinEnhancementCutoffDistance)
        << "The protein should be within the enhancement cutoff radius";
    ASSERT_LT(distanceToMicrotubuleEnd, sylinderContainer[0].radius)
        << "The protein should be within one radius of the microtubule end";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    EXPECT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 1) << "One unbound tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "The protein should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength - mySystem.proteinConfig.tubulinLength, 1e-12)
        << "The microtubule length should decrease by the tubulin length";

    // The protein should have unbound
    EXPECT_EQ(proteinContainer[0].bind.gidBind[0], ID_UB) << "The protein's left head should be unbound";
    EXPECT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // Check the microtubule center and direction
    const Evec3 expectedPos =
        posOld - mySystem.proteinConfig.tubulinLength / 2.0 * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, SingleUnBindEventWithProteinEnhancementAndNoPopoff) {
    // 1 microtubule with 1 protein near its end (within the protein enhancement cutoff distance). The default tubulin unbinding
    // rate should be zero and the enhanced rate should be large enough to force a tubulin to unbind. If the protein enhancement
    // functionality is correct, a tubulin should unbind. Furthermore, the protein should fall outside one radius of the microtubule
    // end such that it will NOT unbind during the timestep.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/RunConfig.yaml";
    std::string configFileProtein =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/ProteinConfig.yaml";
    std::string posFileTubule =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/TubuleInitial.dat";
    std::string posFileProtein =
        "TestData/ImplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // The protein should be within the enhancement cutoff radius and near the microtubule end
    const double proteinEnhancementCutoffDistance = mySystem.proteinConfig.proteinEnhancementCutoffDistance;
    const double distanceToMicrotubuleEnd =
        0.5 * proteinContainer[0].bind.lenBind[0] - proteinContainer[0].bind.distBind[0];
    ASSERT_LT(distanceToMicrotubuleEnd, proteinEnhancementCutoffDistance)
        << "The protein should be within the enhancement cutoff radius";
    ASSERT_GT(distanceToMicrotubuleEnd, sylinderContainer[0].radius)
        << "The protein should be outside one radius of the microtubule end";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    EXPECT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 1) << "One tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "The protein should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength - mySystem.proteinConfig.tubulinLength, 1e-12)
        << "The microtubule length should decrease by the tubulin length";

    // The protein should remain bound
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // Check the microtubule center and direction
    const Evec3 expectedPos =
        posOld - mySystem.proteinConfig.tubulinLength / 2.0 * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, NoEnhancement) {
    // 1 microtubule with 1 protein not near its end (outside the protein enhancement cutoff distance). The enhanced tubulin
    // unbinding rate should be large enough to force a tubulin to unbind and the default binding rate zero. If the protein enhancement
    // functionality is wrong, then a tubulin will erroneously unbind; otherwise, nothing will happen.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/NoEnhancement/RunConfig.yaml";
    std::string configFileProtein = "TestData/ImplicitTubulin/NoEnhancement/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ImplicitTubulin/NoEnhancement/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ImplicitTubulin/NoEnhancement/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // The protein should be outside the enhancement cutoff radius
    const double proteinEnhancementCutoffDistance = mySystem.proteinConfig.proteinEnhancementCutoffDistance;
    const double distanceToMicrotubuleEnd = proteinContainer[0].bind.lenBind[0] - proteinContainer[0].bind.distBind[0];
    ASSERT_GT(distanceToMicrotubuleEnd, proteinEnhancementCutoffDistance)
        << "The protein should be outside the enhancement cutoff radius";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "The protein should remain";
    EXPECT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength, 1e-12) << "The microtubule length should remain the same";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld;
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionImplicit, Depolymerization) {
    // N microtubules with a default tubulin unbinding rate large enough to force a tubulin to unbind from each microtubule each timestep.
    // We'll run time step function until the microtubules are fully depolymerized (length = tubulinLength).

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ImplicitTubulin/Depolymerization/RunConfig.yaml";
    std::string configFileProtein = "TestData/ImplicitTubulin/Depolymerization/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ImplicitTubulin/Depolymerization/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ImplicitTubulin/Depolymerization/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainerNonConst();
    auto &proteinContainer = mySystem.proteinContainer;
    auto &globalTubulinPoolPtr = mySystem.globalTubulinPoolPtr;
    const int nMicrotubules = sylinderContainer.getNumberOfParticleLocal();
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(globalTubulinPoolPtr->get_global_tubulin_count(), 0) << "No unbound tubulin should be present";
    for (int i = 0; i < nMicrotubules; i++) {
        ASSERT_NEAR(sylinderContainer[i].length, 1.0, 1e-12) << "The microtubule should have length 1.0";
        ASSERT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The microtubule should have radius 0.1";
    }

    // Running the step function should perform the tubulin bind event.
    // The microtubules are length 1 and radius 0.1, so we can unbind 4 tubulin per microtubule.
    // For the first 4 timesteps, each microtubule should lose 1 tubulin. After that, they should remain at length tubulinLength.

    // 4 unbinding timesteps and 6 steps where nothing should happen
    for (int i = 0; i < 10; i++) {
        ASSERT_NO_THROW(mySystem.step());

        const int currentNumParticles = sylinderContainer.getNumberOfParticleLocal();
        const int currentNumTubulin = globalTubulinPoolPtr->get_global_tubulin_count();
        ASSERT_EQ(currentNumParticles, nMicrotubules) << "The number of particles should remain the same";
        if (i < 4) {
            ASSERT_EQ(currentNumTubulin, nMicrotubules * (i + 1)) << "N*(i+1) tubulin should be present";
        } else {
            ASSERT_EQ(currentNumTubulin, nMicrotubules * 4) << "N*4 tubulin should be present";
        }
        for (int j = 0; j < currentNumParticles; j++) {
            if (i < 4) {
                EXPECT_NEAR(sylinderContainer[j].length, 1.0 - (i + 1) * mySystem.proteinConfig.tubulinLength, 1e-12)
                    << "The microtubule should have length 1.0 - (i+1)*tubulinLength";
            } else {
                EXPECT_NEAR(sylinderContainer[j].length, mySystem.proteinConfig.tubulinLength, 1e-12)
                    << "The microtubule should have length tubulinLength (fully depolymerized)";
            }
            EXPECT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The microtubule should have radius 0.1";
        }
    }
}
//@}

//! \name Explicit Tubulin
//@{

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, SingleUnbindEvent) {
    // 1 microtubule with no proteins and a large enough default tubulin unbind rate to force a tubulin to unbind.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/SingleUnbindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/SingleUnbindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/SingleUnbindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/SingleUnbindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "Only one microtubule should be present";
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";

    // Save some information for later
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin unbind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    const auto &microtubule = sylinderContainer[0];
    const auto &tubule = sylinderContainer[1];

    EXPECT_EQ(sylinderContainer.getNumberOfParticleLocal(), 2);
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    EXPECT_EQ(microtubule.group, 0) << "The microtubule should remain in the same group";
    EXPECT_EQ(tubule.group, 1) << "The new tubulin should be in the tubulin group";
    EXPECT_NEAR(tubule.length, 0.0, 1e-12) << "The tubulin should have length zero";

    // The tubulin should be within the binding cutoff radius
    Evec3 microtubuleEnd =
        ECmap3(microtubule.pos) + 0.5 * microtubule.length * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    double distranceFromTubuleCenterToMicrotubuleEnd = (microtubuleEnd - ECmap3(tubule.pos)).norm();
    EXPECT_LT(distranceFromTubuleCenterToMicrotubuleEnd, mySystem.proteinConfig.tubulinBindingCutoffRadius)
        << "The tubulin should be within the binding cutoff radius";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld - microtubule.radius * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, SingleBindEvent) {
    // 1 microtubule and 2 tubulin (one within the tubulin binding cutoff radius and another not),
    // a large enough bind rate to force a tubulin to bind, and a near zero unbind rate. If the cutoff functionality is wrong, both will bind.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/SingleBindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/SingleBindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/SingleBindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/SingleBindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);
    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 3) << "One microtubule and two tubulin should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";
    ASSERT_EQ(sylinderContainer[1].group, 1) << "The first tubulin should be in group 1";
    ASSERT_EQ(sylinderContainer[2].group, 1) << "The second tubulin should be in group 1";
    EXPECT_NEAR(sylinderContainer[1].length, 0.0, 1e-12) << "The tubulin should have length zero";
    EXPECT_NEAR(sylinderContainer[2].length, 0.0, 1e-12) << "The tubulin should have length zero";

    // The first tubulin should be within the binding cutoff radius and the second should not.
    const double tubulinBindingCutoffRadius = mySystem.proteinConfig.tubulinBindingCutoffRadius;
    Evec3 microtubuleEnd =
        ECmap3(sylinderContainer[0].pos) +
        0.5 * sylinderContainer[0].length * (ECmapq(sylinderContainer[0].orientation) * Evec3(0, 0, 1));
    const double distanceToMicrotubuleEnd1 = (ECmap3(sylinderContainer[1].pos) - microtubuleEnd).norm();
    const double distanceToMicrotubuleEnd2 = (ECmap3(sylinderContainer[2].pos) - microtubuleEnd).norm();
    ASSERT_LT(distanceToMicrotubuleEnd1, tubulinBindingCutoffRadius)
        << "The first tubulin should be within the binding cutoff radius";
    ASSERT_GT(distanceToMicrotubuleEnd2, tubulinBindingCutoffRadius)
        << "The second tubulin should not be within the binding cutoff radius";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const int tubulinWithinCutoffGID = sylinderContainer[1].gid;
    const int tubulinOutsideCutoffGID = sylinderContainer[2].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 2) << "The microtubule and one tubulin should remain";
    const auto &microtubule = sylinderContainer[0];
    const auto &tubuleOutsideCutoff = sylinderContainer[1];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(tubulinOutsideCutoffGID, tubuleOutsideCutoff.gid)
        << "The tubulin outside the cutoff radius should remain";
    EXPECT_EQ(tubuleOutsideCutoff.group, 1) << "The tubulin should remain a tubulin";
    EXPECT_EQ(microtubule.group, 0) << "The microtubule should remain a microtubule";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength + 2 * microtubule.radius, 1e-12)
        << "The microtubule length should increase by 2 radii";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld + microtubule.radius * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, DoubleBindEvent) {
    // 1 microtubule and 2 tubulin (both within the tubulin binding cutoff radius),
    // a near zero unbind rate, and a large enough bind rate to force a tubulin to bind.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/DoubleBindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/DoubleBindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/DoubleBindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/DoubleBindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 3) << "One microtubule and two tubulin should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";
    ASSERT_EQ(sylinderContainer[1].group, 1) << "The first tubulin should be in group 1";
    ASSERT_EQ(sylinderContainer[2].group, 1) << "The second tubulin should be in group 1";
    EXPECT_NEAR(sylinderContainer[1].length, 0.0, 1e-12) << "The tubulin should have length zero";
    EXPECT_NEAR(sylinderContainer[2].length, 0.0, 1e-12) << "The tubulin should have length zero";

    // Both tubulins should be within the binding cutoff radius
    const double tubulinBindingCutoffRadius = mySystem.proteinConfig.tubulinBindingCutoffRadius;
    Evec3 microtubuleEnd =
        ECmap3(sylinderContainer[0].pos) +
        0.5 * sylinderContainer[0].length * (ECmapq(sylinderContainer[0].orientation) * Evec3(0, 0, 1));
    const double distanceToMicrotubuleEnd1 = (ECmap3(sylinderContainer[1].pos) - microtubuleEnd).norm();
    const double distanceToMicrotubuleEnd2 = (ECmap3(sylinderContainer[2].pos) - microtubuleEnd).norm();
    ASSERT_LT(distanceToMicrotubuleEnd1, tubulinBindingCutoffRadius)
        << "The first tubulin should be within the binding cutoff radius";
    ASSERT_LT(distanceToMicrotubuleEnd2, tubulinBindingCutoffRadius)
        << "The second tubulin should be within the binding cutoff radius";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const int tubulinWithinCutoffGID1 = sylinderContainer[1].gid;
    const int tubulinWithinCutoffGID2 = sylinderContainer[2].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "The microtubule and no tubulin should remain";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength + 4 * microtubule.radius, 1e-12)
        << "The microtubule length should increase by 4 radii (2 for each tubulin)";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld + 2 * microtubule.radius * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, DoubleBindUnbindEvent) {
    // 1 microtubule and 1 tubulin within the tubulin binding cutoff radius,
    // a large enough bind rate to force a tubulin to bind, and a large enough unbind rate to force a tubulin to unbind.
    //
    // Note, even though we preform unbinding before binding, we do not consider newly added tubulin for binding. This prevents a tubulin
    // from unbinding and then immediately rebinding during the same time step.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/DoubleBindUnbindEvent/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/DoubleBindUnbindEvent/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/DoubleBindUnbindEvent/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/DoubleBindUnbindEvent/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 2) << "One microtubule and one tubulin should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";
    ASSERT_EQ(sylinderContainer[1].group, 1) << "The tubulin should be in group 1";
    EXPECT_NEAR(sylinderContainer[1].length, 0.0, 1e-12) << "The tubulin should have length zero";

    // The tubulin should be within the binding cutoff radius
    const double tubulinBindingCutoffRadius = mySystem.proteinConfig.tubulinBindingCutoffRadius;
    Evec3 microtubuleEnd =
        ECmap3(sylinderContainer[0].pos) +
        0.5 * sylinderContainer[0].length * (ECmapq(sylinderContainer[0].orientation) * Evec3(0, 0, 1));
    const double distanceToMicrotubuleEnd = (ECmap3(sylinderContainer[1].pos) - microtubuleEnd).norm();
    ASSERT_LT(distanceToMicrotubuleEnd, tubulinBindingCutoffRadius)
        << "The tubulin should be within the binding cutoff radius";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const int tubulinWithinCutoffGID = sylinderContainer[1].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);
    const Evec3 posTubulinOld = ECmap3(sylinderContainer[1].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 2) << "One microtubule and one tubulin should remain";
    const auto &microtubule = sylinderContainer[0];
    const auto &tubule = sylinderContainer[1];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";

    // The following commented out expectation fails. This indicates that FDPS reuses unused GIDs. That's not a bad thing,
    // just something to be aware of. Instead of comparing the previous and new GIDs, we can compare the positions of the tubulins.
    // EXPECT_NE(tubulinWithinCutoffGID, tubule.gid)
    //     << "The original tubulin should have bound and the new one unbound";
    EXPECT_NE(posTubulinOld, ECmap3(tubule.pos)) << "The original tubulin should have bound and the new one unbound";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld;
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, SingleUnBindEventWithProteinEnhancementAndPopoff) {
    // 1 microtubule with 1 protein near its end (within the protein enhancement cutoff distance). The default tubulin unbinding
    // rate should be zero and the enhanced rate should be large enough to force a tubulin to unbind. If the protein enhancement
    // functionality is correct, a tubulin should unbind. Furthermore, the protein should fall within one radius of the microtubule
    // end such that it will unbind during the timestep.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/RunConfig.yaml";
    std::string configFileProtein =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/ProteinConfig.yaml";
    std::string posFileTubule =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/TubuleInitial.dat";
    std::string posFileProtein =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndPopoff/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";
    EXPECT_NEAR(sylinderContainer[1].length, 0.0, 1e-12) << "The tubulin should have length zero";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // The protein should be within the enhancement cutoff radius and near the microtubule end
    const double proteinEnhancementCutoffDistance = mySystem.proteinConfig.proteinEnhancementCutoffDistance;
    const double distanceToMicrotubuleEnd =
        0.5 * proteinContainer[0].bind.lenBind[0] - proteinContainer[0].bind.distBind[0];
    ASSERT_LT(distanceToMicrotubuleEnd, proteinEnhancementCutoffDistance)
        << "The protein should be within the enhancement cutoff radius";
    ASSERT_LT(distanceToMicrotubuleEnd, sylinderContainer[0].radius)
        << "The protein should be within one radius of the microtubule end";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 2) << "One microtubule and one tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    const auto &tubule = sylinderContainer[1];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "The protein should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength - 2 * microtubule.radius, 1e-12)
        << "The microtubule length should decrease by 2 radii";
    EXPECT_NEAR(tubule.length, 0.0, 1e-12) << "The tubulin should have length zero";

    // The protein should have unbound
    EXPECT_EQ(proteinContainer[0].bind.gidBind[0], ID_UB) << "The protein's left head should be unbound";
    EXPECT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld - microtubule.radius * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, SingleUnBindEventWithProteinEnhancementAndNoPopoff) {
    // 1 microtubule with 1 protein near its end (within the protein enhancement cutoff distance). The default tubulin unbinding
    // rate should be zero and the enhanced rate should be large enough to force a tubulin to unbind. If the protein enhancement
    // functionality is correct, a tubulin should unbind. Furthermore, the protein should fall outside one radius of the microtubule
    // end such that it will NOT unbind during the timestep.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/RunConfig.yaml";
    std::string configFileProtein =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/ProteinConfig.yaml";
    std::string posFileTubule =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/TubuleInitial.dat";
    std::string posFileProtein =
        "TestData/ExplicitTubulin/SingleUnBindEventWithProteinEnhancementAndNoPopoff/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";
    EXPECT_NEAR(sylinderContainer[1].length, 0.0, 1e-12) << "The tubulin should have length zero";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // The protein should be within the enhancement cutoff radius and near the microtubule end
    const double proteinEnhancementCutoffDistance = mySystem.proteinConfig.proteinEnhancementCutoffDistance;
    const double distanceToMicrotubuleEnd =
        0.5 * proteinContainer[0].bind.lenBind[0] - proteinContainer[0].bind.distBind[0];
    ASSERT_LT(distanceToMicrotubuleEnd, proteinEnhancementCutoffDistance)
        << "The protein should be within the enhancement cutoff radius";
    ASSERT_GT(distanceToMicrotubuleEnd, sylinderContainer[0].radius)
        << "The protein should be outside one radius of the microtubule end";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 2) << "One microtubule and one tubulin should be present";
    const auto &microtubule = sylinderContainer[0];
    const auto &tubule = sylinderContainer[1];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "The protein should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength - 2 * microtubule.radius, 1e-12)
        << "The microtubule length should decrease by 2 radii";
    EXPECT_NEAR(tubule.length, 0.0, 1e-12) << "The tubulin should have length zero";

    // The protein should remain bound
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld - microtubule.radius * (ECmapq(microtubule.orientation) * Evec3(0, 0, 1));
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, NoEnhancement) {
    // 1 microtubule with 1 protein not near its end (outside the protein enhancement cutoff distance). The enhanced tubulin
    // unbinding rate should be large enough to force a tubulin to unbind and the default binding rate zero. If the protein enhancement
    // functionality is wrong, then a tubulin will erroneously unbind; otherwise, nothing will happen.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/NoEnhancement/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/NoEnhancement/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/NoEnhancement/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/NoEnhancement/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainer();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "One protein should be present";
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should be present";
    ASSERT_EQ(sylinderContainer[0].group, 0) << "The microtubule should be in group 0";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[0], sylinderContainer[0].gid)
        << "The protein's left head should be bound to the microtubule";
    ASSERT_EQ(proteinContainer[0].bind.gidBind[1], ID_UB) << "The protein's right head should be unbound";

    // The protein should be outside the enhancement cutoff radius
    const double proteinEnhancementCutoffDistance = mySystem.proteinConfig.proteinEnhancementCutoffDistance;
    const double distanceToMicrotubuleEnd = proteinContainer[0].bind.lenBind[0] - proteinContainer[0].bind.distBind[0];
    ASSERT_GT(distanceToMicrotubuleEnd, proteinEnhancementCutoffDistance)
        << "The protein should be outside the enhancement cutoff radius";

    // Save some information for later
    const int microtubuleGID = sylinderContainer[0].gid;
    const double oldMicrotubuleLength = sylinderContainer[0].length;
    const Equatn qOld = ECmapq(sylinderContainer[0].orientation);
    const Evec3 posOld = ECmap3(sylinderContainer[0].pos);

    // Running the step function should perform the tubulin bind event.
    ASSERT_NO_THROW(mySystem.step());

    // Validate the final configuration
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 1) << "One microtubule should remain";
    const auto &microtubule = sylinderContainer[0];
    EXPECT_EQ(microtubuleGID, microtubule.gid) << "The microtubule should remain";
    EXPECT_EQ(proteinContainer.getNumberOfParticleLocal(), 1) << "The protein should remain";
    EXPECT_NEAR(microtubule.length, oldMicrotubuleLength, 1e-12) << "The microtubule length should remain the same";

    // Check the microtubule center and direction
    const Evec3 expectedPos = posOld;
    EXPECT_NEAR(microtubule.pos[0], expectedPos[0], 1e-12);
    EXPECT_NEAR(microtubule.pos[1], expectedPos[1], 1e-12);
    EXPECT_NEAR(microtubule.pos[2], expectedPos[2], 1e-12);
    const Equatn qNew = ECmapq(microtubule.orientation);
    EXPECT_NEAR(qOld.x(), qNew.x(), 1e-12);
    EXPECT_NEAR(qOld.y(), qNew.y(), 1e-12);
    EXPECT_NEAR(qOld.z(), qNew.z(), 1e-12);
    EXPECT_NEAR(qOld.w(), qNew.w(), 1e-12);
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, Depolymerization) {
    // N microtubules with a default tubulin unbinding rate large enough to force a tubulin to unbind from each microtubule each timestep.
    // We'll run time step function until the microtubules are fully depolymerized (length = 0).

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/Depolymerization/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/Depolymerization/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/Depolymerization/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/Depolymerization/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainerNonConst();
    auto &proteinContainer = mySystem.proteinContainer;
    const int nMicrotubules = sylinderContainer.getNumberOfParticleLocal();
    ASSERT_EQ(proteinContainer.getNumberOfParticleLocal(), 0) << "No proteins should be present";
    for (int i = 0; i < nMicrotubules; i++) {
        ASSERT_NEAR(sylinderContainer[i].length, 1.0, 1e-12) << "The microtubule should have length 1.0";
        ASSERT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The microtubule should have radius 0.1";

        // There's no way to set group when creating microtubules using RunConfig's random initialization.
        // We'll set it here.
        sylinderContainer[i].group = 0;
    }

    // Running the step function should perform the tubulin bind event.
    // The microtubules are length 1 and radius 0.1, so we can unbind 4 tubulin per microtubule.
    // For the first 4 timesteps, each microtubule should lose 1 tubulin. After that, they should remain at length 0.

    // 4 unbinding timesteps and 6 steps where nothing should happen
    for (int i = 0; i < 10; i++) {
        ASSERT_NO_THROW(mySystem.step());

        const int currentNumParticles = sylinderContainer.getNumberOfParticleLocal();
        if (i < 4) {
            ASSERT_EQ(currentNumParticles, nMicrotubules + nMicrotubules * (i + 1))
                << "N microtubules + N*(i+1) tubulin should be present";
        } else {
            ASSERT_EQ(currentNumParticles, nMicrotubules + nMicrotubules * 4)
                << "N microtubules + N*4 tubulin should be present";
        }
        int numTubulin = 0;
        int numMicrotubule = 0;
        for (int j = 0; j < currentNumParticles; j++) {
            if (sylinderContainer[j].group == 0) {
                numMicrotubule++;
                if (i < 4) {
                    EXPECT_NEAR(sylinderContainer[j].length, 1.0 - 2 * (i + 1) * 0.1, 1e-12)
                        << "The microtubule should have length 1.0 - 2 * (i+1)*0.1";
                } else {
                    EXPECT_NEAR(sylinderContainer[j].length, 0.2, 1e-12)
                        << "The microtubule should have length 0.2 due to being fully depolymerized";
                }
                EXPECT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The microtubule should have radius 0.1";
            } else {
                numTubulin++;
                EXPECT_NEAR(sylinderContainer[j].length, 0.0, 1e-12) << "The tubulin should have length 0.0";
                EXPECT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The tubulin should have radius 0.1";
            }
        }
        if (i < 4) {
            ASSERT_EQ(numTubulin, nMicrotubules * (i + 1)) << "N*(i+1) tubulin should be present";
        } else {
            ASSERT_EQ(numTubulin, nMicrotubules * 4) << "N*4 tubulin should be present";
        }
        ASSERT_EQ(numMicrotubule, nMicrotubules) << "N microtubules should be present";
    }
}

TEST(TubuleSystemCalcTubulinBindInteractionExplicit, Polymerization) {
    // N microtubules with tubulin in their binding radius.

    // Initialize the TubuleSystem object with test-specific initialization and config files.
    std::string configFile = "TestData/ExplicitTubulin/Polymerization/RunConfig.yaml";
    std::string configFileProtein = "TestData/ExplicitTubulin/Polymerization/ProteinConfig.yaml";
    std::string posFileTubule = "TestData/ExplicitTubulin/Polymerization/TubuleInitial.dat";
    std::string posFileProtein = "TestData/ExplicitTubulin/Polymerization/ProteinInitial.dat";
    TubuleSystem mySystem(configFile, posFileTubule, configFileProtein, posFileProtein, global_argc, global_argv);

    // This test is made for one rank only
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs > 1) {
        return;
    }

    // Validate the initial configuration
    auto &sylinderContainer = mySystem.rodSystem.getContainerNonConst();
    auto &proteinContainer = mySystem.proteinContainer;
    ASSERT_EQ(sylinderContainer.getNumberOfParticleLocal(), 500) << "100 microtubules + 400 tubulin should be present";
    int countMicrotubules = 0;
    int countTubulin = 0;
    for (int i = 0; i < 500; i++) {
        if (sylinderContainer[i].group == 0) {
            countMicrotubules++;
            // We lose some precision from writing the system to file and reading it back in
            ASSERT_NEAR(sylinderContainer[i].length, 0.2, 1e-6)
                << "The microtubule should have length 0.2 due to being fully depolymerized";
            ASSERT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The microtubule should have radius 0.1";
        } else {
            countTubulin++;
            ASSERT_NEAR(sylinderContainer[i].length, 0.0, 1e-12) << "The tubulin should have length 0.0";
            ASSERT_NEAR(sylinderContainer[i].radius, 0.1, 1e-12) << "The tubulin should have radius 0.1";
        }
    }

    // Running the step function should perform the tubulin bind event.
    // TODO(palmerb4): I'm not sure what we can assert here since the tubulin binding is stochastic and it's not clear how many tubulin will bind at each step.
    for (int i = 0; i < 4; i++) {
        ASSERT_NO_THROW(mySystem.step());
    }
}
//@}

} // namespace
