# unit tests depend on GTest
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG        release-1.12.1
# )

# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
  CMAKE_ARGS     -Dgtest_force_shared_crt=ON
  PREFIX         ${CMAKE_BINARY_DIR}/external/googletest
  INSTALL_COMMAND ""
)

# Assuming the structure matches what ExternalProject_Add creates
set(GTEST_ROOT_DIR "${CMAKE_BINARY_DIR}/external/googletest")
set(GTEST_INCLUDE_DIR "${GTEST_ROOT_DIR}/src/googletest/include")
# This path might need adjustment based on the actual output of the build
set(GTEST_LIB_DIR "${GTEST_ROOT_DIR}/src/googletest-build/lib")

add_executable(test_TubuleSystemCalcBindInteraction 
  TubuleSystemCalcBindInteraction_test.cpp
  ${SRC}/TubuleSystem.cpp
  ${PROTEIN}/ProteinConfig.cpp
  ${SIMTOOLBOX}/Boundary/Boundary.cpp
  ${SIMTOOLBOX}/Constraint/BCQPSolver.cpp
  ${SIMTOOLBOX}/Constraint/ConstraintCollector.cpp
  ${SIMTOOLBOX}/Constraint/ConstraintOperator.cpp
  ${SIMTOOLBOX}/Constraint/ConstraintSolver.cpp
  ${SIMTOOLBOX}/Sylinder/Sylinder.cpp
  ${SIMTOOLBOX}/Sylinder/SylinderConfig.cpp
  ${SIMTOOLBOX}/Sylinder/SylinderSystem.cpp
  ${SIMTOOLBOX}/Trilinos/TpetraUtil.cpp
  ${SIMTOOLBOX}/Util/Base64.cpp
)

target_include_directories(test_TubuleSystemCalcBindInteraction PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${SIMTOOLBOX}
  ${KMC}
  ${EIGEN3_INCLUDE_DIR}
  ${Trilinos_INCLUDE_DIRS}
  ${TRNG_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIR}
)

target_link_libraries(test_TubuleSystemCalcBindInteraction PRIVATE
${Trilinos_LIBRARIES}
${Trilinos_TPL_LIBRARIES}
${TRNG_LIBRARY}
  trng4
  yaml-cpp
  VTK::IOXML
  Boost::boost
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  ${GTEST_LIB_DIR}/libgtest.a
  ${GTEST_LIB_DIR}/libgtest_main.a
  ${GTEST_LIB_DIR}/libgmock.a
  ${GTEST_LIB_DIR}/libgmock_main.a
)

add_test(NAME test_TubuleSystemCalcBindInteraction COMMAND test_TubuleSystemCalcBindInteraction)

# install the test files
file(COPY TestData DESTINATION ${CMAKE_BINARY_DIR}/SRC)
