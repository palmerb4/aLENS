# unit tests depend on GTest
find_package(GTest REQUIRED CONFIG PATHS ${SFTPATH} ${SFTPATH}/lib/cmake/GTest NO_DEFAULT_PATH)

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
  GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main  
)

add_test(NAME test_TubuleSystemCalcBindInteraction COMMAND test_TubuleSystemCalcBindInteraction)

# install the test files
file(COPY TestData DESTINATION ${CMAKE_BINARY_DIR}/SRC)
