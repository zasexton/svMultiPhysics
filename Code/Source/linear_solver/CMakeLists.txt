
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(lib ${SV_LIB_LINEAR_SOLVER_NAME})

set(CSRCS 
  add_bc_mul.h add_bc_mul.cpp
  bcast.h bcast.cpp
  bc.cpp
  bicgs.h bicgs.cpp
  commu.h commu.cpp
  cgrad.h cgrad.cpp
  cput.cpp
  dot.h dot.cpp
  fsils.hpp
  fsils_api.hpp
  fils_struct.hpp fils_struct.cpp
  ge.h ge.cpp
  gmres.h gmres.cpp
  in_commu.cpp
  ls.cpp
  lhs.h lhs.cpp
  norm.h norm.cpp
  ns_solver.h ns_solver.cpp
  omp_la.h omp_la.cpp
  pc_gmres.h pc_gmres.cpp
  precond.h precond.cpp
  solve.cpp
  spar_mul.h spar_mul.cpp
)

add_library(${lib} ${SV_LIBRARY_TYPE} ${CSRCS})
target_include_directories(${lib} PRIVATE ${SV_SOURCE_DIR}/ThirdParty/eigen/include/eigen3)

target_link_libraries(${lib} PUBLIC MPI::MPI_CXX)

if(SV_INSTALL_LIBS)
  install(TARGETS ${lib}
    RUNTIME DESTINATION ${SV_INSTALL_RUNTIME_DIR} COMPONENT CoreExecutables
    LIBRARY DESTINATION ${SV_INSTALL_LIBRARY_DIR} COMPONENT CoreLibraries
    ARCHIVE DESTINATION ${SV_INSTALL_ARCHIVE_DIR} COMPONENT CoreLibraries
    )
endif()
if(SV_INSTALL_HEADERS)
  install(FILES ${HDRS}
    DESTINATION ${SV_INSTALL_INCLUDE_DIR}/core COMPONENT CoreHeaders
    )
endif()
if(ENABLE_COVERAGE)
  # set compiler flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -coverage")
endif()
