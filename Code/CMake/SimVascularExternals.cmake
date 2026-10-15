#-----------------------------------------------------------------------------
# Setting up default directories for SimVascular Externals
# Note: *These can be changes by the user if they have moved the sv_extern directories*
# This is part of the legacy build system
find_package(Doxygen)
if(DOXYGEN_FOUND)
  file(TO_NATIVE_PATH "${SV_BINARY_DIR}/Doxygen/" SV_DOCS_DIR_WORK)
  set(SV_DOCS_DIR ${SV_DOCS_DIR_WORK} CACHE PATH "Location to place docs")
  configure_file(${SV_SOURCE_DIR}/../Documentation/Doxyfile.cmake.in
    ${SV_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    COMMAND "${CMAKE_COMMAND}" -E remove_directory
      "${SV_DOCS_DIR}"
    COMMAND "${DOXYGEN_EXECUTABLE}" "${SV_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${SV_SOURCE_DIR}/.."
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

#-----------------------------------------------------------------------------
# Enable Intel Runtime libs if we need or want them
if(SV_USE_INTEL)
  simvascular_external(INTELRUNTIME SYSTEM_DEFAULT SHARED_LIB)
else()
    unset(INTELRUNTIME_LIBRARIES CACHE)
endif()


#-----------------------------------------------------------------------------
# tkcximage (Legacy)
if(WIN32)
  if(SV_USE_TKCXIMAGE)
    find_library(TKCXIMAGE_DLL tkcximage)
    if(TKCXIMAGE_DLL)
      set(TKCXIMAGE_DLL_LIBRARY ${TKCXIMAGE_DLL})
      get_filename_component(TKCXIMAGE_DLL_PATH ${TKCXIMAGE_DLL} DIRECTORY CACHE)
      set(SV_EXTERNAL_SHARED_LIBS ${SV_EXTERNAL_SHARED_LIBS} "TKCXIMAGE")
    endif()
  endif()
endif()


#-----------------------------------------------------------------------------
# Enable MPI
if(SV_USE_MPI)
  if (NOT SV_USE_DUMMY_MPI)
    if(SV_USE_MSMPI)
      set(MPI_GUESS_LIBRARY_NAME MSMPI)
    endif()
    find_package(MPI REQUIRED COMPONENTS C CXX)
  else()
    set(SV_MPI_EXTRA_LIBRARY lib_extra_simvascular_dummympi)
  endif()
endif()
