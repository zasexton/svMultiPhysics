#-----------------------------------------------------------------------------
# Include Macros
#-----------------------------------------------------------------------------
include(InstallRequiredSystemLibraries)
include(CPackComponent)

#-----------------------------------------------------------------------------
# Install program
#-----------------------------------------------------------------------------
install(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
  DESTINATION ${SV_INSTALL_RUNTIME_DIR}
  COMPONENT CoreExecutables)

if(LINUX)
  set(SV_INSTALLER_TYPE DEB CACHE STRING "When distributing, specify the installer package type.")
    set_property(CACHE SV_INSTALLER_TYPE PROPERTY STRINGS DEB TGZ)
endif()

#-----------------------------------------------------------------------------
# Platform independent options
#-----------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "svMultiPhysics")
set(CPACK_PACKAGE_VENDOR "SimVascular.org")
set(CPACK_PACKAGE_CONTACT "SimVascular<sv@simvascular.org>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SimVascular Solvers")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/license.txt")
set(CPACK_PACKAGE_VERSION_MAJOR ${SV_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${SV_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${SV_PATCH_VERSION})
set(INSTALL_TYPE_STR)
string(REPLACE "." "-" SV_INSTALL_VERSION_DIR ${SV_FULL_VERSION})

#-----------------------------------------------------------------------------
# WINDOWS
#-----------------------------------------------------------------------------
if(WIN32)
  # Package name
   set(CPACK_PACKAGE_FILE_NAME "svMultiPhysics-windows-win-x64-${SV_FULL_VERSION}${INSTALL_TYPE_STR}")

  set(SV_INSTALLER_TYPE ZIP CACHE STRING "Windows package generator")
  set(CPACK_GENERATOR "${SV_INSTALLER_TYPE}")
  set(CPACK_PACKAGING_INSTALL_PREFIX "/")
  set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
  set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
  set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY ON)
  set(CPACK_COMPONENTS_ALL CoreExecutables)
  set(SV_PACKAGE_EXTRA_LIBRARIES "" CACHE STRING "Additional dynamically loaded DLLs to package")
  set(SV_PACKAGE_EXTRA_LICENSES "" CACHE STRING "License files for additional packaged libraries")
  get_filename_component(compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  get_filename_component(compiler_prefix "${compiler_bin}" DIRECTORY)
  set(SV_PACKAGE_RUNTIME_DIRS "${compiler_bin}" CACHE STRING "Additional DLL search directories")

  # Run during both installation and CPack staging, after the binary is built.
  set(install_windows_runtime [==[
    if("@MINGW@")
      set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "windows+pe")
      set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "objdump")
      set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "@CMAKE_OBJDUMP@")
    endif()
    set(extra_libraries "@SV_PACKAGE_EXTRA_LIBRARIES@")
    set(runtime_dirs "@SV_PACKAGE_RUNTIME_DIRS@")
    file(GET_RUNTIME_DEPENDENCIES
      EXECUTABLES "$<TARGET_FILE:svmultiphysics>"
      LIBRARIES ${extra_libraries}
      DIRECTORIES ${runtime_dirs}
      PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*" "[Mm][Ss][Mm][Pp][Ii]\\.[Dd][Ll][Ll]"
      POST_EXCLUDE_REGEXES [=[[/\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\]]=]
      RESOLVED_DEPENDENCIES_VAR dependencies
      UNRESOLVED_DEPENDENCIES_VAR unresolved
      CONFLICTING_DEPENDENCIES_PREFIX conflicts)
    if(unresolved OR conflicts_FILENAMES)
      message(FATAL_ERROR "Unresolved DLLs: ${unresolved}; conflicting DLLs: ${conflicts_FILENAMES}")
    endif()
    if(dependencies OR extra_libraries)
      file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/@SV_INSTALL_RUNTIME_DIR@"
        TYPE SHARED_LIBRARY FILES ${dependencies} ${extra_libraries})
    endif()
  ]==])
  string(CONFIGURE "${install_windows_runtime}" install_windows_runtime @ONLY)
  install(CODE "${install_windows_runtime}" COMPONENT CoreExecutables)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/license.txt" "${SV_SOURCE_DIR}/../README.md"
    DESTINATION share/svMultiPhysics COMPONENT CoreExecutables)
  install(FILES ${SV_PACKAGE_EXTRA_LICENSES}
    DESTINATION share/svMultiPhysics/licenses/extra COMPONENT CoreExecutables)
  install(FILES "${SV_BINARY_DIR}/dependencies.txt" "${SV_BINARY_DIR}/toolchain.txt"
    "${SV_BINARY_DIR}/source-revision.txt" "${SV_BINARY_DIR}/package-sha256.txt"
    DESTINATION share/svMultiPhysics COMPONENT CoreExecutables OPTIONAL)
  if(MINGW)
    install(DIRECTORY "${compiler_prefix}/share/licenses/"
      DESTINATION share/svMultiPhysics/licenses COMPONENT CoreExecutables)
  endif()
  foreach(dependency metis parmetis)
    install(FILES "${SV_SOURCE_DIR}/ThirdParty/${dependency}_internal/simvascular_${dependency}_internal/LICENSE.txt"
      DESTINATION "share/svMultiPhysics/licenses/${dependency}" COMPONENT CoreExecutables)
  endforeach()
  install(FILES "${SV_SOURCE_DIR}/ThirdParty/tetgen/simvascular_tetgen/LICENSE"
    DESTINATION share/svMultiPhysics/licenses/tetgen COMPONENT CoreExecutables)

  # Visual studios
  if(MSVC)
    set(CPACK_GENERATOR "WIX")
    set(CPACK_WIX_EXTENSIONS WixUtilExtension)
    set(CPACK_WIX_UI_BANNER ${CMAKE_CURRENT_SOURCE_DIR}/windows/msi-banner.jpg)
    set(CPACK_WIX_UI_DIALOG ${CMAKE_CURRENT_SOURCE_DIR}/windows/msi-dialog.jpg)
    if(SV_RELEASE_TYPE MATCHES "Release")
      set(CPACK_PACKAGE_INSTALL_DIRECTORY "SimVascular/Release")
      set(CPACK_WIX_PROGRAM_MENU_FOLDER "SimVascular")
      set(CPACK_WIX_PRODUCT_GUID "39068F8A-D4E9-405F-B22B-9B417849AEE6")
      set(CPACK_WIX_UPGRADE_GUID "2359EF69-C03F-401C-9BC3-C19B10297E3B")
      set(PACKAGE_NAME "svMultiPhysics")
    elseif(SV_RELEASE_TYPE MATCHES "Beta")
      set(CPACK_WIX_PROGRAM_MENU_FOLDER "SimVascular-Beta")
      set(PACKAGE_NAME "svMultiPhysics-beta")
      set(CPACK_PACKAGE_INSTALL_DIRECTORY "SimVascular/${TIMESTAMP}")
      set(CPACK_WIX_PRODUCT_GUID "9453BB7E-CD7E-46ED-AA9E-5514B56FC901")
      set(CPACK_WIX_UPGRADE_GUID "39C845FF-68E1-4AA7-B329-3407B47B3361")
    endif()
    set(CPACK_WIX_PATCH_FILE "${TEMP_DIR}/patch.xml")
    set(CPACK_WIX_PRODUCT_ICON "${CMAKE_CURRENT_SOURCE_DIR}/windows/SimVascular.ico")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/windows/mywix.template-notargets.in" "${CMAKE_CURRENT_BINARY_DIR}/mywix.template")
    set(CPACK_WIX_TEMPLATE "${CMAKE_CURRENT_BINARY_DIR}/mywix.template")
  endif()
endif()

#-----------------------------------------------------------------------------
# LINUX
#-----------------------------------------------------------------------------
if(LINUX)
  # Package name
  set(CPACK_PACKAGE_FILE_NAME "svMultiPhysics-linux-x64-${SV_FULL_VERSION}${INSTALL_TYPE_STR}")

    if(SV_INSTALLER_TYPE STREQUAL DEB)
        set(CPACK_GENERATOR "DEB")
        set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/sv/svMultiPhysics/${SV_INSTALL_VERSION_DIR}")
    elseif(SV_INSTALLER_TYPE STREQUAL TGZ)
      set(CPACK_GENERATOR "TGZ")
      set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
    endif()
endif()

#-----------------------------------------------------------------------------
# APPLE
#-----------------------------------------------------------------------------
if(APPLE)
  # Package name
  set(CPACK_PACKAGE_FILE_NAME "svMultiPhysics-darwin-x64-${SV_FULL_VERSION}${INSTALL_TYPE_STR}")

  # Components. Just install the svMultiPhysics executable.
  set(CPACK_COMPONENTS_ALL CoreExecutables) 

  # Cpack install help
  set(CPACK_GENERATOR "productbuild")
  #configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OSX/Info.plist.in
  #  ${CMAKE_CURRENT_BINARY_DIR}/OSX/Info.plist @ONLY)

  # Install location
  set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/sv/svMultiPhysics/${SV_INSTALL_VERSION_DIR}")
endif()

if(CPACK_PACKAGE_FILENAME_TAG)
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-${CPACK_PACKAGE_FILENAME_TAG}")
endif()

message(STATUS "Packname name: ${CPACK_PACKAGE_FILE_NAME}")
include(CPack)
