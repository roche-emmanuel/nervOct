SET(DEPS_DIR "W:/Cloud/Dev/Deps" CACHE STRING "Folder containing all the needed dependencies")
SET(LOCAL_DEPS_DIR "X:/Station" CACHE STRING "Folder containing the local dependencies")
SET(TOOLS_DIR "W:/Cloud/Dev/Common/Tools/win32" CACHE STRING "Folder containing all the needed dev tools")
SET(CMAKE_INSTALL_PREFIX "W:/Cloud/Projects/nervtech" CACHE STRING "Installation folder" FORCE)

SET(UPX_DIR "${TOOLS_DIR}/upx-3.91" CACHE STRING "Path where upx.exe can be found")
SET(DOT_DIR "${TOOLS_DIR}/GraphViz-2.28.0/bin" CACHE STRING "Path where dot.exe can be found")
SET(HHC_DIR "${TOOLS_DIR}/HtmlHelpWorkshop-1.3" CACHE STRING "Path where hhc.exe can be found")
SET(DOXYGEN "${TOOLS_DIR}/doxygen-1.8.0/doxygen.exe" CACHE STRING "Doxygen executable")
SET(LUA "${TOOLS_DIR}/luajit.exe" CACHE STRING "LUA executable")

SET(SGTLAUNCHER "${TOOLS_DIR}/sgtgen.exe" CACHE STRING "Singularity launcher")

IF("${FLAVOR}" STREQUAL "win64")
SET(DEP_FLAVOR "win32")
ELSE()
SET(DEP_FLAVOR "${FLAVOR}")
ENDIF()

SET(DEP_BOOST ${DEPS_DIR}/${DEP_FLAVOR}/boost-1.56.0 CACHE STRING "boost path")
SET(DEP_DX ${DEPS_DIR}/${DEP_FLAVOR}/DXSDK-June2010 CACHE STRING "directx path")
SET(DEP_CUDA ${LOCAL_DEPS_DIR}/CUDA_Toolkit-6.5 CACHE STRING "CUDA path")
SET(DEP_GPUMLIB ${LOCAL_DEPS_DIR}/GPUMLib-0.2.3 CACHE STRING "GPUMLib path")

# Include the macro definitions:
INCLUDE(cmake/Macros.cmake)

IF("${FLAVOR}" STREQUAL "win64")
	SET(ARCHMODEL "x64")
ELSE()
	SET(ARCHMODEL "x86")
ENDIF()

# Depdencies definitions:
IF(WIN32)
	SET(FLAVOR_LIBS rpcrt4 oleaut32 ole32 uuid winspool winmm shell32 comctl32 comdlg32 advapi32 ws2_32 wsock32 gdi32 wmvcore vfw32 strmiids)
	SET(GL_LIBS opengl32 GLU32)
	SET(IUP_FLAGS    -DIUP_DLL)
ELSE()
	SET(FLAVOR_LIBS dl)
	SET(GL_LIBS GL GLU)
ENDIF()

SET(BOOST_INC_DIR 	${DEP_BOOST}/include)
SET(BOOST_LIB_DIR 	${DEP_BOOST}/lib/${ARCHMODEL})
SET(BOOST_LIBS 		)

SET(CUDA_INC_DIR 	${DEP_CUDA}/include)
SET(CUDA_LIB_DIR 	${DEP_CUDA}/lib/${ARCHMODEL})
SET(CUDA_LIBS 		cudart.lib)

SET(GPUMLIB_INC_DIR 	${DEP_GPUMLIB}/src)
SET(GPUMLIB_LIB_DIR 	${DEP_GPUMLIB}/lib/${ARCHMODEL})
SET(GPUMLIB_LIBS 			) # GPUMLibMBP.lib

# SET(DX_INC_DIR 	${DEP_DX}/Include)
# SET(DX_LIB_DIR 	${DEP_DX}/Lib/${ARCHMODEL})
# SET(DX_LIBS 	d3dx11.lib d3dx9.lib DxErr.lib)
