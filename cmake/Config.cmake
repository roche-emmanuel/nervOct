SET(DEPS_DIR "W:/Cloud/Dev/Deps" CACHE STRING "Folder containing all the needed dependencies")
SET(TOOLS_DIR "W:/Cloud/Dev/Common/Tools/win32" CACHE STRING "Folder containing all the needed dev tools")
SET(CMAKE_INSTALL_PREFIX "W:/Cloud/Projects/mxsight/software" CACHE STRING "Installation folder" FORCE)

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
# SET(DEP_LUA ${DEPS_DIR}/${FLAVOR}/LuaJIT-2.0.1 CACHE STRING "lua path")

SET(DEP_DX ${DEPS_DIR}/${DEP_FLAVOR}/DXSDK-June2010 CACHE STRING "directx path")
SET(DEP_FUSION ${DEPS_DIR}/${DEP_FLAVOR}/VBSFusion-3.4 CACHE STRING "fusion path")

# SET_DEFAULT(DEP_LUNA sgtLuna-0.2.0)
# SET_DEFAULT(DEP_DX DXSDK-June2010)
# SET_DEFAULT(DEP_WX wxWidgets-2.9.3-static)
# SET_DEFAULT(DEP_OSG OpenSceneGraph-3.1.5)
# SET_DEFAULT(DEP_GEOLIB GeographicLib-1.21)
# SET_DEFAULT(DEP_NOISE libnoise-1.0.0)
# SET_DEFAULT(DEP_OPENCV OpenCV-2.4.0)
# SET_DEFAULT(DEP_KALMAN kalman-1.3)
# SET_DEFAULT(DEP_CRYPTOPP cryptopp-5.6.2)
# SET_DEFAULT(DEP_NVIDIASDISDK "NVIDIA Quadro SDI Video SDK for DirectX v0.9.5")

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

SET(FUSION_INC_DIR  ${DEP_FUSION}/includes/VBSFusion ${DEP_FUSION}/includes/VBSFusionDraw)
SET(FUSION_LIB_DIR 	${DEP_FUSION}/libs/${ARCHMODEL})
SET(FUSION_LIBS 	VBSFusion_2010 VBSFusionDraw_2010)
SET(FUSION_ENCRYPTER "${DEP_FUSION}/Tools/dllEncrypter/${ARCHMODEL}/Crypt_SIMCT.exe" CACHE STRING "Encrypter used to generate .fusion files" FORCE)

SET(DX_INC_DIR 	${DEP_DX}/Include)
SET(DX_LIB_DIR 	${DEP_DX}/Lib/${ARCHMODEL})
SET(DX_LIBS 	d3dx11.lib d3dx9.lib DxErr.lib)
