MACRO(ADD_MSVC_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
  IF(MSVC)
	file(GLOB_RECURSE to_remove "*${PrecompiledSource}")
	list(REMOVE_ITEM ${SourcesVar} ${to_remove})

    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
    SET(PrecompiledBinary "${CMAKE_CURRENT_BINARY_DIR}/${PrecompiledBasename}.pch")
    SET(Sources ${${SourcesVar}})

    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
    SET_SOURCE_FILES_PROPERTIES(${Sources}
                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledBinary}\" /FI\"${PrecompiledBinary}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_DEPENDS "${PrecompiledBinary}")  
    # Add precompiled header to SourcesVar
    LIST(APPEND ${SourcesVar} ${PrecompiledSource})
  ENDIF(MSVC)
ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)

MACRO(GENERATE_FUSION3_PLUGIN)
  IF(ENCRYPT_PLUGIN)
		# MESSAGE("Adding encryption for ${TARGET_NAME}")		
		get_property(lib_location TARGET ${TARGET_NAME} PROPERTY LOCATION)
		SET(targetPath "${lib_location}")
		# MESSAGE("Target location is: ${targetPath}")
		# MESSAGE("Target output is: ${PROJECT_SOURCE_DIR}/plugins/${TARGET_NAME}.fusion")
		
    ADD_CUSTOM_TARGET(
      ${TARGET_NAME}_encrypted ALL
      DEPENDS ${TARGET_NAME}
			# OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.fusion"
			# OUTPUT  "${PROJECT_SOURCE_DIR}/plugins/${TARGET_NAME}.fusion"
			# COMMAND echo "Target file is: ${targetPath}, output is: ${PROJECT_SOURCE_DIR}/plugins/${TARGET_NAME}.fusion"
      COMMAND ${FUSION_ENCRYPTER} ${targetPath} "${PROJECT_SOURCE_DIR}/software/VBS/plugins/${TARGET_NAME}.fusion"
      # COMMAND ${FUSION_ENCRYPTER} ${targetPath} "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.fusion"
      # DEPENDS ${targetPath}
		)
  ENDIF()
ENDMACRO(GENERATE_FUSION3_PLUGIN)

MACRO(GENERATE_REFLECTION STUB_NAME INTERFACE_FILES)
    IF(USE_SGT)
		SET(DOXFILE "doxyfile")
		
		SET(SGT_PATH  "${SGT2_DIR}/bin/win32") # needed to find some dependencies.
		
		STRING(REPLACE "/" "\\" DOX_TEMPLATE  "${PROJECT_SOURCE_DIR}/scripts/gen_Doxyfile_template")
		STRING(REPLACE "/" "\\" SGT_PATH  "${SGT_DIR}/software/bin/win32")
		SET(CAT_EXEC type)
		
		SET(CFGFILE generate.lua)
			
		ADD_CUSTOM_TARGET(
			${TARGET_NAME}_gen
			# PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../include/luna
			COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../src/luna
			COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../include/luna/wrappers
			# COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../src/luna/wrappers
			COMMAND echo "Generating doxygen wrapper docs..."
			COMMAND echo INPUT=${${INTERFACE_FILES}} > ${DOXFILE}
			COMMAND echo FILE_PATTERNS=${FILE_PATTERNS} >> ${DOXFILE}
			COMMAND echo EXPAND_AS_DEFINED=${EXPAND_AS_DEFINED} >> ${DOXFILE}
			COMMAND echo PREDEFINED=_DOXYGEN=1 __DOXYGEN__=1 ${DOXY_PREDEFINED} >> ${DOXFILE}
			COMMAND echo INCLUDE_PATH=${${INTERFACE_FILES}} ${INCLUDE_PATH} >> ${DOXFILE}
			COMMAND echo INCLUDE_FILE_PATTERNS= >> ${DOXFILE}
			COMMAND echo EXCLUDE_PATTERNS= >> ${DOXFILE}
			COMMAND echo DOT_PATH=${DOT_DIR} >> ${DOXFILE}
			COMMAND ${CAT_EXEC} "${DOX_TEMPLATE}" >> ${DOXFILE}
			# Call doxygen on this file:
			COMMAND ${DOXYGEN} ${DOXFILE} > ${CMAKE_CURRENT_BINARY_DIR}/doxygen.log 2>&1
			# COMMAND ${DOXYGEN} ${DOXFILE}
			COMMAND echo "Generating lua reflection..."
			# cd ${SGT_PATH} && 
			COMMAND echo "project='${TARGET_NAME}'" > ${CFGFILE}
			COMMAND echo "sgt_path='${SGT2_DIR}/'" >> ${CFGFILE}
			COMMAND echo "vbssim_path='${PROJECT_SOURCE_DIR}/'" >> ${CFGFILE}
			COMMAND echo "xml_path='${CMAKE_CURRENT_BINARY_DIR}/xml/'" >> ${CFGFILE}
			COMMAND echo "dofile('${CMAKE_CURRENT_SOURCE_DIR}/../generate_reflection.lua');" >> ${CFGFILE}
			
			COMMAND echo "${SGTLAUNCHER} ${CFGFILE} --log sgt_reflection.log"
			COMMAND ${SGTLAUNCHER} ${CFGFILE} --log sgt_reflection.log > temp_log_file.log
			
			# COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt # touch the calling file.
			# COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt # touch the calling file.
			COMMAND echo "Reflection generation done."
		)	
		
		ADD_DEPENDENCIES(${STUB_NAME} ${TARGET_NAME}_gen)
	ENDIF()
ENDMACRO(GENERATE_REFLECTION2)

MACRO(SET_SOFT_DEPENDENCIES STUB_NAME deps)
foreach(dep IN LISTS ${deps})
	# MESSAGE("Checking dependency on ${dep} for ${STUB_NAME}")
	IF(TARGET ${dep})
		MESSAGE("Adding dependency on ${dep} for ${STUB_NAME}")
		ADD_DEPENDENCIES(${STUB_NAME} ${dep})
	ENDIF()
endforeach()

ENDMACRO(SET_SOFT_DEPENDENCIES)

MACRO(GENERATE_LUA_PACKAGE sourceVar)
	# Check if a source path is provided:
	SET(srcFolder "${ARGV1}")
	SET(destFolder "${CMAKE_CURRENT_BINARY_DIR}/")
	SET(destFile "${ARGV2}")
	
	IF("${srcFolder}" STREQUAL "")
		SET(srcFolder "${CMAKE_CURRENT_SOURCE_DIR}/../modules/")
	ENDIF()
	IF("${destFile}" STREQUAL "")
		SET(destFile "bindings.cpp")
	ENDIF()

	# write the complete file name:
	SET(destFile "${destFolder}${destFile}")
	
	# Add generated file to source list:
	LIST(APPEND ${sourceVar} ${destFile})
	
	# look for all the dependencies
	FILE(GLOB_RECURSE DEP_FILES "${srcFolder}*.lua" "${srcFolder}*.ttf" "${srcFolder}*.png" "${srcFolder}*.dll" "${srcFolder}*.hlsl" ) 
	# MESSAGE("Found dependencies: ${DEP_FILES}")
	
	ADD_CUSTOM_COMMAND(OUTPUT ${destFile}
		COMMAND echo "Generating lua package..."
		# COMMAND echo "Dep files: ${DEP_FILES}"
		COMMAND ${LUA} -e "project='${TARGET_NAME}'; vbssim_path='${PROJECT_SOURCE_DIR}/'; src_path='${srcFolder}'; dest_path='${destFolder}';" ${PROJECT_SOURCE_DIR}/scripts/generate_package.lua
		# COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_SOURCE_DIR}/cmake/Macros.cmake # touch the calling file
		DEPENDS ${DEP_FILES})
		
		
	# ADD_CUSTOM_TARGET(
		# ${TARGET_NAME}_package
		# COMMAND echo "Generating lua package MSVC..."
		# COMMAND ${LUA} -e "project='${TARGET_NAME}'; vbssim_path='${PROJECT_SOURCE_DIR}/'; src_path='${srcFolder}'; dest_path='${destFolder}';" ${PROJECT_SOURCE_DIR}/scripts/generate_package.lua
		# COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt # touch the calling file.
	# )
	
	# ADD_DEPENDENCIES(${STUB_NAME} ${TARGET_NAME}_package)
ENDMACRO(GENERATE_LUA_PACKAGE)

MACRO(COMPRESS_BINARY_TARGET)
	SET(THE_TARGET "${TARGET_NAME}")
	MESSAGE("Adding compression for ${THE_TARGET}")
	
	ADD_CUSTOM_COMMAND(
		TARGET ${THE_TARGET}
		POST_BUILD
		COMMAND echo "Compressing ${THE_TARGET}..."
		COMMAND ${UPX_DIR}/upx.exe --best "$<TARGET_FILE:${THE_TARGET}>"
		COMMAND echo "Compression done."
	)		
ENDMACRO(COMPRESS_BINARY_TARGET)

MACRO(GENERATE_LIBRARY_IMAGE dep libName libFile sourceVar)
	SET(destFile "${CMAKE_CURRENT_BINARY_DIR}/${libName}.cpp")

	LIST(APPEND ${sourceVar} ${destFile})
	
	# MESSAGE("Current target name is: ${TARGET_NAME}, dep is: ${dep}")
	
	SET(targetPath "${PROJECT_SOURCE_DIR}/${libFile}")
	SET(depList "")
	
	# if a dependency is provided then we use it directly as source library file (when the target is available):
	IF(TARGET ${dep})
		get_property(lib_location TARGET ${dep} PROPERTY LOCATION)
		SET(targetPath "${lib_location}")
		SET(depList "${dep}")
		IF("${targetPath}" STREQUAL "")
			# restore library file name in that case:
			SET(targetPath "${PROJECT_SOURCE_DIR}/${libFile}")
		ENDIF()
		# MESSAGE("Target location is: ${targetPath}")
	ENDIF()
	
	ADD_CUSTOM_COMMAND(OUTPUT ${destFile}
		COMMAND echo "Generating library image ${libName}..."
		COMMAND ${LUA} -e "libFile='${targetPath}'; libName='${libName}'; destFile='${destFile}';" ${PROJECT_SOURCE_DIR}/scripts/generate_library_image.lua
		DEPENDS ${targetPath})
	
	SET_SOFT_DEPENDENCIES(${TARGET_NAME}_images depList)
ENDMACRO(GENERATE_LUA_PACKAGE)

MACRO(GENERATE_EXT_LIBRARY_IMAGE libName libFile sourceVar)
	SET(destFile "${CMAKE_CURRENT_BINARY_DIR}/${libName}.cpp")

	LIST(APPEND ${sourceVar} ${destFile})
	
	SET(targetPath "${PROJECT_SOURCE_DIR}/${libFile}")
	
	ADD_CUSTOM_COMMAND(OUTPUT ${destFile}
		COMMAND echo "Generating library ${libName} image..."
		COMMAND ${LUA} -e "libFile='${targetPath}'; libName='${libName}'; destFile='${destFile}';" ${PROJECT_SOURCE_DIR}/scripts/generate_library_image.lua
		DEPENDS ${targetPath})
ENDMACRO(GENERATE_LUA_PACKAGE)

MACRO(ADD_FILES file_list regex)
    FILE(GLOB_RECURSE TEMP_FILES ${regex})
    LIST(APPEND ${file_list} ${TEMP_FILES})
ENDMACRO(ADD_FILES)

MACRO(REMOVE_FILES file_list regex)
    FILE(GLOB_RECURSE TEMP_FILES ${regex})
    LIST(REMOVE_ITEM ${file_list} ${TEMP_FILES})
ENDMACRO(REMOVE_FILES)
