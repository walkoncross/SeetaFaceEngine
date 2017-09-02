aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
find_package(OpenCV REQUIRED)

set(fi_required_libs seeta_fa_lib seeta_facedet_lib)
link_directories(.)

enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${OpenCV_LIBS} ${fi_required_libs})
endforeach ()

