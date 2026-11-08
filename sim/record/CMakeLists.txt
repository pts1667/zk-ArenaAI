# libzkrecord: recording a batch's games straight from its arrays (zkrecord.h), for tools/policy_clone_data.py.  Links zstd
# (libzstd-dev), nothing of the sim's.
find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)
find_library(ZSTD_LIBRARY zstd REQUIRED)
add_library(zkrecord SHARED src/record.cpp)
target_include_directories(zkrecord PRIVATE . ../include ${ZSTD_INCLUDE_DIR})
target_link_libraries(zkrecord PRIVATE Threads::Threads ${ZSTD_LIBRARY})
target_compile_options(zkrecord PRIVATE -Wall -Wextra)
# Beside libzksim.so, where sim/py/zkrecord.py looks for it.
set_target_properties(zkrecord PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
