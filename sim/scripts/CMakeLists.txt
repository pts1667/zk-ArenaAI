# libzkscripts: the scripted players of tools/policy_sim_env.py in C++ (zkscripts.h).  It reads the raw-frame records and writes
# orders, and links nothing of the sim's.
add_library(zkscripts SHARED src/api.cpp src/army.cpp src/eco.cpp)
target_include_directories(zkscripts PRIVATE . ../include)
target_link_libraries(zkscripts PRIVATE Threads::Threads)
target_compile_options(zkscripts PRIVATE -Wall -Wextra)
# Beside libzksim.so, where tools/policy_scripts_cpp.py looks for it.
set_target_properties(zkscripts PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
