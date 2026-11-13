# libzkpolicy: the policy's inference side in C++ (zkpolicy.h): the type table, the featurizer and the actions, bit for bit
# with bots/zkbot/policy.  Links nothing of the sim's; its binding is sim/py/zkpolicy.py.
add_library(zkpolicy SHARED src/table.cpp src/bundle.cpp src/featurize.cpp src/actions.cpp src/api.cpp)
target_include_directories(zkpolicy PRIVATE . ../include)
target_link_libraries(zkpolicy PRIVATE Threads::Threads)
target_compile_options(zkpolicy PRIVATE -Wall -Wextra)
# Beside libzksim.so, where sim/py/zkpolicy.py looks for it.
set_target_properties(zkpolicy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
