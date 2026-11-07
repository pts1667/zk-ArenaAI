cmake_minimum_required(VERSION 3.16)
project(zksim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# No -ffast-math and no contraction: a run must not depend on the build or worker count.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffp-contract=off")

option(ZKSIM_PROFILE "Time each tick phase (tools/sim_profile.py)" OFF)

find_package(Threads REQUIRED)
add_library(zksim SHARED src/world.cpp src/combat.cpp src/movement.cpp src/views.cpp src/economy.cpp src/wrecks.cpp src/cloak.cpp src/shields.cpp src/unit_ai.cpp src/jump.cpp src/abilities.cpp src/capi.cpp)
target_include_directories(zksim PUBLIC include)
target_link_libraries(zksim PRIVATE Threads::Threads)
target_compile_options(zksim PRIVATE -Wall -Wextra)
if(ZKSIM_PROFILE)
  target_compile_definitions(zksim PUBLIC ZKSIM_PROFILE)
endif()
