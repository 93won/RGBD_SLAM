add_executable(test_rgbd main.cpp)
target_link_libraries(test_rgbd rgbd_slam ${THIRD_PARTY_LIBS})