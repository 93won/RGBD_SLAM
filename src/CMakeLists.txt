add_library(rgbd_slam SHARED
Config.cpp
types/Camera.cpp
types/Feature.cpp
types/Frame.cpp
types/Map.cpp
types/MapPoint.cpp
utils/PointCloudUtils.cpp
utils/Viewer.cpp
core/Frontend.cpp
core/Matcher.cpp
core/Extractor.cpp
core/Estimator.cpp
core/PoseGraphOptimizer.cpp)

target_link_libraries(rgbd_slam ${THIRD_PARTY_LIBS})
