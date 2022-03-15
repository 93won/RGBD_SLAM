add_library(rgbd_slam SHARED
Config.cpp
types/Camera.cpp 
types/Feature.cpp
types/Frame.cpp
types/Map.cpp
types/MapPoint.cpp
utils/Viewer.cpp
core/Tracker.cpp
core/Matcher.cpp
core/Extractor.cpp
core/Optimizer.cpp)

target_link_libraries(rgbd_slam ${THIRD_PARTY_LIBS})
