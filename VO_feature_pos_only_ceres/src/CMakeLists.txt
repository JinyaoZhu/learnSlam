add_library( myslam SHARED
    camera.cpp
    config.cpp
    g2o_type.cpp
    frame.cpp
    map.cpp
    mappoint.cpp
    vo.cpp
    ceres_bundle.cpp
)

target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)
