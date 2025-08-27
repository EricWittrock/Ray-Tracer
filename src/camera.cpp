#include "camera.h"
#include "config.h"


Camera::Camera()
    : position(0, 0, 0),
      direction(0, 0, -1)
{
}


void Camera::render() const {
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_WIDTH; i++) {
        int x = i % IMAGE_WIDTH;
        int y = i / IMAGE_WIDTH;
    }

}