#include "vec3.h"
#include "ray.h"

class Material {
public:
    __device__ Vec3 reflect(Ray& ray, const Vec3& normal, const Vec3& pos) const
    {
        Vec3 newDir = ray.direction.reflect((normal/* + Vec3::random() * roughness*/).normalize());
        ray.position = pos;
        ray.direction = newDir;
        ray.marchForward(0.0001);

        return color;
    };

    Vec3 color = Vec3(1.0, 1.0, 1.0);
    double roughness = 0.0;
};