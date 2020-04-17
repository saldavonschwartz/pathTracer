//
//  PathTracer.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "PathTracer.hpp"

#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp>
#include "Utils.hpp"
#include "Scene.hpp"

using namespace std;
using namespace glm;

class Ray {
public:
  vec3 origin;
  vec3 dir;

  Ray() = default;

  Ray(vec3 origin, vec3 dir)
  : origin(origin), dir(normalize(dir)) {}

  vec3 operator()(float t) const {
    return origin + t * dir;
  }
};

class Camera {
public:
  vec3 position;
  float aspectRatio;
  float focalLength;
  float aperture;
  float fovy;

  Camera(const vec3& position, const vec3& lookAt, float fovy, float aspectRatio, float focalLength, float aperture)
  : position(position), fovy(fovy), aspectRatio(aspectRatio), focalLength(focalLength), aperture(aperture) {
    float hh = tan(radians(fovy)/2.f);
    float hw = aspectRatio * hh;

    // [x y z p] = new camera orientation:
    vec3 z = normalize(position - lookAt);
    x = normalize(cross({0.f, 1.f, 0.f}, z));
    y = cross(z, x);

    auto& f = focalLength;
    lowerLeftImageOrigin = position - x*hw*f - y*hh*f - z*f;
    hOffset = 2*hh*f*y;
    wOffset = 2*hw*f*x;
  }

  Ray castRay(float u, float v) {
    auto r = (aperture / 2.f) * circularRand(1.f);
    auto offset = x*r.x + y*r.y;
    auto dir = lowerLeftImageOrigin + u*wOffset + v*hOffset - position - offset;
    return {position + offset, dir};
  }

private:
  float hh, hw;
  vec3 x, y;
  vec3 hOffset;
  vec3 wOffset;
  vec3 lowerLeftImageOrigin;
};

class AABA {
public:
  vec3 xmin{0.f};
  vec3 xmax{0.f};
  
  static AABA surroundingBox(const AABA& bBox0, const AABA& bBox1) {
    return AABA(min(bBox0.xmin, bBox1.xmin), max(bBox0.xmax, bBox1.xmax));
  }
  
  AABA() = default;
  
  AABA(vec3 xmin, vec3 xmax)
  : xmin(xmin), xmax(xmax) {}
  
  bool hit(const Ray& ray, float tmin, float tmax) const {
    auto xminIntersect = (xmin - ray.origin) / ray.dir;
    auto xmaxIntersect = (xmax - ray.origin) / ray.dir;
    auto t0 = min(xminIntersect, xmaxIntersect);
    auto t1 = max(xminIntersect, xmaxIntersect);
    tmin = compMax(vec4(t0, tmin));
    tmax = compMin(vec4(t1, tmax));
    return tmin < tmax;
  }
};

class Material;

class Hittable {
public:
  struct HitInfo {
    vec3 hitPoint{0.f};
    vec3 normal{0.f};
    bool isFrontFace = false;
    float t = 0;
    shared_ptr<Material> material;
  };
  
  virtual bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const = 0;
  virtual bool boundingBox(double t0, double t1, AABA& bBox) const = 0;
  virtual ~Hittable() = default;
};

class HittableVector : public Hittable {
public:
  vector<shared_ptr<Hittable>> data;
  
  void push_back(shared_ptr<Hittable> hittable) {
    data.push_back(hittable);
  }
  
  bool boundingBox(double t0, double t1, AABA& bBox) const override {
    bool firstTime = true;
    AABA bBoxTemp;
    
    for (auto& h: data) {
      if (!h->boundingBox(t0, t1, bBoxTemp)) {
        return false;
      }
      
      if (firstTime) {
        firstTime = false;
        bBox = bBoxTemp;
      }
      else {
        bBox = AABA::surroundingBox(bBox, bBoxTemp);
      }
    }
    
    return true;
  }
  
  bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override {
    float closest = tmax;
    bool hitAny = false;
    HitInfo _info;
    
    for (auto& h : data) {
      if (h->hit(ray, tmin, closest, _info)) {
        closest = _info.t;
        hitAny = true;
      }
    }
    
    if (hitAny) {
        info = _info;
    }
    
    return hitAny;
  }
};

class BVHNode : public Hittable {
public:
  shared_ptr<Hittable> right;
  shared_ptr<Hittable> left;
  AABA bBox;

  BVHNode(HittableVector list, float t0, float t1)
  : BVHNode(list.data, 0, list.data.size(), t0, t1)
  {}

  BVHNode(std::vector<shared_ptr<Hittable>>& objects, size_t start, size_t end, float t0, float t1) {
    int axis = rand() % 3;
    auto cmp = [axis](const shared_ptr<Hittable> h1, const shared_ptr<Hittable> h2){
      AABA bBox1;
      AABA bBox2;

      if (!h1->boundingBox(0,0, bBox1) || !h2->boundingBox(0,0, bBox2)) {
        std::cerr << "No bounding box in BVHNode constructor.\n";
      }

      return bBox1.xmin[axis] < bBox2.xmin[axis];
    };

    auto objectsLeft = end - start;

    switch (objectsLeft) {
      case 1: {
        left = right = objects[start];
        break;
      }

      case 2: {
        if (cmp(objects[start], objects[start+1])) {
          left = objects[start];
          right = objects[start+1];
        }
        else {
          left = objects[start+1];
          right = objects[start];
        }

        break;
      }

      default: {
        sort(objects.begin() + start, objects.begin() + end, cmp);
        auto mid = start + objectsLeft/2;
        left = make_shared<BVHNode>(objects, start, mid, t0, t1);
        right = make_shared<BVHNode>(objects, mid, end, t0, t1);

        break;
      }
    }

    AABA bBoxLeft, bBoxRight;

    if (!left->boundingBox(t0, t1, bBoxLeft) || !right->boundingBox(t0, t1, bBoxRight)) {
      std::cerr << "No bounding box in bvh_node constructor.\n";
    }

    bBox = AABA::surroundingBox(bBoxLeft, bBoxRight);
  }

  bool boundingBox(double t0, double t1, AABA& bBox) const override {
    bBox = this->bBox;
    return true;
  }

  bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override {
    if (!bBox.hit(ray, tmin, tmax)) {
      return false;
    }

    bool leftHit = left->hit(ray, tmin, tmax, info);
    bool rightHit = right->hit(ray, tmin,  leftHit ? info.t : tmax, info);
    return leftHit || rightHit;
  }
};

class Sphere : public Hittable {
public:
  shared_ptr<Material> material;
  float radius;
  vec3 position;
  
  Sphere(vec3 position, float radius, shared_ptr<Material> material)
  : position(position), radius(radius), material(material) {}
  
  bool boundingBox(double t0, double t1, AABA& bBox) const override {
    vec3 extent{radius};
    bBox = AABA(position - extent, position + extent);
    return true;
  }
  
  bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override {
    // From ray eq. r(t) = 0 + t*d and sphere eq. (x-c)^2 - r^2 = 0
    // Solve quadratic (r(t)-c)^2 -r^2 = 0 -> a*t^2 + b*t + c = 0
    // 0 roots = no hit, 1 root = tanget hit, 2 roots = went in and thru:
    
    vec3 oc = ray.origin - position;
    float a = dot(ray.dir, ray.dir);
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    
    if (discriminant > 0) {
      float t = (-b - sqrt(discriminant))/a;
      
      if (tmin < t && t < tmax) {
        auto hitPoint = ray(t);
        info.hitPoint = hitPoint;
        info.normal = (hitPoint - position) / radius;
        info.isFrontFace = dot(info.normal, ray.dir) < 0.f;
        info.normal = info.isFrontFace ? info.normal : -info.normal;
        info.t = t;
        info.material = material;
        return true;
      }
      
      t = (-b + sqrt(discriminant))/a;
      
      if (tmin < t && t < tmax) {
        auto hitPoint = ray(t);
        info.hitPoint = hitPoint;
        info.normal = (hitPoint - position) / radius;
        info.isFrontFace = dot(info.normal, ray.dir) < 0.f;
        info.normal = info.isFrontFace ? info.normal : -info.normal;
        info.t = t;
        info.material = material;
        return true;
      }
    }
    
    return false;
  }
};

class Material {
public:
  virtual bool scatter(const Ray& ray, const Hittable::HitInfo& info, vec3& attenuation, Ray& scattered) const = 0;
  virtual ~Material() = default;
};

class Diffuse : public Material {
public:
  vec3 albedo;
  
  Diffuse(const vec3& albedo) : albedo(albedo) {}
  
  bool scatter(const Ray& ray, const Hittable::HitInfo& info, vec3& attenuation, Ray& scattered) const override {
    vec3 scatterDir = info.normal + sphericalRand(1.f);
    scattered = Ray(info.hitPoint, scatterDir);
    attenuation = albedo;
    return true;
  }
};

class Metal : public Material {
public:
  vec3 albedo;
  float fuzziness;
  
  Metal(const vec3& albedo, float fuzziness)
  : albedo(albedo), fuzziness(fuzziness) {}
  
  bool scatter(const Ray& ray, const Hittable::HitInfo& info, vec3& attenuation, Ray& scattered) const override {
    vec3 scatterDir = reflect(ray.dir, info.normal);
    scattered = Ray(info.hitPoint, scatterDir + fuzziness * ballRand(1.f));
    attenuation = albedo;
    return dot(scattered.dir, info.normal) > 0.f;
  }
};

class Dielectric : public Material {
public:
  float refractionIdx;
  
  Dielectric(float refractionIdx)
  : refractionIdx(refractionIdx) {}
  
  bool scatter(const Ray& ray, const Hittable::HitInfo& info, vec3& attenuation, Ray& scattered) const override {
    attenuation = vec3{1};
    auto k1 = info.isFrontFace ? 1.f : refractionIdx;
    auto k2 = info.isFrontFace ? refractionIdx : 1.f;
    auto r = reflectance(ray.dir, info.normal, k1, k2);
    
    if (r == 1.f || urand(0.f, 1.f) < r) {
      vec3 scatterDir = reflect(ray.dir, info.normal);
      scattered = Ray(info.hitPoint, scatterDir);
      return true;
    }
    
    vec3 scatterDir = refract(ray.dir, info.normal, k1, k2);
    scattered = Ray(info.hitPoint, scatterDir);
    return true;
  }
};

vec3 sampleRay(const Ray& ray, float tmin, float tmax, int maxBounces, Hittable& hittable) {
  if (maxBounces <= -1) {
    return {};
  }
  
  Hittable::HitInfo info;
  
  if (hittable.hit(ray, tmin, tmax, info)) {
    vec3 attenuation;
    Ray scattered;
    
    if (info.material->scatter(ray, info, attenuation, scattered)) {
      return attenuation * sampleRay(scattered, tmin, tmax, maxBounces - 1, hittable);
    }
    
    return {};
  }
  
  float t = to01Range(ray.dir).y;
  return (1.f-t) * vec3{1.f} + t * vec3{.5f, .7f, 1.f};
}

int renderScene(int sceneId, string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
  ofstream outputImage(path + "out2.ppm");
  
  if (!outputImage.is_open()) {
    cerr << "ERROR: could not open output file!\n";
    return -1;
  }
  
  float aspectRatio = double(width) / height;
  vec3 lookFrom;
  vec3 lookAt;
  float focalLength;
  float aperture;
  float fovy;
  HittableVector sceneObjects;
  
  if (sceneId == 1) {
    lookFrom = {3.f, 3.f, 2.f};
    lookAt = {0.f, 0.f, -1.f};
    focalLength = length(lookFrom-lookAt);
    aperture = 0.1;
    fovy = 20;
    sceneObjects = generateSimpleScene();
  }
  else {
    lookFrom = {13.f, 2.f, 3.f};
    lookAt = vec3{0.f};
    focalLength = 10.f;
    aperture = 0.1f;
    fovy = 20.f;
    sceneObjects = generateComplexScene();
  }
  
  HittableVector world;
  Camera cam(lookFrom, lookAt, fovy, aspectRatio, focalLength, aperture);
  BVHNode scene(sceneObjects, 0.f, 1.f);
  int rowsProcessed  = 0;
  
  // Output image is in PPM 'plain' format (http://netpbm.sourceforge.net/doc/ppm.html#plainppm)
  outputImage << "P3\n" << width << " " << height << "\n255\n";
  
  // Image origin is bottom-left.
  // Pixels are output one row at a time, top to bottom, left to right:
  {
    auto p = Profiler("[Render Time]");
    
    for (int y = height-1; y >= 0; y--) {
      if (!rowsProcessed) {
        cerr << fixed << setprecision(1) <<
        "\n" << 100.f*(1.f - (float)y/height) << "% complete" <<
        std::flush;
      }
      
      rowsProcessed = (rowsProcessed + 1) % 20;
      
      for (int x = 0; x < width; x++) {
        vec3 pixel{0};
        
        for (int r = 0; r < raysPerPixel; r++) {
          float u = (x + urand(0, 1))/float(width);
          float v = (y + urand(0, 1))/float(height);
          Ray ray = cam.castRay(u, v);
          pixel += sampleRay(ray, 0.001f, inf, maxBouncesPerRay, scene);
        }
        
        pixel = 255.f * glm::clamp(glm::sqrt(pixel/float(raysPerPixel)), 0.f, 1.f);
        outputImage << int(pixel.r) << " " << int(pixel.g) << " " << int(pixel.b) << "\n";
      }
    }
  }
  
  cerr << "\n100% complete" << std::flush;
  outputImage.close();
  return 0;
}


