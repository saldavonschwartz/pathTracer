//
//  PathTracer.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "PathTracer.hpp"

#include <iostream>
#include <fstream>
#include "Utils.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Material.hpp"
#include "Camera.hpp"
#include "HittableVector.hpp"
#include "BVH.hpp"

using std::ofstream;
using std::string;
using std::cerr;

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
        cerr << std::fixed << std::setprecision(1) <<
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


