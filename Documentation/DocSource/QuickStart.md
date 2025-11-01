# Quick Start

The example code provided in this quickstart guide is for educational and demonstration purposes only.
It may not represent best practices for production use.

Most functionality in this package comes from two components: `DepthPreprocessor` and `DepthMesher`. Below is a quick guide to get them set up in your scene for runtime environment meshing on Meta Quest.

For full API details, see the [reference documentation](~/api/Uralstech.UXR.QuestMeshing.yml).

## DepthPreprocessor

This component fetches depth frames from the Meta OpenXR API and exposes the depth texture to shaders. It's similar to Meta's `EnvironmentDepthManager` but adds exposure of intrinsic frame data (like, view and projection matrices). It also generates and updates two RenderTextures:
- Worldspace Positions: 3D positions for each pixel in the depth texture (world coordinates).
- Worldspace Normals: Surface normals for corresponding points.

These textures feed into `DepthMesher` for real-time mesh generation.

This script directly conflicts with `EnvironmentDepthManager`, `AROcclusionManager`, and any other script using `XROcclusionSubsystem.TryGetFrame()`. It maintains partial compatibility by setting global shader vars (like `_EnvironmentDepthTexture`) for Meta's occlusion shaders.

See the [API reference](~/api/Uralstech.UXR.QuestMeshing.DepthPreprocessor.yml) for more.

## DepthMesher

This consumes data from `DepthPreprocessor` to build a dynamic mesh using the Surface Nets algorithm. It can then assign the mesh to a `MeshFilter`, bake collision via the Jobs system and then assign it to a `MeshCollider`, and bake a NavMesh with `NavMeshSurface`.

`DepthMesher` requires an instance of `DepthPreprocessor` to be in the same scene.

### Main Editor Variables
These key settings are exposed in the Inspector for tuning mesh quality, performance, and scope:

- **Volume Size**: The dimensions of the TSDF volume grid (X: width, Y: height, Z: depth). Higher resolutions improve detail but increase memory and compute cost. (Default: 256x64x256)
- **Meters Per Voxel**: The real-world size represented by each voxel (in meters). Smaller values yield finer detail but require larger volumes. (Default: 0.1m, Min: 0)
- **Min View Distance**: The minimum distance from the camera at which depth data is considered for meshing (ignores closer user-occluded data). (Default: 1m, Min: 0)
- **Max View Distance**: The maximum distance from the camera at which depth data is considered for meshing. (Default: 4m, Min: 0)
- **Max Mesh Update Distance**: The maximum distance from the user's position to update the mesh (optimizes for local changes). (Default: 4m, Min: 0)
- **Triangles Budget**: The maximum number of triangles allowed in the generated mesh (caps GPU memory usage). (Default: 262144)
- **Target Volume Update Rate Hertz**: The target update frequency for the TSDF volume (in Hz). Higher rates improve responsiveness but increase GPU load. (Default: 45, Min: 0)
- **Target Mesh Refresh Rate Hertz**: The target refresh frequency for the generated mesh (in Hz). Lower rates reduce CPU overhead for stable scenes. (Default: 1, Min: 0)

## Utility Component: CPUDepthSampler

See the [API reference](~/api/Uralstech.UXR.QuestMeshing.CPUDepthSampler.yml) on how to use `CPUDepthSampler` to asynchronously .

## Breaking Changes Notice

If you've just updated the package, it is recommended to check the [*changelogs*](https://github.com/Uralstech/UXR.QuestMeshing/releases) for information on breaking changes.