import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import soft_renderer as sr


class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=True, fill_back=True, eps=1e-6,
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(Renderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.Rasterizer(image_size, background_color, near, far, 
                                        anti_aliasing, fill_back, eps)

    def forward(self, mesh, mode=None):
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)


class SoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0], shadow=True,
                 light_width=2, softmin_scale=10):
        super(SoftRenderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode,
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale,
                                      eye, camera_direction)

        if not isinstance(light_directions, (torch.Tensor, np.ndarray)):
            light_directions = torch.FloatTensor(light_directions)
        self.light_space_tform = None
        if shadow:
            # TODO: Allow specifying the light distance
            # TODO: the light should really get its own viewing_scale instead of
            #       sharing with the main camera
            self.light_space_tform = sr.Transform('look_at', perspective=False,
                                                  eye=30*light_directions,
                                                  viewing_scale=viewing_scale)
        self.viewing_scale = viewing_scale

        # rasterization
        self.depth_rasterizer = sr.SoftRasterizer(image_size, background_color, near, far,
                                                  False, fill_back, eps,
                                                  sigma_val, dist_func, dist_eps,
                                                  gamma_val, 'depth', aggr_func_alpha,
                                                  texture_type)
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far,
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type, light_width, softmin_scale)

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh):
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh)

        if self.light_space_tform is not None:
            shadow_mesh = self.light_space_tform(mesh)
            shadow_map = self.depth_rasterizer(shadow_mesh)
            shadow_vertices = shadow_mesh.face_vertices
            # map x, y from [-1, 1] to [0, 1] range
            shadow_vertices[:, :, :, 0] *= 0.5
            shadow_vertices[:, :, :, 0] += 0.5
            shadow_vertices[:, :, :, 1] *= 0.5
            shadow_vertices[:, :, :, 1] += 0.5
        else:
            shadow_map = shadow_vertices = None

        return self.rasterizer(self.transform(mesh), shadow_map, shadow_vertices)

    def forward(self, vertices, faces, textures=None, mode=None, texture_type='surface'):
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        return self.render_mesh(mesh, mode)
