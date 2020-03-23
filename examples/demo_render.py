"""
Demo render. 
1. save / load textured .obj file
2. render using SoftRas with different sigma / gamma
"""
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import soft_renderer as sr


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-o', '--output-dir', type=str, 
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    # other settings
    camera_distance = 5
    elevation = 30
    azimuth = 0

    # load from Wavefront .obj file
    mesh = sr.Mesh.from_obj(args.filename_input)
                            # load_texture=True, texture_res=5, texture_type='surface')

    # create renderer with SoftRas
    render_params = dict(
        light_intensity_ambient=0.2,
        light_intensity_directionals=0.8,
        light_directions=[1/np.sqrt(2), 1/np.sqrt(2), 0],
    )
    renderer = sr.SoftRenderer(camera_mode='look_at', **render_params)
    depth_renderer = sr.SoftRenderer(aggr_func_rgb='depth',
                                     camera_mode='look_at',
                                     eye=[8.0, 8.0, 0.0],
                                     camera_direction=render_params['light_directions'],
                                     **render_params)

    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'), mode='I')

    with imageio.get_writer(os.path.join(args.output_dir, 'depth.png')) as depth_writer:
        mesh.reset_()
        depth_images = depth_renderer.render_mesh(mesh)
        depth_image = depth_images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        depth_writer.append_data((255*depth_image).astype(np.uint8))

    for num, azimuth in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        loop.set_description('Drawing rotation')
        renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        images = renderer.render_mesh(mesh)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # draw object from different sigma and gamma
    loop = tqdm.tqdm(list(np.arange(-4, -2, 0.2)))
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, 45)
    writer = imageio.get_writer(os.path.join(args.output_dir, 'bluring.gif'), mode='I')
    for num, gamma_pow in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        renderer.set_gamma(10**gamma_pow)
        renderer.set_sigma(10**(gamma_pow - 1))
        loop.set_description('Drawing blurring')
        images = renderer.render_mesh(mesh)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # save to textured obj
    mesh.reset_()
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'), save_texture=True)


if __name__ == '__main__':
    main()
