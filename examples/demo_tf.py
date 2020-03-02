"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import soft_renderer as sr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1), 
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss


def load_png(path, alpha_val=None):
    img = torch.tensor(imageio.imread(path).astype('float32') / 255.)
    if alpha_val is not None:
        alpha = torch.tensor(alpha_val).expand(img.shape[0], img.shape[1], 1)
        with_alpha = torch.cat((img, alpha), 2)
        img = with_alpha

    return img.permute(2,0,1)

def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str, 
        default=os.path.join(data_dir, 'obj/sphere/sphere_1352.obj'))
    parser.add_argument('-o', '--output-dir', type=str, 
        default=os.path.join(data_dir, 'results/output_deform'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cameras = [
        # eye, direction
        ([-5, 0, 0], [1, 0, 0]),
        ([0, 0, -5], [0, 0, 1]),
    ]

    model = Model(args.template_mesh).cuda()
    renderer = sr.SoftRenderer(image_size=64, sigma_val=1e-4, aggr_func_rgb='softmax', 
                               camera_mode='look', viewing_angle=10)
    renderer.transform.transformer._eye = [ eye for eye, _ in cameras ]
    renderer.transform.transformer.camera_direction = [ direction for _, direction in cameras ]

    # read training images and camera poses
    images = torch.stack([
        load_png(os.path.join(data_dir, 'target/t.png')),
        load_png(os.path.join(data_dir, 'target/f.png'))
    ])
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))

    loop = tqdm.tqdm(list(range(0, 3000)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    images_gt = images.cuda()
    for i in loop:

        mesh, laplacian_loss, flatten_loss = model(len(cameras))
        images_pred = renderer.render_mesh(mesh)

        # optimize mesh with silhouette reprojection error and 
        # geometry constraints
        loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 0]) + \
               0.0003 * flatten_loss
               # 0.0003 * laplacian_loss + \

        loop.set_description('Loss: %.8f'%(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            image1 = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            image2 = images_pred.detach().cpu().numpy()[1].transpose((1, 2, 0))
            writer.append_data((255*image1).astype(np.uint8))
            imageio.imsave(os.path.join(args.output_dir, 'deform1_%05d.png'%i), (255*image1[..., -1]).astype(np.uint8))
            imageio.imsave(os.path.join(args.output_dir, 'deform2_%05d.png'%i), (255*image2[..., -1]).astype(np.uint8))

    # save optimized mesh
    model(1)[0].save_obj(os.path.join(args.output_dir, 'tf.obj'), save_texture=False)


if __name__ == '__main__':
    main()
