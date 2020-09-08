# Adapted from http://docs.enthought.com/mayavi/mayavi/auto/example_mri.html
# http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/

import SimpleITK as sitk
import numpy as np
import  moviepy.editor as mpy

from mayavi import mlab
#mlab.init_notebook()

mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))


sitk_img = sitk.ReadImage("stack-1.nii.gz")
img = sitk.GetArrayFromImage(sitk_img)

sitk_seg = sitk.ReadImage("fetus_segmentation.nii.gz")
seg = sitk.GetArrayFromImage(sitk_seg)

src_img = mlab.pipeline.scalar_field(img)
# non isotropic data
src_img.spacing = sitk_img.GetSpacing()[::-1]
src_img.update_image_data = True

cut_plane1 = mlab.pipeline.scalar_cut_plane(src_img,
                                plane_orientation='x_axes',
                                colormap='black-white',
                                vmin=0,
                                vmax=2500)
cut_plane1.implicit_plane.origin = (68, 140, 190)
cut_plane1.implicit_plane.widget.enabled = False

cut_plane2 = mlab.pipeline.scalar_cut_plane(src_img,
                                plane_orientation='y_axes',
                                colormap='black-white',
                                vmin=0,
                                vmax=2500)
cut_plane2.implicit_plane.origin = (68, 170, 190)
cut_plane2.implicit_plane.widget.enabled = False


colormap = np.array([[0, 0, 255],  # blue
                     [255, 0, 0],  # red
                     [0, 255, 0],  # green
                     [61, 89, 171],  # cobalt
                     [237, 145, 33],  # carrot
                    ], dtype='float')/255

for i in range(1,seg.max()+1):
    src_seg = mlab.pipeline.scalar_field((seg==i).astype('uint8'))
    src_seg.spacing = sitk_seg.GetSpacing()[::-1]
    src_seg.update_image_data = True
    outer = mlab.pipeline.iso_surface(mlab.pipeline.extract_grid(src_seg),
                                      contours=[1],
                                        color=tuple(colormap[i-1]),
                                     opacity=0.5)

#mlab.show()

f = mlab.gcf()
f.scene._lift()

duration = 10 # duration of the animation in seconds (it will loop)


def make_frame(t):
    """ Generates and returns the frame for time t. """
    mlab.view(azimuth= 360*t/duration, elevation=90)
    return mlab.screenshot(antialiased=True) # return a RGB image

animation = mpy.VideoClip(make_frame, duration=duration)
animation.write_videofile("movie.mp4", fps=20)
animation.write_gif("movie.gif", fps=20)