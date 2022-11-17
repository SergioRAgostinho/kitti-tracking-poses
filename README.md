# Demo on how to render 3D bounding boxes KITTI tracking frames

This repo exemplifies how to convert the label poses in KITTI tracking training split, to a usable rotation and translation that can be used to render vertically aligned bounding boxes that look like this

**Sequence 00**

![sequence00](/assets/train-0000.gif)

**Sequence 15**

![sequence15](/assets/train-0015.gif)

**Sequence 20**

![sequence20](/assets/train-0020.gif)

This [section](/render_kitti_tracking_3d_bbox.py#L66) of the code exemplifies how to get the pose with respect to any frame.
