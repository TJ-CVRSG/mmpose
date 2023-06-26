dataset_info = dict(
    dataset_name='tjlp',
    paper_info=dict(
        author='tjlp',
        title='tjlp',
        container='tjlp',
        year='2023',
        homepage='https://cvrsg.tongji.edu.cn/',
    ),
    keypoint_info={
        0:
        dict(name='bottom_right', 
            id=0, 
            color=[0, 255, 0],
            type='upper'),
        1:
        dict(
            name='bottom_left',
            id=1,
            color=[0, 255, 255],
            type='upper'),
        2:
        dict(
            name='top_left',
            id=2,
            color=[255, 255, 0],
            type='upper'),
        3:
        dict(
            name='top_right',
            id=3,
            color=[255, 0, 0],
            type='upper'),
    },
    skeleton_info={
        0:
        dict(link=('bottom_right', 'bottom_left'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('bottom_left', 'top_left'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('top_left', 'top_right'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('top_right', 'bottom_right'), id=3, color=[0, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025
    ])
