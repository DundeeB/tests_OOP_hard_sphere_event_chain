from unittest import TestCase
from SnapShot import View2D
from Structure import Sphere
from Structure import CubeBoundaries
from Structure import BoundaryType
from Structure import ArrayOfCells
from Structure import Cell
import os

spheres = [Sphere((x, y), 0.1) for x,y in
           zip([0.2, 0.5, 0.8, 1.5], [0.1, 0.7, 0.9, 1.95])]
output_dir = 'test_garb'
if not os.path.isdir(output_dir): os.mkdir(output_dir)
boundaries = CubeBoundaries([3, 3], [BoundaryType.CYCLIC for _ in range(2)])
view = View2D(output_dir, boundaries)


class TestView2D(TestCase):

    def test_spheres_snapshot(self):
        view.spheres_snapshot('debug spheres', spheres, 'DebugSpheres')
        self.assertEqual(1, 1, '')

    def test_step_snapshot(self):
        view.step_snapshot('debug step', spheres, 1, 'DebugStep', (0.2, 0.2), 1)
        self.assertEqual(1,1,'')

    def test_array_of_cells_snapshot(self):
        cells = [Cell((0, 0), [1, 1], spheres=spheres[:-1]), Cell((1, 1), [1, 1], spheres=[spheres[-1]])]
        view.array_of_cells_snapshot('debug cells', ArrayOfCells(2, boundaries, cells), 'DebugCells')
        self.assertEqual(1, 1, '')

    def test_save_video(self):
        vid_name = 'tests_video'
        if os.path.exists(view.output_dir+'/'+vid_name):
            os.remove(view.output_dir+'/'+vid_name)
        view.save_video(vid_name, fps=0.2)
        self.assertEqual(1, 1, '')
