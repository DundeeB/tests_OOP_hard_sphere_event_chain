from unittest import TestCase
from Structure import Sphere
from Structure import CubeBoundaries
from Structure import BoundaryType
from EventChainActions import Step
import numpy as np

class TestSphere(TestCase):
    def test_dim(self):
        sphere_1d = Sphere((1.5,), 0.5)
        self.assertEqual(sphere_1d.dim(), 1)
        sphere_2d = Sphere((1.5, 22.3), 0.5)
        self.assertEqual(sphere_2d.dim(), 2)
        sphere_3d = Sphere((1.5, 22.3, 7), 0.5)
        self.assertEqual(sphere_3d.dim(), 3)

    def test_overlap(self):
        sphere1 = Sphere((0, 0), 0.26)
        sphere2 = Sphere((0, 0.5), 0.25)
        self.assertTrue(Sphere.overlap(sphere1, sphere2))
        sphere1 = Sphere((0, 0), 0.1)
        sphere2 = Sphere((0, 0.5), 0.1)
        self.assertFalse(Sphere.overlap(sphere1, sphere2))

    def test_spheres_overlap(self):
        spheres = [Sphere((x, 0), 1) for x in [0, 2, 4, 6]]
        self.assertFalse(Sphere.spheres_overlap(spheres))
        spheres.append(Sphere((1, 0), 0.5))
        self.assertTrue(Sphere.spheres_overlap(spheres))

    def test_box_it(self):
        boundaries = CubeBoundaries([1, 1], [BoundaryType.CYCLIC for _ in range(2)])
        sphere = Sphere((1.5, 1.5), 0.3)
        sphere.box_it(boundaries)
        self.assertGreaterEqual(sphere.center[0], 0)
        self.assertGreaterEqual(sphere.center[1], 0)
        self.assertLessEqual(sphere.center[0], 1)
        self.assertLessEqual(sphere.center[1], 1)

    def test_perform_step(self):
        boundaries = CubeBoundaries([1, 1], [BoundaryType.CYCLIC for _ in range(2)])
        sphere = Sphere((0.5, 0.5), 0.3)
        v_hat = np.array((1, 1))/np.sqrt(2)
        sphere.perform_step(Step(sphere, 9, v_hat, 0.6*np.sqrt(2)), boundaries)
        self.assertAlmostEqual(sphere.center[0], 0.1)
        self.assertAlmostEqual(sphere.center[1], 0.1)

    def test_systems_length_in_v_direction(self):
        boundaries = CubeBoundaries([1, 1], [BoundaryType.CYCLIC for _ in range(2)])
        sphere = Sphere((0.5, 0.5), 0.3)
        v_hat = np.array((1, 1))/np.sqrt(2)
        self.assertAlmostEqual(sphere.systems_length_in_v_direction(v_hat, boundaries), np.sqrt(2))

    def test_trajectory(self):
        boundaries = CubeBoundaries([1, 1], [BoundaryType.CYCLIC for _ in range(2)])
        sphere = Sphere((0.5, 0.5), 0.3)
        v_hat = np.array((1, 0))
        new_loc = sphere.trajectory(0.7, v_hat, boundaries)
        self.assertAlmostEqual(new_loc[0], 0.2)
        self.assertAlmostEqual(new_loc[1], 0.5)

    def test_trajectories_braked_to_lines(self):
        boundaries = CubeBoundaries([1, 1], [BoundaryType.CYCLIC for _ in range(2)])
        sphere = Sphere((0.5, 0.5), 0.3)
        v_hat = np.array((2, 1))/np.sqrt(5)
        ps = sphere.trajectories_braked_to_lines(1.2, v_hat, boundaries)
        print(ps)
        self.assertAlmostEqual(np.linalg.norm(ps[0] - (0.5, 0.5)), 0)
        self.assertAlmostEqual(np.linalg.norm(ps[1] - (0, 0.75)), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(ps[2] - [0.57331263, 0.03665631]), 0, 3)
