from unittest import TestCase
from Structure import *
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


class TestCubeBoundaries(TestCase):
    def test_get_vertices(self):
        cube = CubeBoundaries([1], [BoundaryType.CYCLIC])
        self.assertEqual(cube.dim, 1)
        vertices = cube.vertices
        vertices_should = [(0,), (1,)]
        for v1, v2 in zip(vertices, vertices_should):
            self.assertEqual(v1, v2)

        cube = CubeBoundaries([1, 2], [BoundaryType.CYCLIC, BoundaryType.WALL])
        vertices = cube.vertices
        vertices_should = [(0, 0), (1, 0), (0, 2), (1, 2)]
        self.assertEqual(cube.dim, 2)
        for v1, v2 in zip(vertices, vertices_should):
            self.assertEqual(v1, v2)

        cube = CubeBoundaries([1, 2, 3], [BoundaryType.CYCLIC, BoundaryType.CYCLIC, BoundaryType.WALL])
        vertices = cube.vertices
        vertices_should = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3), (0, 2, 3), (1, 0, 3), (1, 2, 0), (1, 2, 3)]
        self.assertEqual(cube.dim, 3)
        for v1, v2 in zip(vertices, vertices_should):
            self.assertEqual(v1, v2)

    def assert_list(self, vec1, vec2):
        for v1, v2 in zip(vec1, vec2): self.assertAlmostEqual(v1, v2)

    def test_vertical_step_to_wall(self):
        cube = CubeBoundaries([1], [BoundaryType.WALL])
        pt = (0.4,)
        self.assert_list(CubeBoundaries.vertical_step_to_wall(cube.walls[0], pt), (-0.4,))
        self.assert_list(CubeBoundaries.vertical_step_to_wall(cube.walls[1], pt), (0.6,))

        cube = CubeBoundaries([1, 2], [BoundaryType.WALL, BoundaryType.WALL])
        pt = (0.5, 1.5)
        self.assert_list(CubeBoundaries.vertical_step_to_wall(cube.walls[0], pt), (0, -1.5))
        self.assert_list(CubeBoundaries.vertical_step_to_wall(cube.walls[1], pt), (-0.5, 0))
        self.assert_list(CubeBoundaries.vertical_step_to_wall(cube.walls[2], pt), (0.5, 0))
        self.assert_list(CubeBoundaries.vertical_step_to_wall(cube.walls[3], pt), (0, 0.5))

        #TBD implement test for 3d

    def test_flip_v_hat_wall_part(self):
        sphere = Sphere((0.3, 0.5), 0.3)
        cube = CubeBoundaries([1, 2], [BoundaryType.WALL, BoundaryType.WALL])
        n_hat_old = (-1, 0)
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[0], sphere, n_hat_old)
        self.assert_list(n_hat_new, (-1, 0))
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[1], sphere, n_hat_old)
        self.assert_list(n_hat_new, (1, 0))
        n_hat_old = (1, 0)
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[2], sphere, n_hat_old)
        self.assert_list(n_hat_new, (-1, 0))
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[3], sphere, n_hat_old)
        self.assert_list(n_hat_new, (1, 0))

        # TBD implement test for 3d


class TestMetric(TestCase):
    def test_dist_to_boundary(self):
        self.fail()

    def test_dist_to_boundary_without_r(self):
        self.fail()

    def test_dist_to_collision(self):
        self.fail()


class TestCell(TestCase):
    def test_add_spheres(self):
        self.fail()

    def test_remove_sphere(self):
        self.fail()

    def test_should_sphere_be_in_cell(self):
        self.fail()

    def test_dim(self):
        self.fail()

    def test_random_generate_spheres(self):
        self.fail()

    def test_transform(self):
        self.fail()


class TestArrayOfCells(TestCase):
    def test_all_spheres(self):
        self.fail()

    def test_all_centers(self):
        self.fail()

    def test_all_cells(self):
        self.fail()

    def test_overlap_2_cells(self):
        self.fail()

    def test_legal_configuration(self):
        self.fail()

    def test_cell_from_ind(self):
        self.fail()
