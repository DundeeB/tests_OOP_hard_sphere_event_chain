from unittest import TestCase
from Structure import *
from EventChainActions import Step
import numpy as np
from SnapShot import View2D
epsilon = 1e-4


def assert_list(self, vec1, vec2):
    for v1, v2 in zip(vec1, vec2): self.assertAlmostEqual(v1, v2)


class TestSphere(TestCase):
    def test_dim(self):
        sphere_1d = Sphere((1.5,), 0.5)
        self.assertEqual(sphere_1d.dim, 1)
        sphere_2d = Sphere((1.5, 22.3), 0.5)
        self.assertEqual(sphere_2d.dim, 2)
        sphere_3d = Sphere((1.5, 22.3, 7), 0.5)
        self.assertEqual(sphere_3d.dim, 3)

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
        sphere.perform_step(Step(sphere, 9, v_hat, boundaries, 0.6*np.sqrt(2)))
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
        ps, ts = sphere.trajectories_braked_to_lines(1.2, v_hat, boundaries)
        self.assertAlmostEqual(np.linalg.norm(ps[0] - (0.5, 0.5)), 0)
        self.assertAlmostEqual(ts[0], 0)
        self.assertAlmostEqual(np.linalg.norm(ps[1] - (0, 0.75)), 0, 3)
        self.assertAlmostEqual(ts[1], np.linalg.norm(np.array([1, 0.75]) - (0.5, 0.5)), 3)
        self.assertAlmostEqual(np.linalg.norm(ps[2] - [0.57331263, 0.03665631]), 0, 3)
        self.assertAlmostEqual(ts[2], 1.2)


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

    def test_vertical_step_to_wall(self):
        cube = CubeBoundaries([1], [BoundaryType.WALL])
        pt = (0.4,)
        assert_list(self, CubeBoundaries.vertical_step_to_wall(cube.walls[0], pt), (-0.4,))
        assert_list(self, CubeBoundaries.vertical_step_to_wall(cube.walls[1], pt), (0.6,))

        cube = CubeBoundaries([1, 2], [BoundaryType.WALL, BoundaryType.WALL])
        pt = (0.5, 1.5)
        assert_list(self, CubeBoundaries.vertical_step_to_wall(cube.walls[0], pt), (0, -1.5))
        assert_list(self, CubeBoundaries.vertical_step_to_wall(cube.walls[1], pt), (-0.5, 0))
        assert_list(self, CubeBoundaries.vertical_step_to_wall(cube.walls[2], pt), (0.5, 0))
        assert_list(self, CubeBoundaries.vertical_step_to_wall(cube.walls[3], pt), (0, 0.5))

        #TBD implement test for 3d

    def test_flip_v_hat_wall_part(self):
        sphere = Sphere((0.3, 0.5), 0.3)
        cube = CubeBoundaries([1, 2], [BoundaryType.WALL, BoundaryType.WALL])
        n_hat_old = (-1, 0)
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[0], sphere, n_hat_old)
        assert_list(self, n_hat_new, (-1, 0))
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[1], sphere, n_hat_old)
        assert_list(self, n_hat_new, (1, 0))
        n_hat_old = (1, 0)
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[2], sphere, n_hat_old)
        assert_list(self, n_hat_new, (-1, 0))
        n_hat_new = CubeBoundaries.flip_v_hat_wall_part(cube.walls[3], sphere, n_hat_old)
        assert_list(self, n_hat_new, (1, 0))

        # TBD implement test for 3d

    def test_sphere_dist(self):
        boundaries = CubeBoundaries([20, 20], [BoundaryType.CYCLIC for _ in range(2)])
        sphere1 = Sphere((19, 5), 2.0)
        sphere2 = Sphere((4, 5), 2.0)
        self.assertAlmostEqual(boundaries.sphere_dist(sphere1, sphere2), 5.0)
        self.assertAlmostEqual(sphere1.sphere_dist(sphere2), 15)


class TestMetric(TestCase):
    def test_dist_to_boundary(self):
        bound = CubeBoundaries([1, 2], [BoundaryType.CYCLIC, BoundaryType.WALL])
        sphere = Sphere((0.5, 0.5), 0.1)
        v_hat = np.array([1, 1])/np.sqrt(2)
        dist_to_hit, wall_to_hit = Metric.dist_to_boundary(sphere, 3, v_hat, bound)
        self.assertAlmostEqual(dist_to_hit, 1.4*np.sqrt(2))
        self.assertEqual(wall_to_hit, bound.walls[3])

    def test_dist_to_boundary_without_r(self):
        bound = CubeBoundaries([1, 2], [BoundaryType.CYCLIC, BoundaryType.WALL])
        sphere = Sphere((0.5, 0.5), 0.1)
        v_hat = np.array([1, 1]) / np.sqrt(2)
        dist_to_hit, wall_to_hit = Metric.dist_to_boundary_without_r(sphere, 3, v_hat, bound)
        self.assertAlmostEqual(dist_to_hit, 0.5 * np.sqrt(2) + epsilon)
        self.assertEqual(wall_to_hit, bound.walls[2])

    def test_dist_to_collision(self):
        sphere1 = Sphere((0.5, 1), 0.3)
        sphere2 = Sphere((0.5, 2), 0.3)
        diam  = sphere1.rad + sphere2.rad
        v_hat = np.array((0, 1))
        bound = CubeBoundaries([3, 3], [BoundaryType.CYCLIC, BoundaryType.CYCLIC])
        d1 = Metric.dist_to_collision(sphere1, sphere2, 10, v_hat, bound)
        self.assertAlmostEqual(d1, 1-diam)
        d1 = Metric.dist_to_collision(sphere1, sphere2, 0.1, v_hat, bound)
        self.assertEqual(d1, float('inf'))

        d1 = Metric.dist_to_collision(sphere1, sphere2, 10, -v_hat, bound)
        self.assertEqual(d1, 2-diam)


class TestCell(TestCase):
    def test_add_spheres(self):
        cell = Cell((1, 1), [2, 2], (0, 0), [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)])
        sp = Sphere((0,1), 2)
        cell.add_spheres(sp)
        self.assertEqual(cell.spheres[-1], sp)

    def test_remove_sphere(self):
        sp = Sphere((0, 1), 2)
        other_spheres = [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)]
        cell = Cell((1, 1), [2, 2], (0, 0), other_spheres + [sp])
        cell.remove_sphere(sp)
        self.assertEqual(cell.spheres, other_spheres)

    def test_should_sphere_be_in_cell(self):
        cell = Cell((1, 1), [2, 2], (0, 0), [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)])
        sp = Sphere((0, 1, 2), 2)
        self.assertFalse(cell.should_sphere_be_in_cell(sp))
        sp = Sphere((1.5, 1.5, 2), 2)
        self.assertTrue(cell.should_sphere_be_in_cell(sp))

    def test_dim(self):
        cell = Cell((1, 1), [2, 2], (0, 0), [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)])
        self.assertEqual(cell.dim, 2)

    def test_random_generate_spheres(self):
        cell = Cell((0, 0), [4, 4], (0, 0))
        cell.random_generate_spheres(3, 3*[0.5], extra_edges=[1])
        for sphere in cell.spheres:
            self.assertTrue(cell.should_sphere_be_in_cell(sphere))
            z = sphere.center[-1]
            self.assertTrue(z > 0 and z < 1)

    def test_transform(self):
        cell = Cell((0, 0), [4, 4], (0, 0))
        cell.transform((1, 1))
        self.assertEqual(cell.site, (1, 1))


class TestArrayOfCells(TestCase):

    @staticmethod
    def construct_some_arr_cell():
        spheres = []
        bound = CubeBoundaries([2, 2], 2 * [BoundaryType.CYCLIC])
        cell1 = Cell((0, 0), [1, 1], (0, 0))
        cell1.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell1.spheres: spheres.append(sphere)
        cell2 = Cell((1.0, 0), [1, 1], (1, 0))
        cell2.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell2.spheres: spheres.append(sphere)
        cell3 = Cell((0, 1.0), [1, 1], (0, 1))
        cell3.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell3.spheres: spheres.append(sphere)
        cell4 = Cell((1.0, 1.0), [1, 1], (1, 1))
        cell4.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell4.spheres: spheres.append(sphere)
        cells = [cell1, cell2, cell3, cell4]
        return ArrayOfCells(2, bound, cells=[[cell1, cell2], [cell3, cell4]]), spheres, cells

    def test_all_spheres(self):
        arr, spheres, _ = TestArrayOfCells.construct_some_arr_cell()
        self.assertEqual(arr.all_spheres, spheres)

    def test_all_centers(self):
        arr, spheres, _ = TestArrayOfCells.construct_some_arr_cell()
        centers = [sphere.center for sphere in spheres]
        centers2 = arr.all_centers
        for c1, c2 in zip(centers, centers2):
            assert_list(self, c1, c2)

    def test_all_cells(self):
        arr, spheres, cells = TestArrayOfCells.construct_some_arr_cell()
        self.assertEqual(arr.all_cells, cells)

    def test_overlap_2_cells(self):
        while True:
            cell1 = Cell((0, 0), [1, 1], (0,))
            cell1.random_generate_spheres(3, 0.2)
            cell2 = Cell((0, 1), [1, 1], (1,))
            cell2.random_generate_spheres(3, 0.2)
            bound = CubeBoundaries([2, 2], 2 * [BoundaryType.CYCLIC])
            arr = ArrayOfCells(2, bound, [[cell1], [cell2]])
            if not arr.overlap_2_cells(cell1, cell2):
                draw = View2D('test_garb', bound)
                draw.array_of_cells_snapshot('Test overlap 2 cells', arr, 'Test_overlap_2_cells')
                break

    def test_cushioning_array_for_boundary_cond(self):
        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries([2, 2], [BoundaryType.CYCLIC, BoundaryType.WALL])
        cush_arr = arr.cushioning_array_for_boundary_cond()
        vec = np.array([1, 1])
        for cell in cush_arr.all_cells:
            if len(cell.site) > 0:
                cell.transform(cell.site + vec)
        bound = CubeBoundaries([4, 4], arr.boundaries.boundaries_type)
        draw = View2D('test_garb', bound)
        draw.array_of_cells_snapshot('Test Cushioning', cush_arr, 'Test_Cushioning_x_cyclic')

        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries([2, 2], [BoundaryType.WALL, BoundaryType.CYCLIC])
        cush_arr = arr.cushioning_array_for_boundary_cond()
        vec = np.array([1, 1])
        for cell in cush_arr.all_cells:
            if len(cell.site) > 0:
                cell.transform(cell.site + vec)
        bound = CubeBoundaries([4, 4], arr.boundaries.boundaries_type)
        draw = View2D('test_garb', bound)
        draw.array_of_cells_snapshot('Test Cushioning', cush_arr, 'Test_Cushioning_y_cylic')

        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries([2, 2], [BoundaryType.CYCLIC, BoundaryType.CYCLIC])
        cush_arr = arr.cushioning_array_for_boundary_cond()
        vec = np.array([1, 1])
        for cell in cush_arr.all_cells:
            if len(cell.site) > 0:
                cell.transform(cell.site + vec)
        bound = CubeBoundaries([4, 4], arr.boundaries.boundaries_type)
        draw = View2D('test_garb', bound)
        draw.array_of_cells_snapshot('Test Cushioning', cush_arr, 'Test_Cushioning_both_cylic')
        return

    def test_legal_configuration(self):
        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries(arr.boundaries.edges, [BoundaryType.CYCLIC, BoundaryType.WALL])
        arr.random_generate_spheres(5, 0.1)
        draw = View2D('test_garb', arr.boundaries)
        draw.array_of_cells_snapshot('Test Random Generate spheres and legal configuration',
                                     arr, 'TestLegalConfiguration')
        return

    def test_cell_from_ind(self):
        arr, _, cells = TestArrayOfCells.construct_some_arr_cell()
        self.assertEqual(arr.cell_from_ind((0,0)), cells[0])
