from unittest import TestCase
from Structure import *
import os, numpy as np
from SnapShot import WriteOrLoad

output_dir = '../simulation-results/ECMC_tests'
if not os.path.isdir(output_dir): os.mkdir(output_dir)


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

    def test_box_it(self):
        boundaries = CubeBoundaries([1, 1], [BoundaryType.CYCLIC for _ in range(2)])
        sphere = Sphere((1.5, 1.5), 0.3)
        sphere.box_it(boundaries)
        self.assertGreaterEqual(sphere.center[0], 0)
        self.assertGreaterEqual(sphere.center[1], 0)
        self.assertLessEqual(sphere.center[0], 1)
        self.assertLessEqual(sphere.center[1], 1)

    def test_perform_step(self):
        boundaries = CubeBoundaries([1, 1, 10], [BoundaryType.CYCLIC for _ in range(3)])
        sphere = Sphere((0.5, 0.5, 0), 0.3)
        direction = Direction(0)
        sphere.perform_step(direction, 0.6, boundaries)
        self.assertAlmostEqual(sphere.center[0], 0.1)
        self.assertAlmostEqual(sphere.center[1], 0.5)


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


class TestMetric(TestCase):
    def test_dist_to_boundary(self):
        bound = CubeBoundaries([1, 2, 3], [BoundaryType.CYCLIC, BoundaryType.CYCLIC, BoundaryType.WALL])
        sphere = Sphere((0.5, 0.5, 0.5), 0.1)
        direction = Direction(2)
        dist_to_hit = Metric.dist_to_wall(sphere, 3, direction, bound)
        self.assertAlmostEqual(dist_to_hit, 3 - 0.5 - 0.1)

    def test_dist_to_collision(self):
        sphere1 = Sphere((0.5, 1, 1), 0.3)
        sphere2 = Sphere((0.5, 2, 1), 0.3)
        diam = sphere1.rad + sphere2.rad
        direction = Direction(1)
        bound = CubeBoundaries([3, 3, 5], 3*[BoundaryType.WALL])
        d1 = Metric.dist_to_collision(sphere1, sphere2, 10, direction, bound)
        self.assertAlmostEqual(d1, 1 - diam)
        d1 = Metric.dist_to_collision(sphere1, sphere2, 0.1, direction, bound)
        self.assertEqual(d1, float('inf'))


class TestCell(TestCase):
    def test_add_spheres(self):
        cell = Cell((1, 1), [2, 2], (0, 0), [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)])
        sp = Sphere((0, 1), 2)
        cell.append(sp)
        self.assertEqual(cell.spheres[-1], sp)

    def test_remove_sphere(self):
        sp = Sphere((0, 1), 2)
        other_spheres = [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)]
        cell = Cell((1, 1), [2, 2], (0, 0), other_spheres + [sp])
        cell.remove_sphere(sp)
        self.assertEqual(cell.spheres, other_spheres)

    def test_center_in_cell(self):
        cell = Cell((1, 1), [2, 2], (0, 0), [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)])
        sp = Sphere((0, 1, 2), 2)
        self.assertFalse(cell.center_in_cell(sp))
        sp = Sphere((1.5, 1.5, 2), 2)
        self.assertTrue(cell.center_in_cell(sp))

    def test_dim(self):
        cell = Cell((1, 1), [2, 2], (0, 0), [Sphere((0, 0, 0), 1), Sphere((3, 3, 3), 3)])
        self.assertEqual(cell.dim, 2)

    def test_random_generate_spheres(self):
        cell = Cell((0, 0), [4, 4], (0, 0))
        cell.random_generate_spheres(3, 3 * [0.5], extra_edges=[1])
        for sphere in cell.spheres:
            self.assertTrue(cell.center_in_cell(sphere))
            z = sphere.center[-1]
            self.assertTrue(z > 0 and z < 1)

    def test_transform(self):
        cell = Cell((0, 0), [4, 4], (0, 0), [Sphere((0, 1, 7), 1)])
        cell.transform((1, 1))
        self.assertEqual(cell.site, (1, 1))
        assert_list(self, cell.spheres[0].center, (1, 2, 7))


class TestArrayOfCells(TestCase):

    @staticmethod
    def construct_some_arr_cell():
        spheres = []
        bound = CubeBoundaries([3, 3], 2 * [BoundaryType.CYCLIC])
        cell1 = Cell((0, 0), [1, 1], (0, 0))
        cell1.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell1.spheres: spheres.append(sphere)
        cell2 = Cell((1.0, 0), [1, 1], (1, 0))
        cell2.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell2.spheres: spheres.append(sphere)
        cell3 = Cell((2.0, 0), [1, 1], (0, 1))
        cell3.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell3.spheres: spheres.append(sphere)
        cell4 = Cell((0.0, 1.0), [1, 1], (1, 1))
        cell4.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell4.spheres: spheres.append(sphere)
        cell5 = Cell((1.0, 1.0), [1, 1], (0, 0))
        cell5.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell5.spheres: spheres.append(sphere)
        cell6 = Cell((2.0, 1.0), [1, 1], (0, 0))
        cell6.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell6.spheres: spheres.append(sphere)
        cell7 = Cell((0.0, 2.0), [1, 1], (0, 0))
        cell7.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell7.spheres: spheres.append(sphere)
        cell8 = Cell((1.0, 2.0), [1, 1], (0, 0))
        cell8.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell8.spheres: spheres.append(sphere)
        cell9 = Cell((2.0, 2.0), [1, 1], (0, 0))
        cell9.random_generate_spheres(3, 3 * [0.1])
        for sphere in cell9.spheres: spheres.append(sphere)

        cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]
        return ArrayOfCells(2, bound, cells=[[cell1, cell2, cell3], [cell4, cell5, cell6],
                                             [cell7, cell8, cell9]]), spheres, cells

    def test_draw_nominal_arr(self):
        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        draw = WriteOrLoad(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Nominal', arr, 'Nominal')

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
                draw = WriteOrLoad(output_dir, bound)
                draw.array_of_cells_snapshot('Test direct_overlap 2 cells', arr, 'Test_overlap_2_cells')
                break

        cell = Cell((0, 0), [1, 1], (0, 0), [Sphere((0.1, 0.5), 0.2)])
        neighbor = Cell((0, 1), [1, 1], (0, 0), [Sphere((0.1, 1.5), 0.2)])
        arr = ArrayOfCells(2, CubeBoundaries([2, 1], 2 * [BoundaryType.CYCLIC]), [[cell], [neighbor]])
        self.assertTrue(arr.overlap_2_cells(cell, neighbor))

    def test_cushioning_array_for_boundary_cond(self):
        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries([3, 3], [BoundaryType.CYCLIC, BoundaryType.WALL])
        cush_arr = arr.cushioning_array_for_boundary_cond()
        vec = np.array([1, 1])
        for cell in cush_arr.all_cells:
            if len(cell.site) > 0:
                cell.transform(cell.site + vec)
        bound = CubeBoundaries([5, 5], arr.boundaries.boundaries_type)
        draw = WriteOrLoad(output_dir, bound)
        draw.array_of_cells_snapshot('Test Cushioning', cush_arr, 'Test_Cushioning_x_cyclic')

        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries([3, 3], [BoundaryType.WALL, BoundaryType.CYCLIC])
        cush_arr = arr.cushioning_array_for_boundary_cond()
        vec = np.array([1, 1])
        for cell in cush_arr.all_cells:
            if len(cell.site) > 0:
                cell.transform(cell.site + vec)
        bound = CubeBoundaries([5, 5], arr.boundaries.boundaries_type)
        draw = WriteOrLoad(output_dir, bound)
        draw.array_of_cells_snapshot('Test Cushioning', cush_arr, 'Test_Cushioning_y_cylic')

        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries([3, 3], [BoundaryType.CYCLIC, BoundaryType.CYCLIC])
        cush_arr = arr.cushioning_array_for_boundary_cond()
        vec = np.array([1, 1])
        for cell in cush_arr.all_cells:
            if len(cell.site) > 0:
                cell.transform(cell.site + vec)
        bound = CubeBoundaries([5, 5], arr.boundaries.boundaries_type)
        draw = WriteOrLoad(output_dir, bound)
        draw.array_of_cells_snapshot('Test Cushioning', cush_arr, 'Test_Cushioning_both_cylic')
        return

    def test_legal_configuration(self):
        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        arr.boundaries = CubeBoundaries(arr.boundaries.edges, [BoundaryType.CYCLIC, BoundaryType.WALL])
        arr.random_generate_spheres(3, 0.1)
        draw = WriteOrLoad(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Test Random Generate spheres and legal configuration',
                                     arr, 'TestLegalConfiguration')
        return

    def test_rows_different_columns(self):
        arr, _, _ = TestArrayOfCells.construct_some_arr_cell()
        cells = [[[] for _ in range(len(arr.cells) - 1)] for _ in range(len(arr.cells[0]))]
        for i in range(len(cells)):
            for j in range(len(cells[i])):
                cells[i][j] = arr.cells[i][j]
        arr = ArrayOfCells(arr.dim, arr.boundaries, cells)
        arr.boundaries = CubeBoundaries(arr.boundaries.edges + np.array([-1, 0]), arr.boundaries.boundaries_type)
        draw = WriteOrLoad(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Test rows differ columns',
                                     arr, 'TestRowsDiffCol')

    def test_cell_from_ind(self):
        arr, _, cells = TestArrayOfCells.construct_some_arr_cell()
        self.assertEqual(arr.cell_from_ind((0, 0)), cells[0])
