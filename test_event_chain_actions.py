from unittest import TestCase
from EventChainActions import *
from Structure import *
from SnapShot import View2D
import os


def assert_list(self, vec1, vec2):
    for v1, v2 in zip(vec1, vec2): self.assertAlmostEqual(v1, v2, 5)


class TestStep(TestCase):
    @staticmethod
    def some_step():
        sphere = Sphere((0.5, 0.5), 0.1)
        v_hat = np.array([1, 1]) / np.sqrt(2)
        bound = CubeBoundaries([2, 2], [BoundaryType.CYCLIC, BoundaryType.WALL])
        step = Step(sphere, 1, v_hat, bound, current_step=0.1 * np.sqrt(2))
        return step

    def test_perform_step(self):
        step = TestStep.some_step()
        step.perform_step()
        assert_list(self, step.sphere.center, (0.6, 0.6))
        self.assertAlmostEqual(step.total_step, 1-0.1*np.sqrt(2))

    def test_next_event(self):
        step = TestStep.some_step()
        step.v_hat = np.array([2, 1])/np.sqrt(5)
        event, current_step = step.next_event([])
        self.assertAlmostEqual(current_step, 1)
        self.assertEqual(event.event_type, EventType.FREE)

        step.total_step = 100
        step.v_hat = (1, 0)
        event, current_step = step.next_event([])
        self.assertAlmostEqual(current_step, 100)
        self.assertEqual(event.event_type, EventType.FREE)

        step.total_step = 2
        step.v_hat = (0, 1)
        event, current_step = step.next_event([])
        self.assertAlmostEqual(current_step, 1.5-step.sphere.rad)
        self.assertEqual(event.event_type, EventType.WALL)

        other_sphere = Sphere((0.5, 1.5), 0.2)
        event, current_step = step.next_event([other_sphere])
        self.assertAlmostEqual(current_step, 1 - step.sphere.rad - other_sphere.rad)
        self.assertEqual(event.event_type, EventType.COLLISION)
        self.assertEqual(event.other_sphere, other_sphere)


class TestEvent2DCells(TestCase):
    @staticmethod
    def some_arr():
        eff_arr = Event2DCells(1, 3, 3)
        r = 0.3
        eff_arr.add_third_dimension_for_sphere(2*r)
        eff_arr.random_generate_spheres(1, r)
        return eff_arr

    def test_init(self):
        arr = TestEvent2DCells.some_arr()
        draw = View2D('test_garb', arr.boundaries)
        draw.array_of_cells_snapshot('Test init evend 2d cells',
                                     arr, 'TestEvend2dCells')
        return

    def test_closest_site_2d(self):
        arr = TestEvent2DCells.some_arr()
        sphere = Sphere((0.1, 1, 0.3), 0.3)
        cite_ind = arr.closest_site_2d(sphere.center)
        cell = arr.cell_from_ind(cite_ind)
        self.assertEqual(cell.site, (0, 1))

    def test_relevant_cells_around_point_2d(self):
        arr = TestEvent2DCells.some_arr()
        p = (1.2, 1.5, 0.3)
        rad = 0.3
        rel_cells = arr.relevant_cells_around_point_2d(rad, p)
        self.assertEqual(rel_cells, [arr.cells[1][1], arr.cells[1][0]])

    def test_get_all_crossed_points_2d(self):
        arr = TestEvent2DCells.some_arr()
        step = Step(Sphere((0.5, 0.5, 0.3), 0.3), 4, (1, 0, 0), arr.boundaries)
        ts = arr.get_all_crossed_points_2d(step)
        assert_list(self, ts, [0, 0.5, 1.5, 2.5, 3.5, 4])

        step = Step(Sphere((0.5, 0.5, 0.3), 0.3), 4*np.sqrt(2), (1, 1, 0)/np.sqrt(2), arr.boundaries)
        ts = arr.get_all_crossed_points_2d(step)
        assert_list(self, ts, [0, 0.5, 1.5, 2.5, 3.5, 4]/np.sqrt(1/2))

        step = Step(Sphere((0.5, 0.5, 0.3), 0.3), 4, (0, 1, 0), arr.boundaries)
        ts = arr.get_all_crossed_points_2d(step)
        assert_list(self, ts, [0, 0.5, 1.5, 2.5, 3.5, 4])

    def test_cushion_l_x_not_l_y(self):
        arr = Event2DCells(1, 1, 2)
        arr.boundaries.boundaries_type = [BoundaryType.CYCLIC, BoundaryType.WALL]
        r = 0.2
        sphere1 = Sphere((0.5, 0.7), r)
        sphere2 = Sphere((0.1, 0.5), r)
        v_hat = (1, 0)
        total_step = 2

        arr.cells[0][0].add_spheres(sphere1)
        arr.cells[0][0].add_spheres(sphere2)
        output_dir = 'test_garb/2-spheres-cushion'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        step = Step(sphere1, total_step, v_hat, arr.boundaries)
        draw.array_of_cells_snapshot('Without Boundary', arr, 'Without_boundary', step)
        cushioned = arr.cushioning_array_for_boundary_cond()
        for cell in cushioned.all_cells:
            cell.transform(cell.site + np.array([1, 0]))
        draw.boundaries = CubeBoundaries([4, 1], 2*[BoundaryType.WALL])
        draw.array_of_cells_snapshot('Cushioned', cushioned, 'cushion')

    def test_simple_step_2_spheres(self):
        arr = Event2DCells(1, 1, 2)
        r = 0.2
        sphere1 = Sphere((0.5, 0.5), r)
        sphere2 = Sphere((1.5, 0.5), r)
        v_hat = (1, 0)
        total_step = 1.2
        new_loc_1 = (1.1, 0.5)
        new_loc_2 = (0.1, 0.5)

        arr.cells[0][0].add_spheres(sphere1)
        arr.cells[0][1].add_spheres(sphere2)
        output_dir = 'test_garb/2-spheres'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        step = Step(sphere1, total_step, v_hat, arr.boundaries)
        arr.perform_total_step(arr.cells[0][0], step, draw)
        draw.array_of_cells_snapshot('After Step (Searching overlap bug)',
                                     arr, 'After_step')
        self.assertTrue(arr.legal_configuration())
        assert_list(self, sphere1.center, new_loc_1)
        assert_list(self, sphere2.center, new_loc_2)

    def test_collide_through_cyclic_boundary(self):
        arr = Event2DCells(1, 1, 2)
        r = 0.2
        sphere1 = Sphere((0.5, 0.5), r)
        sphere2 = Sphere((0.1, 0.5), r)
        v_hat = (1, 0)
        total_step = 2
        new_loc_1 = (1.7, 0.5)
        new_loc_2 = (0.9, 0.5)

        arr.cells[0][0].add_spheres(sphere1)
        arr.cells[0][0].add_spheres(sphere2)
        output_dir = 'test_garb/2-spheres-cyclic'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        step = Step(sphere1, total_step, v_hat, arr.boundaries)
        arr.perform_total_step(arr.cells[0][0], step, draw)
        draw.array_of_cells_snapshot('After Step (Searching overlap bug)',
                                     arr, 'After_step', )
        assert_list(self, sphere1.center, new_loc_1)
        assert_list(self, sphere2.center, new_loc_2)

    def test_collide_through_cyclic_boundary_3_spheres(self):
        arr = Event2DCells(1, 3, 3)
        r = 0.3
        sphere1 = Sphere((0.99, 2.7), r)
        sphere2 = Sphere((1.2, 0.4), r)
        sphere3 = Sphere((1.8, 0.6), r)
        v_hat = (1, 1)/np.sqrt(2)
        total_step = 5

        arr.cells[2][0].add_spheres(sphere1)
        arr.cells[0][1].add_spheres(sphere2)
        arr.cells[0][1].add_spheres(sphere3)
        output_dir = 'test_garb/3-spheres-cyclic'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        step = Step(sphere3, total_step, v_hat, arr.boundaries)
        arr.perform_total_step(arr.cells[0][1], step, draw)
        draw.array_of_cells_snapshot('After Step (Searching overlap bug)',
                                     arr, 'After_step', )
        self.assertTrue(arr.legal_configuration())

    def test_generate_spheres_save_pic(self):
        arr = TestEvent2DCells.some_arr()
        cell = arr.all_cells[0]
        sphere = arr.all_spheres[0]
        v_hat = (1, 1, 0)/np.sqrt(2)
        step = Step(sphere, 7, v_hat, arr.boundaries)

        output_dir = 'test_garb/random_spheres'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        arr.perform_total_step(cell, step, draw)
        draw.array_of_cells_snapshot('After Step (Searching overlap bug)',
                                     arr, 'After_step', step)
        self.assertTrue(arr.legal_configuration())

    def test_generate_spheres_many_times_perform_large_step(self):
        for i in range(100):
            arr = TestEvent2DCells.some_arr()
            cell = arr.all_cells[0]
            sphere = arr.all_spheres[0]
            v_hat = (1, 1, 0)/np.sqrt(2)
            step = Step(sphere, 7, v_hat, arr.boundaries)
            arr.perform_total_step(cell, step)
        pass
