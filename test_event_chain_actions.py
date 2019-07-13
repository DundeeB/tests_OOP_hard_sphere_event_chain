from unittest import TestCase
from EventChainActions import *
from Structure import *
from SnapShot import View2D


def assert_list(self, vec1, vec2):
    for v1, v2 in zip(vec1, vec2): self.assertAlmostEqual(v1, v2)


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

    def test_perform_total_step(self):
        for _ in range(200):
            arr = TestEvent2DCells.some_arr()
            cell = arr.all_cells[0]
            sphere = arr.all_spheres[0]
            arr.perform_total_step(cell, sphere, 7, (1, 1, 0))
        pass
