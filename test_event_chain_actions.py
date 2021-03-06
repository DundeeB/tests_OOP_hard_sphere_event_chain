from unittest import TestCase
from EventChainActions import *
from Structure import *
from SnapShot import View2D
import os, shutil, random
epsilon = 1e-8

garb = '../simulation-results/ECMC_tests'
if os.path.exists(garb):
    shutil.rmtree(garb)
os.mkdir(garb)


def assert_list(self, vec1, vec2):
    for v1, v2 in zip(vec1, vec2): self.assertAlmostEqual(v1, v2, 5)


class TestStep(TestCase):
    @staticmethod
    def some_step():
        sphere = Sphere((0.5, 0.5), 0.1)
        v_hat = np.array([1, 1]) / np.sqrt(2)
        bound = CubeBoundaries([2, 2], [BoundaryType.CYCLIC, BoundaryType.WALL])
        step = Step(sphere, 1, v_hat, bound, current_step = 0.1 * np.sqrt(2))
        return step

    def test_perform_step(self):
        step = TestStep.some_step()
        step.perform_step()
        assert_list(self, step.sphere.center, (0.6, 0.6))
        self.assertAlmostEqual(step.total_step, 1-0.1*np.sqrt(2))

    def test_next_event(self):
        step = TestStep.some_step()
        step.current_step = np.nan
        step.v_hat = np.array([2, 1])/np.sqrt(5)
        event, current_step = step.next_event([])
        self.assertAlmostEqual(current_step, step.current_step)
        self.assertEqual(event.event_type, EventType.FREE)

        step.total_step = 100
        step.current_step = np.nan
        step.v_hat = (1, 0)
        event, current_step = step.next_event([])
        self.assertAlmostEqual(current_step, 100)
        self.assertEqual(event.event_type, EventType.FREE)

        step.total_step = 2
        step.current_step = np.nan
        step.v_hat = (0, 1)
        event, current_step = step.next_event([])
        self.assertAlmostEqual(current_step, 1.5-step.sphere.rad)
        self.assertEqual(event.event_type, EventType.WALL)

        other_sphere = Sphere((0.5, 1.5), 0.2)
        step.current_step = np.nan
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

    def three_spheres_test(self, sphere1, sphere2, sphere3, output_dir, total_step=7):
        arr = Event2DCells(1, 3, 3)
        sphere = sphere1
        spheres = [sphere1, sphere2, sphere3]
        spheres_added = []
        for s in spheres:
            for i in [0, 1, 2]:
                for j in [0, 1, 2]:
                    if arr.cells[i][j].center_in_cell(s) and s not in spheres_added:
                        arr.cells[i][j].append(s)
                        spheres_added.append(s)
                        if s == sphere1:
                            cell = arr.cells[i][j]
                            i_cell, j_cell = i, j

        v_hat = (1, 1) / np.sqrt(2)
        step = Step(sphere, total_step, v_hat, arr.boundaries)

        self.assertTrue(arr.legal_configuration())
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        arr.perform_total_step(i_cell, j_cell, step, draw)
        draw.array_of_cells_snapshot('After Step (Searching direct_overlap bug)',
                                     arr, 'After_step', step)
        self.assertTrue(arr.legal_configuration())

    @staticmethod
    def track_step(arr_before, output_dir, i_cell, i_sphere, v_hat, total_step=7):
        assert arr_before.legal_configuration()
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        draw = View2D(output_dir, arr_before.boundaries)
        cell = arr_before.all_cells[i_cell]
        sphere = cell.spheres[i_sphere]
        step = Step(sphere, 7, v_hat, arr_before.boundaries)
        i, j = cell.ind[:2]
        draw.dump_spheres(arr_before.all_centers, 'Before')
        arr_before.perform_total_step(i, j, step, draw)
        raise

    def test_init(self):
        arr = TestEvent2DCells.some_arr()
        draw = View2D(garb, arr.boundaries)
        draw.array_of_cells_snapshot('Test init evend 2d cells',
                                     arr, 'TestEvend2dCells')
        return

    def test_closest_site_2d(self):
        arr = TestEvent2DCells.some_arr()
        sphere = Sphere((0.1, 1, 0.3), 0.3)
        cite_ind = arr.closest_site_2d(sphere.center)
        cell = arr.cell_from_ind(cite_ind)
        self.assertEqual(cell.site, (0, 1))

    def test_cushion_l_x_not_l_y(self):
        arr = Event2DCells(1, 1, 2)
        arr.boundaries.boundaries_type = [BoundaryType.CYCLIC, BoundaryType.WALL]
        r = 0.2
        sphere1 = Sphere((0.5, 0.7), r)
        sphere2 = Sphere((0.1, 0.5), r)
        v_hat = (1, 0)
        total_step = 2

        arr.cells[0][0].append(sphere1)
        arr.cells[0][0].append(sphere2)
        output_dir = garb + '/2-spheres-cushion'
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

        arr.cells[0][0].append(sphere1)
        arr.cells[0][1].append(sphere2)
        output_dir = garb + '/2-spheres'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        step = Step(sphere1, total_step, v_hat, arr.boundaries)
        arr.perform_total_step(0, 0, step, draw)
        draw.array_of_cells_snapshot('After Step (Searching direct_overlap bug)',
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

        arr.cells[0][0].append(sphere1)
        arr.cells[0][0].append(sphere2)
        output_dir = garb + '/2-spheres-cyclic'
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        step = Step(sphere1, total_step, v_hat, arr.boundaries)
        arr.perform_total_step(0, 0, step, draw)
        draw.array_of_cells_snapshot('After Step (Searching direct_overlap bug)',
                                     arr, 'After_step', )
        assert_list(self, sphere1.center, new_loc_1)
        assert_list(self, sphere2.center, new_loc_2)

    def test_collide_through_cyclic_boundary_3_spheres(self):
        r = 0.3
        sphere1 = Sphere((0.99, 2.7), r)
        sphere2 = Sphere((1.2, 0.4), r)
        sphere3 = Sphere((1.8, 0.6), r)
        output_dir = garb + '/3-spheres-cyclic'
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir)

    def test_generate_spheres_save_pic(self):
        sphere1 = Sphere((1.10, 0.50), 0.3)
        sphere2 = Sphere((2.01, 0.80), 0.3)
        sphere3 = Sphere((2.20, 1.75), 0.3)
        output_dir = garb + '/random_spheres'
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir)

    def test_around_ps_should_be_center_off_sphere(self):
        sphere1 = Sphere((1.4, 2.01), 0.3)
        sphere2 = Sphere((2.01, 1.99), 0.3)
        sphere3 = Sphere((2.5, 2.5), 0.3)
        output_dir = garb + '/ps_should_be_center_off_sphere'
        total_step = 0.3 * np.sqrt(2)
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir, total_step)

    def test_up_boundary_left(self):
        sphere1 = Sphere((0.05, 0.8), 0.3)
        sphere2 = Sphere((1.05, 1.55), 0.3)
        sphere3 = Sphere((2.99, 1.4), 0.3)
        output_dir = garb + '/up_boundary_left'
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir)

    def test_up_boundary_right(self):
        sphere1 = Sphere((0.5, 1.29), 0.3)
        sphere2 = Sphere((0.29, 0.7), 0.3)
        sphere3 = Sphere((2.9, 1.4), 0.3)
        output_dir = garb + '/up_boundary_right'
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir)

    def test_large_to_up_right(self):
        sphere1 = Sphere((1.01, 1.01), 0.3)
        sphere2 = Sphere((2.35, 2.85), 0.3)
        sphere3 = Sphere((2.9, 2.6), 0.3)
        output_dir = garb + '/large_to_up_right'
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir)

    def test_on_boundaries(self):
        sphere1 = Sphere((0.10, 1.05), 0.3)
        sphere2 = Sphere((1.80, 2.95), 0.3)
        sphere3 = Sphere((2.00, 0.60), 0.3)
        output_dir = garb + '/on_boundaries'
        self.three_spheres_test(sphere1, sphere2, sphere3, output_dir)

    def test_free_step_overlap(self):
        sphere1 = Sphere((2.1, 0.9), 0.3)
        sphere2 = Sphere((0.8, 0.5), 0.3)
        output_dir = garb + '/free_step_over_lap'
        arr = Event2DCells(1, 5, 5)
        for s in [sphere1, sphere2]:
            for i in range(len(arr.cells)):
                for j in range(len(arr.cells[i])):
                    if arr.cells[i][j].center_in_cell(s):
                        arr.cells[i][j].append(s)
                        if s == sphere1:
                            cell = arr.cells[i][j]
                            i_cell, j_cell = i, j
        final_pos = np.array([1.05, 0.5])
        vec = final_pos - sphere1.center
        total_step = np.linalg.norm(vec)
        v_hat = vec / total_step
        step = Step(sphere1, total_step, v_hat, arr.boundaries)

        self.assertTrue(arr.legal_configuration())
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        arr.perform_total_step(i_cell, j_cell, step, draw)
        draw.array_of_cells_snapshot('After Step (Searching direct_overlap bug)',
                                     arr, 'After_step', step)
        self.assertTrue(arr.legal_configuration())

    def test_generate_spheres_in_cubic_structure(self):
        arr = Event2DCells(0.6, 10, 10)
        arr.generate_spheres_in_cubic_structure(1, 0.3)
        output_dir = garb
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('cubic structure', arr, 'cubic_struct')

    def test_generate_spheres_many_times_perform_large_step(self):
        for i in range(100):
            arr = TestEvent2DCells.some_arr()
            cell = arr.cells[0][0]
            sphere = cell.spheres[0]
            v_hat = (1, 1, 0)/np.sqrt(2)
            step = Step(sphere, 7, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                arr.perform_total_step(0, 0, step)
            except:
                output_dir = garb + '/many_random_spheres'
                TestEvent2DCells.track_step(temp_arr, output_dir, 0, 0, v_hat)
                raise
        pass

    def test_spheres_do_many_steps(self):
        arr = TestEvent2DCells.some_arr()
        for i in range(10):
            while True:
                i_cell = random.randint(0, len(arr.all_cells) - 1)
                cell = arr.all_cells[i_cell]
                if len(cell.spheres) > 0:
                    break
            i_sphere = random.randint(0, len(cell.spheres) - 1)
            sphere = cell.spheres[i_sphere]
            v_hat = (1, 1, 0)/np.sqrt(2)
            step = Step(sphere, 7, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                i, j = cell.ind[:2]
                arr.perform_total_step(i, j, step)
                assert arr.legal_configuration()
            except:
                output_dir = garb + '/many_steps'
                TestEvent2DCells.track_step(temp_arr, output_dir, i_cell, i_sphere, v_hat)
                raise
        pass

    def test_steps_in_random_directions(self):
        arr = Event2DCells(1, 6, 5)
        arr.generate_spheres_in_cubic_structure(2, 0.2)
        for i in range(50):
            while True:
                i_cell = random.randint(0, len(arr.all_cells) - 1)
                cell = arr.all_cells[i_cell]
                if len(cell.spheres) > 0:
                    break
            i_sphere = random.randint(0, len(cell.spheres) - 1)
            sphere = cell.spheres[i_sphere]
            t = random.random()
            v_hat = (np.cos(t), np.sin(t))
            step = Step(sphere, 7, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                i, j = cell.ind[:2]
                arr.perform_total_step(i, j, step)
                assert arr.legal_configuration()
            except:
                output_dir = garb + '/steps_in_random_directions'
                TestEvent2DCells.track_step(temp_arr, output_dir, i_cell, i_sphere, v_hat)
                raise
        pass

    def test_3d_1_sphere(self):
        r = 0.1
        arr = Event2DCells(1, 1, 1)
        arr.add_third_dimension_for_sphere(2)
        total_step = 0.5*np.sqrt(2)
        output_dir = garb + '/3d_1_sphere'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        i, j = 0, 0
        cell = arr.cells[i][j]
        sphere = Sphere((epsilon, epsilon, 0.15), r)
        cell.append(sphere)
        v_hat = np.array([0, 1, 1])/np.sqrt(2)
        step = Step(sphere, total_step, v_hat, arr.boundaries)
        temp_arr = copy.deepcopy(arr)
        arr.perform_total_step(i, j, step)
        assert temp_arr.legal_configuration()
        assert_list(self, sphere.center, (0, 0.5, 0.65))

    def test_3d_2_sphere(self):
        r = 0.1 * np.sqrt(2)
        arr = Event2DCells(2, 1, 1)
        arr.add_third_dimension_for_sphere(0.9 + r)
        total_step = (0.1 + 0.4 + 0.1) * np.sqrt(2)
        output_dir = garb + '/3d_1_sphere'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        cell = arr.all_cells[0]
        x1 = 0.2
        sphere = Sphere((epsilon, x1, x1), r)
        x2 = 0.5
        sphere2 = Sphere((epsilon, x2, x2), r)
        cell.append([sphere, sphere2])

        assert arr.legal_configuration()

        v_hat = np.array([0, 1, 1])/np.sqrt(2)
        step = Step(sphere, total_step, v_hat, arr.boundaries)
        i, j = cell.ind[:2]
        arr.perform_total_step(i, j, step)
        assert arr.legal_configuration()
        assert_list(self, sphere.center, (0, x1 + 0.1, x1 + 0.1))
        assert_list(self, sphere2.center, (0, 1, 0.8))

    def test_cubic_comb_transition(self):
        n_row = 40
        n_col = n_row
        n_per_cell = 1
        rad = 0.48  # eta=0.72 > 0.7 so should be solid
        N = n_per_cell * n_row * n_col
        print('Number of disks: ' + str(N))
        print('eta= ' + str(n_per_cell * np.pi * rad ** 2))

        arr = Event2DCells(1, n_row, n_col)
        arr.generate_spheres_in_cubic_structure(n_per_cell, rad)
        total_step = np.sqrt(n_row)
        output_dir = garb + '/cubic_comb_transition'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Before run', arr, '0')
        N_iteration = 2*n_row  # **2 not implemented for faster simulation
        for i in range(N_iteration):
            while True:
                i_cell = random.randint(0, len(arr.all_cells) - 1)
                cell = arr.all_cells[i_cell]
                if len(cell.spheres) > 0:
                    break
            i_sphere = random.randint(0, len(cell.spheres) - 1)
            sphere = cell.spheres[i_sphere]
            # v_hat = [-1, -1] + 2*np.random.random(2)
            # v_hat = v_hat/np.linalg.norm(v_hat)
            if i%2:
                v_hat = (1, 0)
            else:
                v_hat = (0, 1)
            v_hat = np.array(v_hat)
            step = Step(sphere, total_step, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                if i % n_row == n_row-1:  # i == 0 or i==1000:  #
                    draw.array_of_cells_snapshot(str(i + 1),
                                                 arr, str(i + 1).zfill(int(np.floor(np.log10(N_iteration)) + 1)))
                i, j = cell.ind[:2]
                arr.perform_total_step(i, j, step)
            except Exception as e:
                print(e)
                draw.array_of_cells_snapshot('Most recent image',
                                             arr, 'Most_recent_img', step)
                output_dir += '/Bug!'
                TestEvent2DCells.track_step(temp_arr, output_dir, i_cell, i_sphere, v_hat)
                raise
        draw.save_video("cubic_comb_transition", fps=6)
        pass

    def test_cubic_comb_2_species_transition(self):
        n_row = 3
        n_col = n_row
        n_per_cell = 5
        r1 = 0.35
        r2 = 0.15
        rads = [r1] + (n_per_cell - 1)*[r2]
        N = n_per_cell * n_row * n_col
        print('Number of disks: ' + str(N))
        print('eta= ' + str(np.pi * r1 ** 2 + (n_per_cell-1) * np.pi * r2 ** 2))

        arr = Event2DCells(1, n_row, n_col)
        arr.generate_spheres_in_cubic_structure(n_per_cell, rads)
        total_step = 2
        output_dir = garb + '/cubic_comb_2_species_transition'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        draw = View2D(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Before run', arr, '0')
        for i in range(N):
            while True:
                i_cell = random.randint(0, len(arr.all_cells) - 1)
                cell = arr.all_cells[i_cell]
                if len(cell.spheres) > 0:
                    break
            i_sphere = random.randint(0, len(cell.spheres) - 1)
            sphere = cell.spheres[i_sphere]
            if i % 2:
                v_hat = [1, 0]
            else:
                v_hat = [0, 1]
            v_hat = np.array(v_hat)
            step = Step(sphere, total_step, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                draw.array_of_cells_snapshot(str(i + 1),
                                             arr, str(i + 1).zfill(int(np.floor(np.log10(200)) + 1)))
                i, j = cell.ind[:2]
                arr.perform_total_step(i, j, step)
            except:
                draw.array_of_cells_snapshot('Most recent image',
                                             arr, 'Most_recent_img', step)
                output_dir += '/Bug!'
                TestEvent2DCells.track_step(temp_arr, output_dir, i_cell, i_sphere, v_hat, total_step)
                raise
        draw.save_video("cubic_comb_2_species_transition", fps=6)
        pass

    def test_3D_rhoH_N_h_structure(self):
        # Input
        rho_H = 0.45
        h = 1
        n_row = 15
        n_col = 15
        N_iteration = 100 # not implemented for faster simulation
        n_sp_per_cell = 1
        # More physical properties calculated from Input
        N = n_row*n_col*n_sp_per_cell
        r = 1
        sig = 2*r
        H = (h+1)*sig
        a = sig*np.sqrt(1/(rho_H*(1+h)))  # rho_H = N*sig^3/(H*A), A = N*a^2
        e = a*np.sqrt(n_sp_per_cell)
        # Folder Handeling
        sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_square_ECMC'
        output_dir = garb + '/' + sim_name
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        # construct array of cells and fill with spheres
        arr = Event2DCells(edge=e, n_rows=n_row, n_columns=n_col)
        arr.add_third_dimension_for_sphere(H)
        arr.generate_spheres_in_cubic_structure(n_spheres_per_cell=n_sp_per_cell, rad=r)
        total_step = np.sqrt(n_row) * a
        # Initialize View
        draw = View2D(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Before run', arr, '0')
        for i in range(N_iteration):
            while True:
                i_cell = random.randint(0, len(arr.all_cells) - 1)
                cell = arr.all_cells[i_cell]
                if len(cell.spheres) > 0:
                    break
            i_sphere = random.randint(0, len(cell.spheres) - 1)
            sphere = cell.spheres[i_sphere]
            if i % 4 == 0:
                v_hat = (1, 0, 1)
            if i % 4 == 1:
                v_hat = (0, 1, 1)
            if i % 4 == 2:
                v_hat = (1, 0, -1)
            else:
                v_hat = (0, 1, -1)
            v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
            step = Step(sphere, total_step, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                if i % n_row == n_row - 1:  # i == 0 or i==1000:  #
                    draw.array_of_cells_snapshot(str(i + 1),
                                                 arr, str(i + 1).zfill(int(np.floor(np.log10(N_iteration)) + 1)))
                    draw.dump_spheres(arr.all_centers, str(i + 1))
                i_cell, j_cell = cell.ind[:2]
                arr.perform_total_step(i_cell, j_cell, step)
                assert arr.legal_configuration()  # tbd remove it to speed up simulation
            except Exception as e:
                print(e)
                draw.array_of_cells_snapshot('Most recent image',
                                             arr, 'Most_recent_img', step)
                output_dir += '/Bug'
                TestEvent2DCells.track_step(temp_arr, output_dir, i_cell, i_sphere, v_hat)
                raise
        draw.save_video("2D_image_of_3D_spheres", fps=6)
        pass

    def test_3D_rhoH_N_h_AF_structure(self):
        # Input
        rho_H = 0.9  # closest rho_H, the code will generate it with different but close value
        h = 1
        n_row = 25
        n_col = 9
        N_iteration = 100  # not implemented for faster simulation
        n_sp_per_dim_per_cell = 1
        # More physical properties calculated from Input
        N = n_row*n_col
        r = 1
        sig = 2*r
        H = (h+1)*sig
        # build input parameters for cells
        a_dest = sig*np.sqrt(2/(rho_H*(1+h)*np.sin(np.pi/3)))
        l_y_dest = a_dest * n_row/2 * np.sin(np.pi/3)
        e = n_sp_per_dim_per_cell*a_dest
        n_col_cells = int(n_col/n_sp_per_dim_per_cell)
        n_row_cells = int(round(l_y_dest/e))
        l_x = n_col_cells * e
        l_y = n_row_cells * e

        a = np.sqrt(l_x*l_y/N)  # rho_H = N*sig^3/(H*A), A = N*a^2
        rho_H = (sig**2)/((a**2)*(h+1))
        print("New rho_H chosen: " + str(rho_H))
        # Folder Handeling
        sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_ECMC'
        output_dir = garb + '/' + sim_name
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        # construct array of cells and fill with spheres
        arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells)
        arr.add_third_dimension_for_sphere(H)
        arr.generate_spheres_in_AF_triangular_structure(n_row, n_col, r)
        total_step = np.sqrt(n_row) * a
        # Initialize View
        draw = View2D(output_dir, arr.boundaries)
        draw.array_of_cells_snapshot('Before run', arr, '0')
        for i in range(N_iteration):
            while True:
                i_cell = random.randint(0, len(arr.all_cells) - 1)
                cell = arr.all_cells[i_cell]
                if len(cell.spheres) > 0:
                    break
            i_sphere = random.randint(0, len(cell.spheres) - 1)
            sphere = cell.spheres[i_sphere]
            if i % 4 == 0:
                v_hat = (1, 0, 1)
            if i % 4 == 1:
                v_hat = (0, 1, 1)
            if i % 4 == 2:
                v_hat = (1, 0, -1)
            else:
                v_hat = (0, 1, -1)
            v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
            step = Step(sphere, total_step, v_hat, arr.boundaries)
            temp_arr = copy.deepcopy(arr)
            try:
                if i % n_row == n_row - 1:  # i == 0 or i==1000:  #
                    draw.array_of_cells_snapshot(str(i + 1),
                                                 arr, str(i + 1).zfill(int(np.floor(np.log10(N_iteration)) + 1)))
                    draw.dump_spheres(arr.all_centers, str(i + 1))
                i_cell, j_cell = cell.ind[:2]
                arr.perform_total_step(i_cell, j_cell, step)
                assert arr.legal_configuration()  # tbd remove it to speed up simulation
            except Exception as e:
                print(e)
                draw.array_of_cells_snapshot('Most recent image',
                                             arr, 'Most_recent_img', step)
                output_dir += '/Bug'
                TestEvent2DCells.track_step(temp_arr, output_dir, i_cell, i_sphere, v_hat)
                raise
        draw.save_video("2D_image_of_3D_spheres", fps=6)
        pass
