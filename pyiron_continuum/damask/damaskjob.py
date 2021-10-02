from pyiron_base import TemplateJob, GenericJob, DataContainer
import numpy as np
import matplotlib.pyplot as plt
from damask import Grid, Result, Config, ConfigMaterial, seeds
import pyvista as pv
import h5py
import os


class DAMASK(GenericJob):
    def __init__(self, project, job_name):
        super(DAMASK, self).__init__(project, job_name)
        #self.working_directory = os.path.join(project.path, job_name)
        # if not os.path.exists(self.working_directory):
        #     os.mkdir(self.working_directory)
        self.input = DataContainer()
        self.output = DataContainer()
        self._damask_hdf = os.path.join(self.working_directory, "damask_tensionX.hdf5")
        self.create = Create()
        self._material = None
        self._loading = None
        self._grid = None
        self._results = None
        self._executable_activate()

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        self._material = value

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value

    def _write_material(self):
        file_path = os.path.join(self.working_directory, "material.yaml")
        self._material.save(fname=file_path)
        self.input.material.read(file_path)

    def _write_loading(self):
        file_path = os.path.join(self.working_directory, "loading.yaml")
        self._loading.save(file_path)
        self.input.loading.read(file_path)

    def _write_geometry(self):
        file_path = os.path.join(self.working_directory, "damask")
        self._grid.save(file_path)
        self.input.geometry.read(file_path)

    def write_input(self):
        os.chdir(self.working_directory)
        self._write_loading()
        self._write_geometry()
        self._write_material()

    def collect_output(self):
        self._load_results()
        self._stress()
        self._strain()

    def _load_results(self, file_name="damask_tensionX.hdf5"):
        """
        Open ‘damask_tensionX.hdf5’,add the Mises equivalent of the Cauchy stress, and export it to VTK (file).
        """
        if self._results is None:
            if file_name != "damask_tensionX.hdf5":
                self._damask_hdf = os.path.join(self.working_directory, file_name)
            self._results = Result(self._damask_hdf)
        return self._results

    def _stress(self):
        """
        return the stress as a numpy array
        """
        if self._results is not None:
            stress_path = self._results.get_dataset_location('avg_sigma')
            stress = np.zeros(len(stress_path))
            hdf = h5py.File(self._results.fname)
            for count, path in enumerate(stress_path):
                stress[count] = np.array(hdf[path])
            self.output.stress = np.array(stress)/1E6

    def _strain(self):
        """
        return the strain as a numpy array
        Parameters
        ----------
        job_file : str
          Name of the job_file
        """
        if self._damask_results is not None:
            stress_path = self._damask_results.get_dataset_location('avg_sigma')
            strain = np.zeros(len(stress_path))
            hdf = h5py.File(self._damask_results.fname)
            for count,path in enumerate(stress_path):
                strain[count] = np.array(hdf[path.split('avg_sigma')[0]+ 'avg_epsilon'])
            self.output.strain = strain
    
    def plot_stress_strain(self, ax=None):
        """
        Plot the stress strain curve from the job file
        Parameters
        ----------
        ax (matplotlib axis /None): axis to plot on (created if None)
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.output.strain, self.output.stress, linestyle='-', linewidth='2.5')
        ax.grid(True)
        ax.set_xlabel(r'$\varepsilon_{VM} $', fontsize=18)
        ax.set_ylabel(r'$\sigma_{VM}$ (MPa)', fontsize=18)
        return fig, ax
    
    def load_mesh(self, inc=20):
        """
        Return the mesh for particular increment
        """
        mesh = pv.read(os.path.join(self.working_directory, self._file_name.split('.')[0] + f'_inc0{inc}.vtr'))
        return mesh

    @staticmethod
    def list_solver():
        return [{'mechanical': 'spectral_basic'},
                {'mechanical': 'spectral_polarization'},
                {'mechanical': 'FEM'}]


class Create:
    def __init__(self):
        self._grid = GridRefactory()
        # self._material = MaterialRefactory()
#        self._loading = LoadingRefactory()
        # self._homogenization = {}

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value
    #
    # @property
    # def material(self):
    #     return self._material
    #
    # @material.setter
    # def material(self, value):
    #     self._material = value
    #
    # @property
    # def loading(self):
    #     return self._loading
    #
    # @loading.setter
    # def loading(self, value):
    #     self._loading = value

    @staticmethod
    def loading(solver, load_steps):
        return DamaskLoading(solver=solver, load_steps=load_steps)

    @staticmethod
    def material(rotation, elements, phase, homogenization):
        return MaterialRefactory.config(rotation, elements, phase, homogenization)

    @staticmethod
    def homogenization(method, parameters):
        return {method: parameters}

    @staticmethod
    def phase(composition, lattice, output_list, elasticity, plasticity):
        return {composition: {'lattice': lattice, 'mechanics': {'output': output_list, 'elasticity': elasticity, 'plasticity': plasticity}}}

    @staticmethod
    def elasticity(**kwargs):
        _elast = {}
        for key, val in kwargs.items():
            _elast[key] = val
        return _elast

    @staticmethod
    def plasticity(**kwargs):
        _plast = {}
        for key, val in kwargs.items():
            _plast[key] = val
        return _plast

    @staticmethod
    def rotation(method, *args):
        return method(*args)

class MaterialRefactory:
    def __init__(self):
        pass

    @staticmethod
    def config(rotation, elements, phase, homogenization):
        _config = ConfigMaterial({'material': [], 'phase': phase, 'homogenization': homogenization})
        for r, e in zip(rotation, elements):
            _config = _config.material_add(O=r, phase=e, homogenization=list(homogenization.keys())[0])
        return _config

    @staticmethod
    def read(self, file_path):
        return ConfigMaterial.load(fname=file_path)

    @staticmethod
    def write(self, file_path):
        ConfigMaterial.save(fname=file_path)


class GridRefactory:
    def __init__(self):
        self.origin = Grid(material=np.ones((1, 1, 1)), size=[1., 1., 1.])

    @staticmethod
    def read(file_path):
        return Grid.load(fname=file_path)

    @staticmethod
    def via_voronoi_tessellation(grid_dim, num_grains, box_size):
        if type(grid_dim) == int or type(grid_dim) == float:
            grid_dim = np.array([grid_dim, grid_dim, grid_dim])
        if type(box_size) == int or type(box_size) == float:
            box_size = np.array([box_size, box_size, box_size])
        seed = seeds.from_random(box_size, num_grains)
        return Grid.from_Voronoi_tessellation(grid_dim, box_size, seed)


class DamaskLoading(Config):
    def __init__(self, load_steps, solver):
        super(DamaskLoading, self).__init__(self)
        self["solver"] = solver
        if isinstance(load_steps, list):
            self["loadstep"] = [
                LoadStep(mech_bc_dict=load_step['mech_bc_dict'],
                         discretization=load_step['discretization'],
                         additional_parameters_dict=load_step["additional"])
                        for load_step in load_steps ]
        else:
            self["loadstep"] = [
                LoadStep(mech_bc_dict=load_steps['mech_bc_dict'],
                         discretization=load_steps['discretization'],
                         additional_parameters_dict=load_steps["additional"])
            ]


class LoadStep(dict):
    def __init__(self, mech_bc_dict, discretization, additional_parameters_dict=None):
        super(LoadStep, self).__init__(self)
        self.update({'boundary_conditions': {'mechanical': {}},
                     'discretization': discretization})

        if additional_parameters_dict is not None and isinstance(additional_parameters_dict, dict):
            self.update(additional_parameters_dict)

        for key, val in mech_bc_dict.items():
            self['boundary_conditions']['mechanical'].update({key: LoadStep.load_tensorial(val)})

    @staticmethod
    def load_tensorial(arr):
        return [arr[0:3], arr[3:6], arr[6:9]]


