"""mirgecom driver for the Y2 prediction."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import sys
import yaml
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math
import grudge.op as op
from dataclasses import dataclass, fields
from pytools.obj_array import make_obj_array
from functools import partial
from mirgecom.discretization import create_discretization_collection

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import VolumeDomainTag, DOFDesc
from grudge.op import nodal_max, nodal_min
from grudge.dof_desc import DD_VOLUME_ALL
from grudge.trace_pair import inter_volume_trace_pairs
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
    logmgr_add_mempool_usage,
)

from mirgecom.simutil import (
    check_step,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    force_evaluation
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)
from mirgecom.inviscid import (inviscid_facial_flux_rusanov,
                               inviscid_facial_flux_hll)
from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.fluid import make_conserved
from mirgecom.limiter import bound_preserving_limiter
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalWallBoundary,
    AdiabaticSlipBoundary,
    AdiabaticNoslipWallBoundary,
    DummyBoundary
)
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary
)
#from mirgecom.initializers import (Uniform, PlanarDiscontinuity)
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import (SimpleTransport,
                                PowerLawTransport,
                                ArtificialViscosityTransportDiv)
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_grad_t_operator,
    coupled_ns_heat_operator
)
from mirgecom.navierstokes import grad_cv_operator
# driver specific utilties
from utils import (
    getIsentropicPressure,
    getIsentropicTemperature,
    getMachFromAreaRatio
)

from grudge.trace_pair import TracePair
from mirgecom.viscous import viscous_facial_flux_central


class PressureOutflowBoundary(PrescribedFluidBoundary):
    r"""Outflow boundary treatment with prescribed pressure.

    This class implements an outflow boundary as described by
    [Mengaldo_2014]_.  The boundary condition is implemented as:

    .. math::

        \rho^+ &= \rho^-

        \rho\mathbf{Y}^+ &= \rho\mathbf{Y}^-

        \rho\mathbf{V}^+ &= \rho\mathbf{V}^-

    For an ideal gas at super-sonic flow conditions, i.e. when:

    .. math::

       \rho\mathbf{V} \cdot \hat{\mathbf{n}} \ge c,

    then the pressure is extrapolated from interior points:

    .. math::

        P^+ = P^-

    Otherwise, if the flow is sub-sonic, then the prescribed boundary pressure,
    $P^+$, is used. In both cases, the energy is computed as:

    .. math::

        \rho{E}^+ = \frac{\left(2~P^+ - P^-\right)}{\left(\gamma-1\right)}
        + \frac{1}{2}\rho^+\left(\mathbf{V}^+\cdot\mathbf{V}^+\right).

    For mixtures, the pressure is imposed or extrapolated in a similar fashion
    to the ideal gas case.
    However, the total energy depends on the temperature to account for the
    species enthalpy and variable specific heat at constant volume. For super-sonic
    flows, it is extrapolated from interior points:

    .. math::

       T^+ = T^-

    while for sub-sonic flows, it is evaluated using ideal gas law

    .. math::

        T^+ = \frac{P^+}{R_{mix} \rho^+}

    .. automethod:: __init__
    .. automethod:: outflow_state
    """

    def __init__(self, boundary_pressure=101325):
        """Initialize the boundary condition object."""
        self._pressure = boundary_pressure
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.outflow_state,
            inviscid_flux_func=self.inviscid_boundary_flux,
            viscous_flux_func=self.viscous_boundary_flux,
            boundary_temperature_func=self.temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

    def outflow_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.

        For super-sonic outflow, the interior flow properties (minus) are
        extrapolated to the exterior point (plus).
        For sub-sonic outflow, the pressure is imposed on the external point.

        For mixtures, the internal energy is obtained via temperature, which comes
        from ideal gas law with the mixture-weighted gas constant.
        For ideal gas, the internal energy is obtained directly from pressure.
        """
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        # boundary-normal velocity
        boundary_vel = np.dot(state_minus.velocity, nhat)*nhat
        boundary_speed = actx.np.sqrt(np.dot(boundary_vel, boundary_vel))
        speed_of_sound = state_minus.speed_of_sound
        kinetic_energy = gas_model.eos.kinetic_energy(state_minus.cv)
        gamma = gas_model.eos.gamma(state_minus.cv, state_minus.temperature)

        # evaluate internal energy based on prescribed pressure
        pressure_plus = 2.0*self._pressure - state_minus.pressure
        #pressure_plus = state_minus.pressure
        pressure_plus = 0.95*state_minus.pressure
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                state_minus.temperature,
                pressure_plus/(state_minus.cv.mass*gas_const))
            )

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(temp_plus,
                                            state_minus.species_mass_fractions))
        else:
            boundary_pressure = actx.np.where(actx.np.greater(boundary_speed,
                                                              speed_of_sound),
                                              state_minus.pressure, pressure_plus)
            internal_energy = boundary_pressure/(gamma - 1.0)

        total_energy = internal_energy + kinetic_energy
        cv_outflow = make_conserved(dim=state_minus.dim, mass=state_minus.cv.mass,
                                    momentum=state_minus.cv.momentum,
                                    energy=total_energy,
                                    species_mass=state_minus.cv.species_mass)

        return make_fluid_state(cv=cv_outflow, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def outflow_state_for_diffusion(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
        """Return state."""
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # boundary-normal velocity
        boundary_vel = np.dot(state_minus.velocity, nhat)*nhat
        boundary_speed = actx.np.sqrt(np.dot(boundary_vel, boundary_vel))
        speed_of_sound = state_minus.speed_of_sound
        kinetic_energy = gas_model.eos.kinetic_energy(state_minus.cv)
        gamma = gas_model.eos.gamma(state_minus.cv, state_minus.temperature)

        # evaluate internal energy based on prescribed pressure
        #pressure_plus = self._pressure + 0.0*state_minus.pressure
        pressure_plus = state_minus.pressure
        pressure_plus = 0.95*state_minus.pressure
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                state_minus.temperature,
                pressure_plus/(state_minus.cv.mass*gas_const))
            )

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(
                    temp_plus, state_minus.species_mass_fractions)
            )
        else:
            boundary_pressure = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                              state_minus.pressure, pressure_plus)
            )
            internal_energy = (boundary_pressure / (gamma - 1.0))

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=kinetic_energy + internal_energy,
            momentum=state_minus.momentum_density,
            species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def inviscid_boundary_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """."""
        outflow_state = self.outflow_state(
            dcoll, dd_bdry, gas_model, state_minus)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=outflow_state)

        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(state_pair, gas_model, normal)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus

    def grad_temperature_bc(self, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus

    def viscous_boundary_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                                           **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        state_plus = self.outflow_state_for_diffusion(dcoll=dcoll,
            dd_bdry=dd_bdry, gas_model=gas_model, state_minus=state_minus)

        grad_cv_plus = self.grad_cv_bc(state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)
        grad_t_plus = self.grad_temperature_bc(grad_t_minus, normal)

        # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
        # *not* the numerical viscous flux as advised by [Bassi_1997]_.
        f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                             grad_t=grad_t_plus)

        return f_ext@normal


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _SmoothnessCVGradCommTag:
    pass


class _OxCommTag:
    pass


class _FluidOxDiffCommTag:
    pass


class _WallOxDiffCommTag:
    pass


class HeatSource:
    r"""Deposit energy from an ignition source."

    Internal energy is deposited as a gaussian of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    Density if modified to keep pressure constant, according to the eos

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the spark parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._amplitude = amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, state, eos, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        If modify density is true, only adjust the temperature. Pressure
        is maintained by adjusting the density.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            Fluid conserved quantities
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """
        t = time
        if self._amplitude_func is not None:
            amplitude = self._amplitude*self._amplitude_func(t)
        else:
            amplitude = self._amplitude

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(-(r**2)/(2*self._width*self._width))

        # elevate the local temperature
        # if it's below some threshold
        #temperature = state.temperature + expterm
        temp_max = 10000.0
        temperature = actx.np.where(
            actx.np.greater(state.temperature, temp_max),
            state.temperature,
            state.temperature + expterm)
        pressure = state.pressure
        y = state.species_mass_fractions

        # density of this new state
        new_mass = eos.get_density(pressure=pressure, temperature=temperature,
                                   species_mass_fractions=y)

        # change the density so the pressure stays constant
        mass_source = new_mass - state.mass_density

        # keep the velocity constant
        momentum_source = state.velocity*mass_source

        # keep the mass fractions constant
        species_mass_source = state.species_mass_fractions*mass_source

        # the source term that keeps the energy constant having changed the mass
        energy_source = 0.5*np.dot(state.velocity, state.velocity)*mass_source

        return make_conserved(dim=self._dim, mass=mass_source,
                              energy=energy_source,
                              momentum=momentum_source,
                              species_mass=species_mass_source)


class SparkSource:
    r"""Energy deposition from a ignition source"

    Internal energy is deposited as a gaussian  of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the spark parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """

        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._amplitude = amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, cv, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.gas_model.FluidState`
            Fluid state object with the conserved and thermal state.
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """

        t = time
        if self._amplitude_func is not None:
            amplitude = self._amplitude*self._amplitude_func(t)
        else:
            amplitude = self._amplitude

        #print(f"{time=} {amplitude=}")

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(-(r**2)/(2*self._width*self._width))

        mass = 0*cv.mass
        momentum = 0*cv.momentum
        species_mass = 0*cv.species_mass

        energy = cv.energy + cv.mass*expterm

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=species_mass)


class InitSponge:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness) *
            ((xpos - self._x0)/self._thickness)),
            zeros + 0.0
        )


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(
            self,
            heat_capacity,
            thermal_conductivity_func,
            *,
            effective_surface_area_func=None,
            mass_loss_func=None,
            oxygen_diffusivity=0.):
        self._heat_capacity = heat_capacity
        self._thermal_conductivity_func = thermal_conductivity_func
        self._effective_surface_area_func = effective_surface_area_func
        self._mass_loss_func = mass_loss_func
        self._oxygen_diffusivity = oxygen_diffusivity

    @property
    def heat_capacity(self):
        return self._heat_capacity

    def thermal_conductivity(self, mass, temperature):
        return self._thermal_conductivity_func(mass, temperature)

    def thermal_diffusivity(self, mass, temperature, thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(mass, temperature)
        return thermal_conductivity/(mass * self.heat_capacity)

    def mass_loss_rate(self, mass, ox_mass, temperature):
        dm = mass*0.
        if self._effective_surface_area_func is not None:
            eff_surf_area = self._effective_surface_area_func(mass)
            if self._mass_loss_func is not None:
                dm = self._mass_loss_func(mass, ox_mass, temperature, eff_surf_area)
        return dm

    @property
    def oxygen_diffusivity(self):
        return self._oxygen_diffusivity

    def temperature(self, wv):
        return wv.energy/(wv.mass * self.heat_capacity)

    def dependent_vars(self, wv):
        temperature = self.temperature(wv)
        kappa = self.thermal_conductivity(wv.mass, temperature)
        return WallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         restart_filename=None, target_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False, log_path="log_data"):

    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    logname = log_path + "/" + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    comm.Barrier()

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1

    # verbosity for what gets written to viz dumps, increase for more stuff
    viz_level = 1
    # control the time interval for writing viz dumps
    viz_interval_type = 0

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    t_viz_interval = 1.e-8
    current_t = 0
    t_start = 0.
    t_wall_start = 0.
    current_step = 0
    first_step = 0
    current_cfl = 1.0
    constant_cfl = False
    last_viz_interval = 0
    force_eval = True

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6
    health_temp_min = 1.0
    health_temp_max = 5000.
    health_mass_frac_min = -10
    health_mass_frac_max = 10

    # discretization and model control
    order = 1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2
    inv_num_flux = "rusanov"
    mesh_filename = "data/actii_2d.msh"
    noslip = True
    adiabatic = False
    use_1d_part = True

    # material properties
    gas_mat_prop = 0
    mu = 1.0e-5
    spec_diff = 1.e-4
    mu_override = False  # optionally read in from input
    nspecies = 0
    pyro_temp_iter = 3  # for pyrometheus, number of newton iterations
    pyro_temp_tol = 1.e-4  # for pyrometheus, toleranace for temperature residual
    transport_type = 0
    eos_type = 0
    # for overwriting the defaults
    fluid_mw = -1.
    fluid_gamma = -1.
    fluid_kappa = -1.

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # for graphite
    wall_insert_rho = 1625
    wall_insert_cp = 770
    wall_insert_kappa = 247.5  # This seems high

    # Fiberform
    # wall_insert_rho = 183.6
    # wall_insert_cp = 710
    wall_insert_ox_diff = spec_diff

    # Averaging from http://www.matweb.com/search/datasheet.aspx?bassnum=MS0001
    # for steel
    wall_surround_rho = 7.9e3
    wall_surround_cp = 470
    wall_surround_kappa = 48

    # rhs control
    use_ignition = 0
    use_sponge = True
    use_combustion = True
    use_injection = True
    use_wall_ox = True
    use_wall_mass = True

    # artificial viscosity control
    #    0 - none
    #    1 - physical viscosity based, div(velocity) indicator
    use_av = 0

    # species limiter
    #    0 - none
    #    1 - limit in on call to make_fluid_state
    use_species_limiter = 0

    sponge_sigma = 1.0

    # Filtering is implemented according to HW Sec. 5.3
    # The modal response function is e^-(alpha * eta ^ 2s), where
    # - alpha is a user parameter (defaulted like HW's)
    # - eta := (mode - N_c)/(N - N_c)
    # - N_c := cutoff mode ( = *filter_frac* x order)
    # - s := order of the filter (divided by 2)
    # Modes below N_c are unfiltered. Modes above Nc are weighted
    # by the modal response function described above.
    #
    # Two different filters can be used with the prediction driver.
    # 1) Solution filtering: filters the solution every *soln_nfilter* steps
    # 2) RHS filtering: filters the RHS every substep
    #
    # Turn on SOLUTION filtering by setting soln_nfilter > 0
    # Turn on RHS filtering by setting use_rhs_filter = 1.
    #
    # --- Filtering settings ---
    # ------ Solution filtering
    soln_nfilter = -1  # filter every *nfilter* steps (-1 = no filtering)
    soln_filter_cutoff = -1  # (-1 = filter_frac*order)
    soln_filter_frac = .5
    soln_filter_order = 8
    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Section 5.3
    # DOI: 10.1007/978-0-387-72067-8
    soln_filter_alpha = -1.0*np.log(np.finfo(float).eps)
    # ------ RHS filtering
    use_rhs_filter = False
    rhs_filter_cutoff = -1
    rhs_filter_frac = .5
    rhs_filter_order = 8
    rhs_filter_alpha = soln_filter_alpha

    # ACTII flow properties
    total_pres_inflow = 2.745e5
    total_temp_inflow = 2076.43

    # injection flow properties
    total_pres_inj = 50400
    total_temp_inj = 300.0
    mach_inj = 1.0

    # parameters to adjust the shape of the initialization
    vel_sigma = 1000
    temp_sigma = 1250
    # adjusted to match the mass flow rate
    vel_sigma_inj = 5000
    temp_sigma_inj = 5000
    temp_wall = 300
    sponge_thickness = 0.09
    sponge_x0 = 0.9

    # wall stuff
    wall_penalty_amount = 25
    wall_time_scale = 50
    wall_material = 0

    # initialize the ignition spark
    spark_init_loc_x = 0.677
    spark_init_loc_y = -0.021
    spark_init_loc_z = 0.035/2.
    spark_diameter = 0.0025
    spark_strength = 20000000.
    spark_init_time = 999999999.
    spark_duration = 1.e-8

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            t_viz_interval = float(input_data["t_viz_interval"])
        except KeyError:
            pass
        try:
            viz_interval_type = int(input_data["viz_interval_type"])
        except KeyError:
            pass
        try:
            viz_level = int(input_data["viz_level"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            soln_nfilter = int(input_data["soln_nfilter"])
        except KeyError:
            pass
        try:
            soln_filter_frac = float(input_data["soln_filter_frac"])
        except KeyError:
            pass
        try:
            soln_filter_cutoff = int(input_data["soln_filter_cutoff"])
        except KeyError:
            pass
        try:
            soln_filter_alpha = float(input_data["soln_filter_alpha"])
        except KeyError:
            pass
        try:
            soln_filter_order = int(input_data["soln_filter_order"])
        except KeyError:
            pass
        try:
            use_rhs_filter = bool(input_data["use_rhs_filter"])
        except KeyError:
            pass
        try:
            rhs_filter_frac = float(input_data["rhs_filter_frac"])
        except KeyError:
            pass
        try:
            rhs_filter_cutoff = int(input_data["rhs_filter_cutoff"])
        except KeyError:
            pass
        try:
            rhs_filter_alpha = float(input_data["rhs_filter_alpha"])
        except KeyError:
            pass
        try:
            rhs_filter_order = int(input_data["rhs_filter_order"])
        except KeyError:
            pass
        try:
            use_species_limiter = int(input_data["use_species_limiter"])
        except KeyError:
            pass
        try:
            constant_cfl = int(input_data["constant_cfl"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            current_cfl = float(input_data["current_cfl"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            fluid_gamma = float(input_data["fluid_gamma"])
        except KeyError:
            pass
        try:
            fluid_mw = float(input_data["fluid_mw"])
        except KeyError:
            pass
        try:
            fluid_kappa = float(input_data["fluid_kappa"])
        except KeyError:
            pass
        try:
            mu_input = float(input_data["mu"])
            mu_override = True
        except KeyError:
            pass
        try:
            spec_diff = float(input_data["spec_diff"])
            wall_insert_ox_diff = spec_diff
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            noslip = bool(input_data["noslip"])
        except KeyError:
            pass
        try:
            adiabatic = bool(input_data["adiabatic"])
        except KeyError:
            pass
        try:
            use_1d_part = bool(input_data["use_1d_part"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            total_pres_inflow = float(input_data["total_pres_inflow"])
        except KeyError:
            pass
        try:
            total_temp_inflow = float(input_data["total_temp_inflow"])
        except KeyError:
            pass
        try:
            total_pres_inj = float(input_data["total_pres_inj"])
        except KeyError:
            pass
        try:
            total_temp_inj = float(input_data["total_temp_inj"])
        except KeyError:
            pass
        try:
            mach_inj = float(input_data["mach_inj"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass
        try:
            eos_type = int(input_data["eos"])
        except KeyError:
            pass
        try:
            transport_type = int(input_data["transport"])
        except KeyError:
            pass
        try:
            pyro_temp_iter = int(input_data["pyro_temp_iter"])
        except KeyError:
            pass
        try:
            pyro_temp_tol = float(input_data["pyro_temp_tol"])
        except KeyError:
            pass
        try:
            vel_sigma = float(input_data["vel_sigma"])
        except KeyError:
            pass
        try:
            temp_sigma = float(input_data["temp_sigma"])
        except KeyError:
            pass
        try:
            vel_sigma_inj = float(input_data["vel_sigma_inj"])
        except KeyError:
            pass
        try:
            temp_sigma_inj = float(input_data["temp_sigma_inj"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            inv_num_flux = input_data["inviscid_numerical_flux"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass
        try:
            health_temp_min = float(input_data["health_temp_min"])
        except KeyError:
            pass
        try:
            health_temp_max = float(input_data["health_temp_max"])
        except KeyError:
            pass
        try:
            health_mass_frac_min = float(input_data["health_mass_frac_min"])
        except KeyError:
            pass
        try:
            health_mass_frac_max = float(input_data["health_mass_frac_max"])
        except KeyError:
            pass
        try:
            use_ignition = int(input_data["use_ignition"])
        except KeyError:
            pass
        try:
            use_injection = int(input_data["use_injection"])
        except KeyError:
            pass
        try:
            spark_init_time = float(input_data["ignition_init_time"])
        except KeyError:
            pass
        try:
            spark_strength = float(input_data["ignition_strength"])
        except KeyError:
            pass
        try:
            spark_duration = float(input_data["ignition_duration"])
        except KeyError:
            pass
        try:
            spark_diameter = float(input_data["ignition_diameter"])
        except KeyError:
            pass
        try:
            spark_init_loc_x = float(input_data["ignition_loc_x"])
        except KeyError:
            pass
        try:
            spark_init_loc_y = float(input_data["ignition_loc_y"])
        except KeyError:
            pass
        try:
            use_sponge = bool(input_data["use_sponge"])
        except KeyError:
            pass
        try:
            sponge_sigma = float(input_data["sponge_sigma"])
        except KeyError:
            pass
        try:
            sponge_thickness = float(input_data["sponge_thickness"])
        except KeyError:
            pass
        try:
            sponge_x0 = float(input_data["sponge_x0"])
        except KeyError:
            pass
        try:
            use_av = int(input_data["use_av"])
        except KeyError:
            pass
        try:
            use_combustion = bool(input_data["use_combustion"])
        except KeyError:
            pass
        try:
            use_wall_ox = bool(input_data["use_wall_ox"])
        except KeyError:
            pass
        try:
            use_wall_mass = bool(input_data["use_wall_mass"])
        except KeyError:
            pass
        try:
            mesh_filename = input_data["mesh_filename"]
        except KeyError:
            pass
        try:
            wall_penalty_amount = float(input_data["wall_penalty_amount"])
        except KeyError:
            pass
        try:
            wall_time_scale = float(input_data["wall_time_scale"])
        except KeyError:
            pass
        try:
            wall_material = int(input_data["wall_material"])
        except KeyError:
            pass
        try:
            wall_insert_rho = float(input_data["wall_insert_rho"])
        except KeyError:
            pass
        try:
            wall_insert_cp = float(input_data["wall_insert_cp"])
        except KeyError:
            pass
        try:
            wall_insert_kappa = float(input_data["wall_insert_kappa"])
        except KeyError:
            pass
        try:
            wall_surround_rho = float(input_data["wall_surround_rho"])
        except KeyError:
            pass
        try:
            wall_surround_cp = float(input_data["wall_surround_cp"])
        except KeyError:
            pass
        try:
            wall_surround_kappa = float(input_data["wall_surround_kappa"])
        except KeyError:
            pass
        try:
            gas_mat_prop = int(input_data["gas_material_properties"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144", "compiled_lsrk54"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if integrator == "compiled_lsrk54":
        print("Setting force_eval = False for pre-compiled time integration")
        force_eval = False

    if viz_interval_type > 2:
        error_message = "Invalid value for viz_interval_type [0-2]"
        raise RuntimeError(error_message)

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    # use_av=1 specific parameters
    # flow stagnation temperature
    static_temp = 2076.43
    # steepness of the smoothed function
    theta_sc = 100
    # cutoff, smoothness below this value is ignored
    beta_sc = 0.01
    gamma_sc = 1.5

    if rank == 0:
        if use_av == 0:
            print("Artificial viscosity disabled")
        else:
            print("Artificial viscosity using modified physical viscosity")
            print("Using velocity divergence indicator")
            print(f"Shock capturing parameters: alpha {alpha_sc}, "
                  f"gamma_sc {gamma_sc}"
                  f"theta_sc {theta_sc}, beta_sc {beta_sc}, Pr 0.75, "
                  f"stagnation temperature {static_temp}")

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if constant_cfl == 1:
            print(f"\tConstant cfl mode, current_cfl = {current_cfl}")
        else:
            print(f"\tConstant dt mode, current_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        if noslip:
            print("Fluid wall boundary conditions are noslip for veloctiy")
        else:
            print("Fluid wall boundary conditions are slip for veloctiy")
        if adiabatic:
            print("Fluid wall boundary conditions are adiabatic for temperature")
        else:
            print("Fluid wall boundary conditions are isothermal for temperature")
        print("#### Simluation control data: ####\n")

    if rank == 0:
        print("\n#### Visualization setup: ####")
        if viz_level >= 0:
            print("\tBasic visualization output enabled.")
            print("\t(cv, dv, cfl)")
        if viz_level >= 1:
            print("\tExtra visualization output enabled for derived quantities.")
            print("\t(velocity, mass_fractions, etc.)")
        if viz_level >= 2:
            print("\tNon-dimensional parameter visualization output enabled.")
            print("\t(Re, Pr, etc.)")
        if viz_level >= 3:
            print("\tDebug visualization output enabled.")
            print("\t(rhs, grad_cv, etc.)")
        if viz_interval_type == 0:
            print(f"\tWriting viz data every {nviz} steps.")
        if viz_interval_type == 1:
            print(f"\tWriting viz data roughly every {t_viz_interval} seconds.")
        if viz_interval_type == 2:
            print(f"\tWriting viz data exactly every {t_viz_interval} seconds.")
        print("#### Visualization setup: ####")

    """
    if not noslip:
        vel_sigma = 0.
    if adiabatic:
        temp_sigma = 0.
    """

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\ttotal_pres_injection = {total_pres_inj}")
        print(f"\ttotal_temp_injection = {total_temp_inj}")
        print(f"\tvel_sigma = {vel_sigma}")
        print(f"\ttemp_sigma = {temp_sigma}")
        print(f"\tvel_sigma_injection = {vel_sigma_inj}")
        print(f"\ttemp_sigma_injection = {temp_sigma_inj}")
        print("#### Simluation setup data: ####")

    spark_center = np.zeros(shape=(dim,))
    spark_center[0] = spark_init_loc_x
    spark_center[1] = spark_init_loc_y
    if dim == 3:
        spark_center[2] = spark_init_loc_z
    if rank == 0 and use_ignition > 0:
        print("\n#### Ignition control parameters ####")
        print(f"spark center ({spark_center[0]},{spark_center[1]})")
        print(f"spark FWHM {spark_diameter}")
        print(f"spark strength {spark_strength}")
        print(f"ignition time {spark_init_time}")
        print(f"ignition duration {spark_duration}")
        if use_ignition == 1:
            print("spark ignition")
        elif use_ignition == 2:
            print("heat source ignition")
        print("#### Ignition control parameters ####\n")

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step
    if integrator == "compiled_lsrk54":
        timestepper = _compiled_stepper_wrapper

    if inv_num_flux == "rusanov":
        inviscid_numerical_flux_func = inviscid_facial_flux_rusanov
        if rank == 0:
            print("\nRusanov inviscid flux")
    if inv_num_flux == "hll":
        inviscid_numerical_flux_func = inviscid_facial_flux_hll
        if rank == 0:
            print("\nHLL inviscid flux")

    # }}}

    # constants
    mw_o = 15.999
    mw_o2 = mw_o*2
    mw_co = 28.010
    mw_n2 = 14.0067*2
    mw_c2h4 = 28.05
    mw_h2 = 1.00784*2
    mw_ar = 39.948
    univ_gas_const = 8314.59

    mf_o2 = 0.273

    if gas_mat_prop == 0:
        # working gas: O2/N2 #
        #   O2 mass fraction 0.273
        #   gamma = 1.4
        #   cp = 37.135 J/mol-K,
        #   rho= 1.977 kg/m^3 @298K
        gamma = 1.4
        mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    if gas_mat_prop == 1:
        # working gas: Ar #
        #   O2 mass fraction 0.273
        #   gamma = 1.4
        #   cp = 37.135 J/mol-K,
        #   rho= 1.977 kg/m^3 @298K
        gamma = 5/3
        mw = mw_ar

    if fluid_gamma > 0:
        gamma = fluid_gamma

    mf_c2h4 = mw_c2h4/(mw_c2h4 + mw_h2)
    mf_h2 = 1 - mf_c2h4

    # user can reset the mw to whatever they want
    if fluid_mw > 0:
        mw = fluid_mw

    r = univ_gas_const/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75

    # viscosity @ 400C, Pa-s
    if gas_mat_prop == 0:
        # working gas: O2/N2 #
        mu_o2 = 3.76e-5
        mu_n2 = 3.19e-5
        mu = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    if gas_mat_prop == 1:
        # working gas: Ar #
        mu_ar = 4.22e-5
        mu = mu_ar
    if mu_override:
        mu = mu_input

    kappa = cp*mu/Pr
    if fluid_kappa > 0:
        kappa = fluid_kappa
    init_temperature = 300.0

    # don't allow limiting on flows without species
    if nspecies == 0:
        use_species_limiter = 0
        use_injection = False

    # Turn off combustion unless EOS supports it
    if nspecies < 3:
        use_combustion = False

    if nspecies > 3:
        eos_type = 1

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print("#### Fluid domain: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        if nspecies == 0:
            print("\tno passive scalars, uniform species mixture")
            if gas_mat_prop == 0:
                print("\tO2/N2 mix material properties.")
            else:
                print("\tAr material properties.")
        elif nspecies == 3:
            print("\tpassive scalars to track air/fuel/inert mixture, ideal gas eos")
        elif nspecies == 5:
            print("\tfull multi-species initialization with pyrometheus eos")
            print("\tno combustion source terms")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")
            print("\tcombustion source terms enabled")

        if eos_type == 0:
            print("\tIdeal Gas EOS")
        elif eos_type == 1:
            print("\tPyrometheus EOS")

        if use_species_limiter == 1:
            print("\nSpecies mass fractions limited to [0:1]")

    transport_alpha = 0.6
    transport_beta = 4.093e-7
    transport_sigma = 2.0
    transport_n = 0.666

    if rank == 0:
        if transport_type == 0:
            print("\t Simple transport model:")
            print("\t\t constant viscosity, species diffusivity")
            print(f"\tmu = {mu}")
            print(f"\tkappa = {kappa}")
            print(f"\tspecies diffusivity = {spec_diff}")
        elif transport_type == 1:
            print("\t Power law transport model:")
            print("\t\t temperature dependent viscosity, species diffusivity")
            print(f"\ttransport_alpha = {transport_alpha}")
            print(f"\ttransport_beta = {transport_beta}")
            print(f"\ttransport_sigma = {transport_sigma}")
            print(f"\ttransport_n = {transport_n}")
            print(f"\tspecies diffusivity = {spec_diff}")
        elif transport_type == 2:
            print("\t Pyrometheus transport model:")
            print("\t\t temperature/mass fraction dependence")
        else:
            error_message = "Unknown transport_type {}".format(transport_type)
            raise RuntimeError(error_message)

        print("#### Wall domain: ####")

        if wall_material == 0:
            print("\tNon-reactive wall model")
        elif wall_material == 1:
            print("\tReactive wall model for non-porous media")
        elif wall_material == 2:
            print("\tReactive wall model for porous media")
        else:
            error_message = "Unknown wall_material {}".format(wall_material)
            raise RuntimeError(error_message)

        if use_wall_ox:
            print("\tWall oxidizer transport enabled")
        else:
            print("\tWall oxidizer transport disabled")

        if use_wall_mass:
            print("\t Wall mass loss enabled")
        else:
            print("\t Wall mass loss disabled")

        print(f"\tWall density = {wall_insert_rho}")
        print(f"\tWall cp = {wall_insert_cp}")
        print(f"\tWall O2 diff = {wall_insert_ox_diff}")
        print(f"\tWall surround density = {wall_surround_rho}")
        print(f"\tWall surround cp = {wall_surround_cp}")
        print(f"\tWall surround kappa = {wall_surround_kappa}")
        print(f"\tWall time scale = {wall_time_scale}")
        print(f"\tWall penalty = {wall_penalty_amount}")
        print("#### Simluation material properties: ####")

    spec_diffusivity = spec_diff * np.ones(nspecies)
    if transport_type == 0:
        physical_transport_model = SimpleTransport(
            viscosity=mu, thermal_conductivity=kappa,
            species_diffusivity=spec_diffusivity)
    if transport_type == 1:
        physical_transport_model = PowerLawTransport(
            alpha=transport_alpha, beta=transport_beta,
            sigma=transport_sigma, n=transport_n,
            species_diffusivity=spec_diffusivity)

    transport_model = physical_transport_model
    if use_av:
        transport_model = ArtificialViscosityTransportDiv(
            physical_transport=physical_transport_model,
            av_mu=alpha_sc, av_prandtl=0.75)

    #
    # stagnation tempertuare 2076.43 K
    # stagnation pressure 2.745e5 Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=54e-3m) and
    # the throat (r=3.167e-3)
    #
    vel_inflow = np.zeros(shape=(dim,))
    vel_outflow = np.zeros(shape=(dim,))
    vel_injection = np.zeros(shape=(dim,))

    throat_height = 3.61909e-3
    inlet_height = 54.129e-3
    outlet_height = 28.54986e-3
    inlet_area_ratio = inlet_height/throat_height
    outlet_area_ratio = outlet_height/throat_height

    chem_source_tol = 1.e-10
    # make the eos
    if eos_type == 0:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
        species_names = ["air", "fuel", "inert"]
    else:
        from mirgecom.thermochemistry import get_pyrometheus_wrapper_class
        from uiuc import Thermochemistry
        pyro_mech = get_pyrometheus_wrapper_class(
            pyro_class=Thermochemistry, temperature_niter=pyro_temp_iter,
            zero_level=chem_source_tol)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        species_names = pyro_mech.species_names

    gas_model = GasModel(eos=eos, transport=transport_model)

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    if nspecies == 3:
        y[0] = 1
        y_fuel[1] = 1
    elif nspecies > 4:
        # find name species indicies
        for i in range(nspecies):
            if species_names[i] == "C2H4":
                i_c2h4 = i
            if species_names[i] == "H2":
                i_h2 = i
            if species_names[i] == "O2":
                i_ox = i
            if species_names[i] == "N2":
                i_di = i

        # Set the species mass fractions to the free-stream flow
        y[i_ox] = mf_o2
        y[i_di] = 1. - mf_o2
        # Set the species mass fractions to the free-stream flow
        y_fuel[i_c2h4] = mf_c2h4
        y_fuel[i_h2] = mf_h2

    inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                      gamma=gamma,
                                      mach_guess=0.01)
    pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                        P0=total_pres_inflow,
                                        gamma=gamma)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                           T0=total_temp_inflow,
                                           gamma=gamma)

    if eos_type == 0:
        rho_inflow = pres_inflow/temp_inflow/r
        sos = math.sqrt(gamma*pres_inflow/rho_inflow)
        inlet_gamma = gamma
    else:
        rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                          temperature=temp_inflow,
                                          mass_fractions=y)
        inlet_gamma = (pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                       pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))

        gamma_error = (gamma - inlet_gamma)
        gamma_guess = inlet_gamma
        toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
        while gamma_error > toler:

            inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                              gamma=gamma_guess,
                                              mach_guess=0.01)
            pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                                P0=total_pres_inflow,
                                                gamma=gamma_guess)
            temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                                   T0=total_temp_inflow,
                                                   gamma=gamma_guess)

            rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                              temperature=temp_inflow,
                                              mass_fractions=y)
            inlet_gamma = \
                (pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                 pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))
            gamma_error = (gamma_guess - inlet_gamma)
            gamma_guess = inlet_gamma

        sos = math.sqrt(inlet_gamma*pres_inflow/rho_inflow)

    vel_inflow[0] = inlet_mach*sos

    if rank == 0:
        print("#### Simluation initialization data: ####")
        print(f"\tinlet Mach number {inlet_mach}")
        print(f"\tinlet gamma {inlet_gamma}")
        print(f"\tinlet temperature {temp_inflow}")
        print(f"\tinlet pressure {pres_inflow}")
        print(f"\tinlet rho {rho_inflow}")
        print(f"\tinlet velocity {vel_inflow[0]}")
        #print(f"final inlet pressure {pres_inflow_final}")

    outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                       gamma=gamma,
                                       mach_guess=1.1)
    pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                         P0=total_pres_inflow,
                                         gamma=gamma)
    temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                            T0=total_temp_inflow,
                                            gamma=gamma)

    if eos_type == 0:
        rho_outflow = pres_outflow/temp_outflow/r
        sos = math.sqrt(gamma*pres_outflow/rho_outflow)
        outlet_gamma = gamma
    else:
        rho_outflow = pyro_mech.get_density(p=pres_outflow,
                                            temperature=temp_outflow,
                                            mass_fractions=y)
        outlet_gamma = \
            (pyro_mech.get_mixture_specific_heat_cp_mass(temp_outflow, y) /
             pyro_mech.get_mixture_specific_heat_cv_mass(temp_outflow, y))

        gamma_error = (gamma - outlet_gamma)
        gamma_guess = outlet_gamma
        toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
        while gamma_error > toler:

            outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                              gamma=gamma_guess,
                                              mach_guess=0.01)
            pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                                P0=total_pres_inflow,
                                                gamma=gamma_guess)
            temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                                   T0=total_temp_inflow,
                                                   gamma=gamma_guess)
            rho_outflow = pyro_mech.get_density(p=pres_outflow,
                                                temperature=temp_outflow,
                                                mass_fractions=y)
            outlet_gamma = \
                (pyro_mech.get_mixture_specific_heat_cp_mass(temp_outflow, y) /
                 pyro_mech.get_mixture_specific_heat_cv_mass(temp_outflow, y))
            gamma_error = (gamma_guess - outlet_gamma)
            gamma_guess = outlet_gamma

    vel_outflow[0] = outlet_mach*math.sqrt(gamma*pres_outflow/rho_outflow)

    if rank == 0:
        print("\t********")
        print(f"\toutlet Mach number {outlet_mach}")
        print(f"\toutlet gamma {outlet_gamma}")
        print(f"\toutlet temperature {temp_outflow}")
        print(f"\toutlet pressure {pres_outflow}")
        print(f"\toutlet rho {rho_outflow}")
        print(f"\toutlet velocity {vel_outflow[0]}")

    gamma_injection = gamma
    if nspecies > 0:
        # injection mach number
        if eos_type == 0:
            gamma_injection = gamma
        else:
            #MJA: Todo, get the gamma from cantera to get the correct
            # inflow properties
            # needs to be iterative with the call below
            gamma_injection = 0.5*(1.24 + 1.4)

        pres_injection = getIsentropicPressure(mach=mach_inj,
                                               P0=total_pres_inj,
                                               gamma=gamma_injection)
        temp_injection = getIsentropicTemperature(mach=mach_inj,
                                                  T0=total_temp_inj,
                                                  gamma=gamma_injection)

        if eos_type == 0:
            rho_injection = pres_injection/temp_injection/r
            sos = math.sqrt(gamma*pres_injection/rho_injection)
        else:
            rho_injection = pyro_mech.get_density(p=pres_injection,
                                                  temperature=temp_injection,
                                                  mass_fractions=y)
            gamma_injection = \
                (pyro_mech.get_mixture_specific_heat_cp_mass(temp_injection, y) /
                 pyro_mech.get_mixture_specific_heat_cv_mass(temp_injection, y))

            gamma_error = (gamma - gamma_injection)
            gamma_guess = gamma_injection
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                                  gamma=gamma_guess,
                                                  mach_guess=0.01)
                pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                                    P0=total_pres_inj,
                                                    gamma=gamma_guess)
                temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                                       T0=total_temp_inj,
                                                       gamma=gamma_guess)
                rho_injection = pyro_mech.get_density(p=pres_injection,
                                                      temperature=temp_injection,
                                                      mass_fractions=y)
                gamma_injection = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(temp_injection, y) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(temp_injection, y))
                gamma_error = (gamma_guess - gamma_injection)
                gamma_guess = gamma_injection

            sos = math.sqrt(gamma_injection*pres_injection/rho_injection)

        vel_injection[0] = -mach_inj*sos

        if rank == 0:
            print("\t********")
            print(f"\tinjector Mach number {mach_inj}")
            print(f"\tinjector gamma {gamma_injection}")
            print(f"\tinjector temperature {temp_injection}")
            print(f"\tinjector pressure {pres_injection}")
            print(f"\tinjector rho {rho_injection}")
            print(f"\tinjector velocity {vel_injection[0]}")
            print("#### Simluation initialization data: ####\n")
    else:
        if rank == 0:
            print("\t********")
            print("\tnspecies=0, injection disabled")

    # read geometry files
    geometry_bottom = None
    geometry_top = None
    if rank == 0:
        from numpy import loadtxt
        geometry_bottom = loadtxt("nozzleBottom.dat", comments="#", unpack=False)
        geometry_top = loadtxt("nozzleTop.dat", comments="#", unpack=False)
    geometry_bottom = comm.bcast(geometry_bottom, root=0)
    geometry_top = comm.bcast(geometry_top, root=0)

    inj_ymin = -0.0243245
    inj_ymax = -0.0227345
    from actii import InitACTII
    bulk_init = InitACTII(dim=dim,
                          geom_top=geometry_top, geom_bottom=geometry_bottom,
                          P0=total_pres_inflow, T0=total_temp_inflow,
                          temp_wall=temp_wall, temp_sigma=temp_sigma,
                          vel_sigma=vel_sigma, nspecies=nspecies,
                          mass_frac=y, gamma_guess=inlet_gamma,
                          inj_gamma_guess=gamma_injection,
                          inj_pres=total_pres_inj,
                          inj_temp=total_temp_inj,
                          inj_vel=vel_injection, inj_mass_frac=y_fuel,
                          inj_temp_sigma=temp_sigma_inj,
                          inj_vel_sigma=vel_sigma_inj,
                          inj_ytop=inj_ymax, inj_ybottom=inj_ymin,
                          inj_mach=mach_inj, injection=use_injection)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )

    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        first_step = current_step
        current_t = restart_data["t"]
        last_viz_interval = restart_data["last_viz_interval"]
        t_start = current_t
        t_wall_start = restart_data["t_wall"]
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert restart_data["nparts"] == nparts
        assert restart_data["nspecies"] == nspecies
    else:  # generate the grid from scratch
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "fluid": ["fluid"],
                "wall": ["wall_insert", "wall_surround"]}
            return mesh, tag_to_elements, volume_to_tags

        def my_partitioner(mesh, tag_to_elements, num_ranks):
            from mirgecom.simutil import geometric_mesh_partitioner
            return geometric_mesh_partitioner(
                mesh, num_ranks, auto_balance=True, debug=True)

        part_func = my_partitioner if use_1d_part else None

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data, partition_generator_func=part_func)

    local_nelements = (
        volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["wall"][0].nelements)

    # target data, used for sponge and prescribed boundary condtitions
    if target_filename:  # read the grid from restart data
        target_filename = f"{target_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        target_data = read_restart_data(actx, target_filename)
        #volume_to_local_mesh_data = target_data["volume_to_local_mesh_data"]
        global_nelements = target_data["global_nelements"]
        target_order = int(target_data["order"])

        assert target_data["nparts"] == nparts
        assert target_data["nspecies"] == nspecies
        assert target_data["global_nelements"] == global_nelements
    else:
        logger.warning("No target file specied, using restart as target")

    if rank == 0:
        logger.info("Making discretization")

    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_wall = DOFDesc(VolumeDomainTag("wall"), DISCR_TAG_BASE)

    wall_vol_discr = dcoll.discr_from_dd(dd_vol_wall)
    wall_tag_to_elements = volume_to_local_mesh_data["wall"][1]
    wall_insert_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_insert"])
    wall_surround_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_surround"])

    from grudge.dt_utils import characteristic_lengthscales
    char_length = characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid)
    char_length_wall = characteristic_lengthscales(actx, dcoll, dd=dd_vol_wall)

    if rank == 0:
        logger.info("Before restart/init")

    #########################
    # Convenience Functions #
    #########################

    def limit_fluid_state(cv, pressure, temperature, dd=dd_vol_fluid):

        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll=dcoll, dd=dd,
                                     field=cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True)
            for i in range(nspecies)
        ])

        # limit the sum to 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        kin_energy = 0.5*np.dot(cv.velocity, cv.velocity)

        mass_lim = eos.get_density(pressure=pressure, temperature=temperature,
                                   species_mass_fractions=spec_lim)

        energy_lim = mass_lim*(
            gas_model.eos.get_internal_energy(temperature,
                                              species_mass_fractions=spec_lim)
            + kin_energy
        )

        mom_lim = mass_lim*cv.velocity

        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                              momentum=mom_lim,
                              species_mass=mass_lim*spec_lim)

    if soln_filter_cutoff < 0:
        soln_filter_cutoff = int(soln_filter_frac * order)
    if rhs_filter_cutoff < 0:
        rhs_filter_cutoff = int(rhs_filter_frac * order)

    if soln_filter_cutoff >= order:
        raise ValueError("Invalid setting for solution filter (cutoff >= order).")
    if rhs_filter_cutoff >= order:
        raise ValueError("Invalid setting for RHS filter (cutoff >= order).")

    from mirgecom.filter import (
        exponential_mode_response_function as xmrfunc,
        filter_modally
    )
    soln_frfunc = partial(xmrfunc, alpha=soln_filter_alpha,
                          filter_order=soln_filter_order)
    rhs_frfunc = partial(xmrfunc, alpha=rhs_filter_alpha,
                         filter_order=rhs_filter_order)

    def filter_cv(cv):
        return filter_modally(dcoll, soln_filter_cutoff, soln_frfunc, cv,
                              dd=dd_vol_fluid)

    def filter_rhs(rhs):
        return filter_modally(dcoll, rhs_filter_cutoff, rhs_frfunc, rhs,
                              dd=dd_vol_fluid)

    filter_cv_compiled = actx.compile(filter_cv)

    if soln_nfilter >= 0 and rank == 0:
        logger.info("Solution filtering settings:")
        logger.info(f" - filter every {soln_nfilter} steps")
        logger.info(f" - filter alpha  = {soln_filter_alpha}")
        logger.info(f" - filter cutoff = {soln_filter_cutoff}")
        logger.info(f" - filter order  = {soln_filter_order}")
    if use_rhs_filter and rank == 0:
        logger.info("RHS filtering settings:")
        logger.info(f" - filter alpha  = {rhs_filter_alpha}")
        logger.info(f" - filter cutoff = {rhs_filter_cutoff}")
        logger.info(f" - filter order  = {rhs_filter_order}")

    limiter_func = None
    if use_species_limiter:
        limiter_func = limit_fluid_state

    def _create_fluid_state(cv, temperature_seed, smoothness=None):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed,
                                smoothness=smoothness,
                                limiter_func=limiter_func,
                                limiter_dd=dd_vol_fluid)

    create_fluid_state = actx.compile(_create_fluid_state)

    def update_dv(cv, temperature, smoothness):
        from mirgecom.eos import MixtureDependentVars, GasDependentVars
        if eos_type == 0:
            return GasDependentVars(
                temperature=temperature,
                pressure=eos.pressure(cv, temperature),
                speed_of_sound=eos.sound_speed(cv, temperature),
                smoothness=smoothness)
        else:
            return MixtureDependentVars(
                temperature=temperature,
                pressure=eos.pressure(cv, temperature),
                speed_of_sound=eos.sound_speed(cv, temperature),
                species_enthalpies=eos.species_enthalpies(cv, temperature),
                smoothness=smoothness)

    def update_tv(cv, dv):
        return gas_model.transport.transport_vars(cv, dv, eos)

    def update_fluid_state(cv, dv, tv):
        from mirgecom.gas_model import ViscousFluidState
        return ViscousFluidState(cv, dv, tv)

    update_dv_compiled = actx.compile(update_dv)
    update_tv_compiled = actx.compile(update_tv)
    update_fluid_state_compiled = actx.compile(update_fluid_state)

    def _create_wall_dependent_vars(wv):
        return wall_model.dependent_vars(wv)

    create_wall_dependent_vars_compiled = actx.compile(
        _create_wall_dependent_vars)

    def _get_wv(wv):
        return wv

    get_wv = actx.compile(_get_wv)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv)/cv.mass
        return actx.np.abs(
            pyro_mech.get_temperature_update_energy(e, temperature, y))

    get_temperature_update_compiled = actx.compile(get_temperature_update)

    def compute_smoothness(cv, dv, grad_cv):

        from mirgecom.fluid import velocity_gradient
        div_v = np.trace(velocity_gradient(cv, grad_cv))

        gamma = gas_model.eos.gamma(cv=cv, temperature=dv.temperature)
        r = gas_model.eos.gas_const(cv)
        c_star = actx.np.sqrt(gamma*r*(2/(gamma+1)*static_temp))
        indicator = -gamma_sc*char_length*div_v/c_star

        smoothness = actx.np.log(
            1 + actx.np.exp(theta_sc*(indicator - beta_sc)))/theta_sc
        return smoothness*gamma_sc*char_length

    compute_smoothness_compiled = actx.compile(compute_smoothness) # noqa

    if restart_filename:
        if rank == 0:
            logger.info("Restarting soln.")
        temperature_seed = restart_data["temperature_seed"]
        restart_cv = restart_data["cv"]
        restart_wv = restart_data["wv"]
        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                restart_dcoll.discr_from_dd(dd_vol_fluid)
            )
            wall_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_wall),
                restart_dcoll.discr_from_dd(dd_vol_wall)
            )
            restart_cv = fluid_connection(restart_data["cv"])
            temperature_seed = fluid_connection(restart_data["temperature_seed"])
            restart_wv = wall_connection(restart_data["wv"])

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logger.info("Initializing soln.")
        restart_cv = bulk_init(
            dcoll=dcoll, x_vec=actx.thaw(dcoll.nodes(dd_vol_fluid)), eos=eos,
            time=0)
        temperature_seed = 0*restart_cv.mass + init_temperature
        wall_mass = (
            wall_insert_rho * wall_insert_mask
            + wall_surround_rho * wall_surround_mask)
        wall_cp = (
            wall_insert_cp * wall_insert_mask
            + wall_surround_cp * wall_surround_mask)
        restart_wv = WallVars(
            mass=wall_mass,
            energy=wall_mass * wall_cp * temp_wall,
            ox_mass=0*wall_mass)

    if target_filename:
        if rank == 0:
            logger.info("Reading target soln.")
        if target_order != order:
            target_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=target_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                target_dcoll.discr_from_dd(dd_vol_fluid)
            )
            target_cv = fluid_connection(target_data["cv"])
        else:
            target_cv = target_data["cv"]
    else:
        # Set the current state from time 0
        target_cv = restart_cv

    no_smoothness = force_evaluation(actx, 0.*restart_cv.mass)
    smoothness = no_smoothness
    target_smoothness = smoothness

    restart_cv = force_evaluation(actx, restart_cv)
    target_cv = force_evaluation(actx, target_cv)
    temperature_seed = force_evaluation(actx, temperature_seed)

    current_fluid_state = create_fluid_state(restart_cv, temperature_seed,
                                             smoothness=smoothness)
    target_fluid_state = create_fluid_state(target_cv, temperature_seed,
                                            smoothness=target_smoothness)
    current_wv = force_evaluation(actx, restart_wv)
    #current_wv = get_wv(restart_wv)

    # use dummy boundaries to setup the smoothness state for the target
    wall_bnd = dd_vol_fluid.trace("isothermal_wall")
    inflow_bnd = dd_vol_fluid.trace("inflow")
    outflow_bnd = dd_vol_fluid.trace("outflow")
    inj_bnd = dd_vol_fluid.trace("injection")
    flow_bnd = dd_vol_fluid.trace("flow")
    wall_ffld_bnd = dd_vol_wall.trace("wall_farfield")

    if use_injection:
        target_boundaries = {
            flow_bnd.domain_tag:  # pylint: disable=no-member
            DummyBoundary(),
            wall_bnd.domain_tag:  # pylint: disable=no-member
            IsothermalWallBoundary()
        }
    else:
        target_boundaries = {
            inflow_bnd.domain_tag:   # pylint: disable=no-member
            DummyBoundary(),
            outflow_bnd.domain_tag:  # pylint: disable=no-member
            DummyBoundary(),
            inj_bnd.domain_tag:      # pylint: disable=no-member
            IsothermalWallBoundary(),
            wall_bnd.domain_tag:     # pylint: disable=no-member
            IsothermalWallBoundary()
        }

    def _grad_cv_operator_target(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                dd=dd_vol_fluid,
                                boundaries=target_boundaries,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_operator_target_compiled = actx.compile(_grad_cv_operator_target) # noqa

    if use_av:
        target_grad_cv = grad_cv_operator_target_compiled(
            target_fluid_state, time=0.)
        target_smoothness = compute_smoothness_compiled(
            cv=target_cv, dv=target_fluid_state.dv, grad_cv=target_grad_cv)

        target_fluid_state = create_fluid_state(cv=target_cv,
                                          temperature_seed=temperature_seed,
                                          smoothness=target_smoothness)

    stepper_state = make_obj_array([current_fluid_state.cv,
                                    temperature_seed, current_wv])

    ##################################
    # Set up the boundary conditions #
    ##################################

    from mirgecom.gas_model import project_fluid_state

    def get_target_state_on_boundary(btag):
        return project_fluid_state(
            dcoll, dd_vol_fluid,
            dd_vol_fluid.trace(btag).with_discr_tag(quadrature_tag),
            target_fluid_state, gas_model
        )

    flow_ref_state = \
        get_target_state_on_boundary("flow")

    flow_ref_state = force_evaluation(actx, flow_ref_state)

    def _target_flow_state_func(**kwargs):
        return flow_ref_state

    flow_boundary = PrescribedFluidBoundary(
        boundary_state_func=_target_flow_state_func)

    inflow_ref_state = \
        get_target_state_on_boundary("inflow")

    inflow_ref_state = force_evaluation(actx, inflow_ref_state)

    def _target_inflow_state_func(**kwargs):
        return inflow_ref_state

    inflow_boundary = PrescribedFluidBoundary(
        boundary_state_func=_target_inflow_state_func)

    outflow_ref_state = \
        get_target_state_on_boundary("outflow")

    outflow_ref_state = force_evaluation(actx, outflow_ref_state)

    def _target_outflow_state_func(**kwargs):
        return outflow_ref_state

    outflow_boundary = PrescribedFluidBoundary(
        boundary_state_func=_target_outflow_state_func)
    #outflow_pressure = 2000
    #outflow_boundary = PressureOutflowBoundary(outflow_pressure)

    if noslip:
        if adiabatic:
            fluid_wall = AdiabaticNoslipWallBoundary()
        else:
            fluid_wall = IsothermalWallBoundary(temp_wall)
    else:
        fluid_wall = AdiabaticSlipBoundary()

    wall_farfield = DirichletDiffusionBoundary(temp_wall)

    if use_injection:
        fluid_boundaries = {
            flow_bnd.domain_tag: flow_boundary,   # pylint: disable=no-member
            wall_bnd.domain_tag: fluid_wall  # pylint: disable=no-member
        }
    else:
        fluid_boundaries = {
            inflow_bnd.domain_tag: inflow_boundary,    # pylint: disable=no-member
            outflow_bnd.domain_tag: outflow_boundary,  # pylint: disable=no-member
            inj_bnd.domain_tag: fluid_wall,       # pylint: disable=no-member
            wall_bnd.domain_tag: fluid_wall       # pylint: disable=no-member
        }

    wall_boundaries = {
        wall_ffld_bnd.domain_tag: wall_farfield  # pylint: disable=no-member
    }

    # compiled wrapper for grad_cv_operator
    def _grad_cv_operator(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                boundaries=fluid_boundaries,
                                dd=dd_vol_fluid,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_operator_compiled = actx.compile(_grad_cv_operator) # noqa

    def get_production_rates(cv, temperature):
        return eos.get_production_rates(cv, temperature)

    compute_production_rates = actx.compile(get_production_rates)

    def _grad_t_operator(t, fluid_state, wall_kappa, wall_temperature):
        fluid_grad_t, wall_grad_t = coupled_grad_t_operator(
            dcoll,
            gas_model,
            dd_vol_fluid, dd_vol_wall,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_kappa, wall_temperature,
            time=t,
            quadrature_tag=quadrature_tag)
        return make_obj_array([fluid_grad_t, wall_grad_t])

    grad_t_operator = actx.compile(_grad_t_operator)

    ####################
    # Ignition Sources #
    ####################

    # if you divide by 2.355, 50% of the spark is within this diameter
    # if you divide by 6, 99% of the energy is deposited in this time
    #spark_diameter /= 2.355
    spark_diameter /= 6.0697
    spark_duration /= 6.0697

    # gaussian application in time
    def spark_time_func(t):
        expterm = actx.np.exp((-(t - spark_init_time)**2) /
                              (2*spark_duration*spark_duration))
        return expterm

    if use_ignition == 2:
        ignition_source = HeatSource(dim=dim, center=spark_center,
                                      amplitude=spark_strength,
                                      amplitude_func=spark_time_func,
                                      width=spark_diameter)
    else:
        ignition_source = SparkSource(dim=dim, center=spark_center,
                                      amplitude=spark_strength,
                                      amplitude_func=spark_time_func,
                                      width=spark_diameter)

    ##################
    # Sponge Sources #
    ##################

    # initialize the sponge field
    sponge_amp = sponge_sigma/current_dt/1000

    sponge_init = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                             amplitude=sponge_amp)
    x_vec = actx.thaw(dcoll.nodes(dd_vol_fluid))

    def _sponge_sigma(x_vec):
        return sponge_init(x_vec=x_vec)

    get_sponge_sigma = actx.compile(_sponge_sigma)
    sponge_sigma = get_sponge_sigma(x_vec)

    def _sponge_source(cv):
        """Create sponge source."""
        return sponge_sigma*(current_fluid_state.cv - cv)

    def experimental_kappa(temperature):
        return (
            1.766e-10 * temperature**3
            - 4.828e-7 * temperature**2
            + 6.252e-4 * temperature
            + 6.707e-3)

    def puma_kappa(mass_loss_frac):
        return (
            0.0988 * mass_loss_frac**2
            - 0.2751 * mass_loss_frac
            + 0.201)

    def puma_effective_surface_area(mass_loss_frac):
        # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
        # Rescale by x==0 value and rearrange
        return 1.1794e5 * (
            1
            - 0.0547736137 * mass_loss_frac
            - 0.9336950992 * mass_loss_frac**2)

    def _get_wall_kappa_fiber(mass, temperature):
        mass_loss_frac = (
            (wall_insert_rho - mass)/wall_insert_rho
            * wall_insert_mask)
        scaled_insert_kappa = (
            experimental_kappa(temperature)
            * puma_kappa(mass_loss_frac)
            / puma_kappa(0))
        return (
            scaled_insert_kappa * wall_insert_mask
            + wall_surround_kappa * wall_surround_mask)

    def _get_wall_kappa_inert(mass, temperature):
        return (
            wall_insert_kappa * wall_insert_mask
            + wall_surround_kappa * wall_surround_mask)

    def _get_wall_effective_surface_area_fiber(mass):
        mass_loss_frac = (
            (wall_insert_rho - mass)/wall_insert_rho
            * wall_insert_mask)
        return (
            puma_effective_surface_area(mass_loss_frac) * wall_insert_mask)

    def _mass_loss_rate_fiber(mass, ox_mass, temperature, eff_surf_area):
        actx = mass.array_context
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha*actx.np.sqrt(
            (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
        return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*ox_mass*k*eff_surf_area

    # inert
    if wall_material == 0:
        wall_model = WallModel(
            heat_capacity=(
                wall_insert_cp * wall_insert_mask
                + wall_surround_cp * wall_surround_mask),
            thermal_conductivity_func=_get_wall_kappa_inert)
    # non-porous
    elif wall_material == 1:
        wall_model = WallModel(
            heat_capacity=(
                wall_insert_cp * wall_insert_mask
                + wall_surround_cp * wall_surround_mask),
            thermal_conductivity_func=_get_wall_kappa_fiber,
            effective_surface_area_func=_get_wall_effective_surface_area_fiber,
            mass_loss_func=_mass_loss_rate_fiber,
            oxygen_diffusivity=wall_insert_ox_diff * wall_insert_mask)
    # porous
    elif wall_material == 2:
        wall_model = WallModel(
            heat_capacity=(
                wall_insert_cp * wall_insert_mask
                + wall_surround_cp * wall_surround_mask),
            thermal_conductivity_func=_get_wall_kappa_fiber,
            effective_surface_area_func=_get_wall_effective_surface_area_fiber,
            mass_loss_func=_mass_loss_rate_fiber,
            oxygen_diffusivity=wall_insert_ox_diff * wall_insert_mask)

    vis_timer = None
    monitor_memory = True
    monitor_performance = 2

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)

        if monitor_performance > 0:
            logmgr.add_watches([
                ("t_step.max", "| Performance:\n| \t walltime: {value:6g} s")
            ])

        if monitor_performance > 1:

            logmgr.add_watches([
                ("t_vis.max", "\n| \t visualization time: {value:6g} s\n"),
                ("t_gc.max", "| \t garbage collection time: {value:6g} s\n"),
                ("t_log.max", "| \t log walltime: {value:6g} s\n")
            ])

        if monitor_memory:
            logmgr_add_device_memory_usage(logmgr, queue)
            logmgr_add_mempool_usage(logmgr, alloc)

            logmgr.add_watches([
                ("memory_usage_python.max",
                 "| Memory:\n| \t python memory: {value:7g} Mb\n")
            ])

            try:
                logmgr.add_watches([
                    ("memory_usage_gpu.max",
                     "| \t gpu memory: {value:7g} Mb\n")
                ])
            except KeyError:
                pass

            logmgr.add_watches([
                ("memory_usage_hwm.max",
                 "| \t memory hwm: {value:7g} Mb\n"),
                ("memory_usage_mempool_managed.max",
                 "| \t mempool total: {value:7g} Mb\n"),
                ("memory_usage_mempool_active.max",
                 "| \t mempool active: {value:7g} Mb")
            ])

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
    wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall)

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    # some utility functions
    def vol_min_loc(dd_vol, x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(dcoll, dd_vol, x,
                                           initial=np.inf))[()]

    def vol_max_loc(dd_vol, x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(dcoll, dd_vol, x,
                                           initial=-np.inf))[()]

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x,
                                       initial=np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x,
                                       initial=-np.inf))[()]

    def global_range_check(dd_vol, array, min_val, max_val):
        return global_reduce(
            check_range_local(
                dcoll, dd_vol, array, min_val, max_val), op="lor")

    def my_write_status_lite(step, t, t_wall):
        status_msg = (f"\n--     step {step:9d}:"
                      f"\n----   fluid sim time {t:1.8e},"
                      f" wall sim time {t_wall:1.8e}")

        if rank == 0:
            logger.info(status_msg)

    def my_write_status(cv, dv, wall_temperature, dt, cfl_fluid, cfl_wall):
        status_msg = (f"----   dt {dt:1.3e},"
                      f" cfl_fluid {cfl_fluid:1.8f},"
                      f" cfl_wall {cfl_wall:1.8f}")

        pmin = vol_min(dd_vol_fluid, dv.pressure)
        pmax = vol_max(dd_vol_fluid, dv.pressure)
        tmin = vol_min(dd_vol_fluid, dv.temperature)
        tmax = vol_max(dd_vol_fluid, dv.temperature)
        twmin = vol_min(dd_vol_wall, wall_temperature)
        twmax = vol_max(dd_vol_wall, wall_temperature)

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(dd_vol_fluid, x),
                                      cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(dd_vol_fluid, x),
                                      cv.species_mass_fractions)

        dv_status_msg = (
            f"\n------ P       (min, max) (Pa) = ({pmin:1.9e}, {pmax:1.9e})")
        dv_status_msg += (
            f"\n------ T_fluid (min, max) (K)  = ({tmin:7g}, {tmax:7g})")
        dv_status_msg += (
            f"\n------ T_wall  (min, max) (K)  = ({twmin:7g}, {twmax:7g})")

        if eos_type == 1:
            # check the temperature convergence
            # a single call to get_temperature_update is like taking an additional
            # Newton iteration and gives us a residual
            temp_resid = get_temperature_update_compiled(
                cv, dv.temperature)/dv.temperature
            temp_err_min = vol_min(dd_vol_fluid, temp_resid)
            temp_err_max = vol_max(dd_vol_fluid, temp_resid)
            dv_status_msg += (
                f"\n------ T_resid (min, max)      = "
                f"({temp_err_min:1.5e}, {temp_err_max:1.5e})")

        for i in range(nspecies):
            dv_status_msg += (
                f"\n------ y_{species_names[i]:5s} (min, max)      = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")
        #dv_status_msg += "\n"
        status_msg += dv_status_msg

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, t_wall, fluid_state, wv, wall_kappa,
                     wall_temperature, ts_field_fluid, ts_field_wall,
                     dump_number):

        if rank == 0:
            print(f"******** Writing Visualization File {dump_number}"
                  f" at step {step},"
                  f" sim time {t:1.6e} s ********")

        cv = fluid_state.cv
        dv = fluid_state.dv
        mu = fluid_state.viscosity

        # basic viz quantities, things here are difficult (or impossible) to compute
        # in post-processing
        fluid_viz_fields = [("cv", cv),
                            ("dv", dv),
                            ("dt" if constant_cfl else "cfl", ts_field_fluid)]
        wall_viz_fields = [
            ("wv", wv),
            ("wall_kappa", wall_kappa),
            ("wall_temperature", wall_temperature),
            ("dt" if constant_cfl else "cfl", ts_field_wall)
        ]

        # extra viz quantities, things here are often used for post-processing
        if viz_level > 0:
            mach = fluid_state.speed / dv.speed_of_sound
            fluid_viz_ext = [("mach", mach),
                             ("velocity", cv.velocity)]
            fluid_viz_fields.extend(fluid_viz_ext)

            # species mass fractions
            fluid_viz_fields.extend(
                ("Y_"+species_names[i], cv.species_mass_fractions[i])
                for i in range(nspecies))

            if eos_type == 1:
                temp_resid = get_temperature_update_compiled(
                    cv, dv.temperature)/dv.temperature
                production_rates = compute_production_rates(fluid_state.cv,
                                                            fluid_state.temperature)
                fluid_viz_ext = [("temp_resid", temp_resid),
                                 ("production_rates", production_rates)]
                fluid_viz_fields.extend(fluid_viz_ext)

            if use_av:
                fluid_viz_ext = [("mu", mu)]
                fluid_viz_fields.extend(fluid_viz_ext)

            if nparts > 1:
                fluid_viz_ext = [("rank", rank)]
                fluid_viz_fields.extend(fluid_viz_ext)

        # additional viz quantities, add in some non-dimensional numbers
        if viz_level > 1:
            cell_Re = (cv.mass*cv.speed*char_length /
                fluid_state.viscosity)
            cp = gas_model.eos.heat_capacity_cp(cv, fluid_state.temperature)
            alpha_heat = fluid_state.thermal_conductivity/cp/fluid_state.viscosity
            cell_Pe_heat = char_length*cv.speed/alpha_heat
            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )
            cell_Pe_mass = char_length*cv.speed/d_alpha_max
            # these are useful if our transport properties
            # are not constant on the mesh
            # prandtl
            # schmidt_number
            # damkohler_number

            viz_ext = [("Re", cell_Re),
                       ("Pe_mass", cell_Pe_mass),
                       ("Pe_heat", cell_Pe_heat)]
            fluid_viz_fields.extend(viz_ext)

            cell_alpha = wall_model.thermal_diffusivity(
                wv.mass, wall_temperature, wall_kappa)

            viz_ext = [
                       ("alpha", cell_alpha)]
            wall_viz_fields.extend(viz_ext)

        # debbuging viz quantities, things here are used for diagnosing run issues
        if viz_level > 2:
            from mirgecom.fluid import (
                velocity_gradient,
                species_mass_fraction_gradient
            )
            """
            ns_rhs, grad_cv, grad_t = \
                ns_operator(dcoll, state=fluid_state, time=t,
                            boundaries=boundaries, gas_model=gas_model,
                            return_gradients=True)
            """
            grad_cv = grad_cv_operator_compiled(fluid_state,
                                                time=t)
            grad_v = velocity_gradient(cv, grad_cv)
            grad_y = species_mass_fraction_gradient(cv, grad_cv)

            grad_temperature = grad_t_operator(
                dv.temperature, fluid_state, wall_kappa, wall_temperature)
            fluid_grad_temperature = grad_temperature[0]
            wall_grad_temperature = grad_temperature[1]

            #viz_ext = [("rhs", ns_rhs),
            viz_ext = [("sponge_sigma", sponge_sigma),
                       ("grad_temperature", fluid_grad_temperature),
                       ("grad_v_x", grad_v[0]),
                       ("grad_v_y", grad_v[1])]
            if dim == 3:
                viz_ext.extend(("grad_v_z", grad_v[2]))

            viz_ext.extend(("grad_Y_"+species_names[i], grad_y[i])
                           for i in range(nspecies))
            fluid_viz_fields.extend(viz_ext)

            viz_ext = [("grad_temperature", wall_grad_temperature)]
            wall_viz_fields.extend(viz_ext)

        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=dump_number, t=t,
            overwrite=True, comm=comm, vis_timer=vis_timer)
        write_visfile(
            dcoll, wall_viz_fields, wall_visualizer,
            vizname=vizname+"-wall", step=dump_number, t=t_wall,
            overwrite=True, comm=comm, vis_timer=vis_timer)

        if rank == 0:
            print("******** Done Writing Visualization File ********")

    def my_write_restart(step, t, t_wall, state):
        if rank == 0:
            print(f"******** Writing Restart File at step {step}, "
                  f"sim time {t:1.6e} s ********")

        cv, tseed, wv = state
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "nspecies": nspecies,
                "wv": wv,
                "t": t,
                "t_wall": t_wall,
                "step": step,
                "order": order,
                "last_viz_interval": last_viz_interval,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

        if rank == 0:
            print("******** Done Writing Restart File ********")

    def my_health_check(fluid_state, wall_temperature):
        health_error = False
        cv = fluid_state.cv
        dv = fluid_state.dv

        if check_naninf_local(dcoll, dd_vol_fluid, dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_naninf_local(dcoll, dd_vol_wall, wall_temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in wall temperature data.")

        if global_range_check(dd_vol_fluid, dv.pressure,
                              health_pres_min, health_pres_max):
            health_error = True
            p_min = vol_min(dd_vol_fluid, dv.pressure)
            p_max = vol_max(dd_vol_fluid, dv.pressure)
            logger.info(f"Pressure range violation: "
                        f"Simulation Range ({p_min=}, {p_max=}) "
                        f"Specified Limits ({health_pres_min=}, {health_pres_max=})")

        if global_range_check(dd_vol_fluid, dv.temperature,
                              health_temp_min, health_temp_max):
            health_error = True
            t_min = vol_min(dd_vol_fluid, dv.temperature)
            t_max = vol_max(dd_vol_fluid, dv.temperature)
            logger.info(f"Temperature range violation: "
                        f"Simulation Range ({t_min=}, {t_max=}) "
                        f"Specified Limits ({health_temp_min=}, {health_temp_max=})")

        if global_range_check(dd_vol_wall, wall_temperature,
                              health_temp_min, health_temp_max):
            health_error = True
            t_min = vol_min(dd_vol_wall, wall_temperature)
            t_max = vol_max(dd_vol_wall, wall_temperature)
            logger.info(f"Wall temperature range violation: "
                        f"Simulation Range ({t_min=}, {t_max=}) "
                        f"Specified Limits ({health_temp_min=}, {health_temp_max=})")

        for i in range(nspecies):
            if global_range_check(dd_vol_fluid, cv.species_mass_fractions[i],
                                  health_mass_frac_min, health_mass_frac_max):
                health_error = True
                y_min = vol_min(dd_vol_fluid, cv.species_mass_fractions[i])
                y_max = vol_max(dd_vol_fluid, cv.species_mass_fractions[i])
                logger.info(f"Species mass fraction range violation. "
                            f"{species_names[i]}: ({y_min=}, {y_max=})")

        if eos_type == 1:
            # check the temperature convergence
            # a single call to get_temperature_update is like taking an additional
            # Newton iteration and gives us a residual
            temp_resid = get_temperature_update_compiled(
                cv, dv.temperature)/dv.temperature
            temp_err = vol_max(dd_vol_fluid, temp_resid)
            if temp_err > pyro_temp_tol:
                health_error = True
                logger.info(f"Temperature is not converged "
                            f"{temp_err=} > {pyro_temp_tol}.")

        return health_error

    def my_get_viscous_timestep(dcoll, fluid_state):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        dcoll: grudge.eager.EagerDGDiscretization
            the discretization to use
        fluid_state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.dof_array.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        nu = 0
        d_alpha_max = 0

        if fluid_state.is_viscous:
            from mirgecom.viscous import get_local_max_species_diffusivity
            nu = fluid_state.viscosity/fluid_state.mass_density
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )

        return (
            char_length / (fluid_state.wavespeed
            + ((nu + d_alpha_max) / char_length))
        )

    def my_get_wall_timestep(dcoll, wv, wall_kappa, wall_temperature):
        """Routine returns the the node-local maximum stable thermal timestep.

        Parameters
        ----------
        dcoll: grudge.eager.EagerDGDiscretization
            the discretization to use

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """

        return (
            char_length_wall*char_length_wall
            / (
                wall_time_scale
                * actx.np.maximum(
                    wall_model.thermal_diffusivity(
                        wv.mass, wall_temperature, wall_kappa),
                    wall_model.oxygen_diffusivity)))

    def _my_get_timestep_wall(
            dcoll, wv, wall_kappa, wall_temperature, t, dt, cfl, t_final,
            constant_cfl=False, wall_dd=DD_VOLUME_ALL):
        """Return the maximum stable timestep for a typical heat transfer simulation.

        This routine returns *dt*, the users defined constant timestep, or *max_dt*,
        the maximum domain-wide stability-limited timestep for a fluid simulation.

        .. important::
            This routine calls the collective: :func:`~grudge.op.nodal_min` on the
            inside which makes it domain-wide regardless of parallel domain
            decomposition. Thus this routine must be called *collectively*
            (i.e. by all ranks).

        Two modes are supported:
            - Constant DT mode: returns the minimum of (t_final-t, dt)
            - Constant CFL mode: returns (cfl * max_dt)

        Parameters
        ----------
        dcoll
            Grudge discretization or discretization collection?
        t: float
            Current time
        t_final: float
            Final time
        dt: float
            The current timestep
        cfl: float
            The current CFL number
        constant_cfl: bool
            True if running constant CFL mode

        Returns
        -------
        float
            The dt (contant cfl) or cfl (constant dt) at every point in the mesh
        float
            The minimum stable cfl based on conductive heat transfer
        float
            The maximum stable DT based on conductive heat transfer
        """
        actx = wall_kappa.array_context
        mydt = dt
        if constant_cfl:
            from grudge.op import nodal_min
            ts_field = cfl*my_get_wall_timestep(
                dcoll=dcoll, wv=wv, wall_kappa=wall_kappa,
                wall_temperature=wall_temperature)
            mydt = actx.to_numpy(
                nodal_min(
                    dcoll, wall_dd, ts_field, initial=np.inf))[()]
        else:
            from grudge.op import nodal_max
            ts_field = mydt/my_get_wall_timestep(
                dcoll=dcoll, wv=wv, wall_kappa=wall_kappa,
                wall_temperature=wall_temperature)
            cfl = actx.to_numpy(
                nodal_max(
                    dcoll, wall_dd, ts_field, initial=0.))[()]

        return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    my_get_timestep_wall = _my_get_timestep_wall

    def _my_get_timestep(
            dcoll, fluid_state, t, dt, cfl, t_final, constant_cfl=False,
            fluid_dd=DD_VOLUME_ALL):
        """Return the maximum stable timestep for a typical fluid simulation.

        This routine returns *dt*, the users defined constant timestep, or *max_dt*,
        the maximum domain-wide stability-limited timestep for a fluid simulation.

        .. important::
            This routine calls the collective: :func:`~grudge.op.nodal_min` on the
            inside which makes it domain-wide regardless of parallel domain
            decomposition. Thus this routine must be called *collectively*
            (i.e. by all ranks).

        Two modes are supported:
            - Constant DT mode: returns the minimum of (t_final-t, dt)
            - Constant CFL mode: returns (cfl * max_dt)

        Parameters
        ----------
        dcoll
            Grudge discretization or discretization collection?
        fluid_state: :class:`~mirgecom.gas_model.FluidState`
            The full fluid conserved and thermal state
        t: float
            Current time
        t_final: float
            Final time
        dt: float
            The current timestep
        cfl: float
            The current CFL number
        alpha: :class:`~meshmode.dof_array.DOFArray`
            The contribution from artificial viscosity
        constant_cfl: bool
            True if running constant CFL mode

        Returns
        -------
        float
            The dt (contant cfl) or cfl (constant dt) at every point in the mesh
        float
            The minimum stable cfl based on a viscous fluid.
        float
            The maximum stable DT based on a viscous fluid.
        """
        mydt = dt
        if constant_cfl:
            from grudge.op import nodal_min
            ts_field = cfl*my_get_viscous_timestep(
                dcoll=dcoll, fluid_state=fluid_state)
            mydt = fluid_state.array_context.to_numpy(nodal_min(
                    dcoll, fluid_dd, ts_field, initial=np.inf))[()]
        else:
            from grudge.op import nodal_max
            ts_field = mydt/my_get_viscous_timestep(
                dcoll=dcoll, fluid_state=fluid_state)
            cfl = fluid_state.array_context.to_numpy(nodal_max(
                    dcoll, fluid_dd, ts_field, initial=0.))[()]

        return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    my_get_timestep = _my_get_timestep

    def _check_time(time, dt, interval, interval_type):
        toler = 1.e-6
        status = False

        dumps_so_far = math.floor((time-t_start)/interval)

        # dump if we just passed a dump interval
        if interval_type == 2:
            time_till_next = (dumps_so_far + 1)*interval - time
            steps_till_next = math.floor(time_till_next/dt)

            # reduce the timestep going into a dump to avoid a big variation in dt
            if steps_till_next < 5:
                dt_new = dt
                extra_time = time_till_next - steps_till_next*dt
                #if actx.np.abs(extra_time/dt) > toler:
                if abs(extra_time/dt) > toler:
                    dt_new = time_till_next/(steps_till_next + 1)

                if steps_till_next < 1:
                    dt_new = time_till_next

                dt = dt_new

            time_from_last = time - t_start - (dumps_so_far)*interval
            if abs(time_from_last/dt) < toler:
                status = True
        else:
            time_from_last = time - t_start - (dumps_so_far)*interval
            if time_from_last < dt:
                status = True

        return status, dt, dumps_so_far + last_viz_interval

    #check_time = _check_time

    def my_pre_step(step, t, dt, state):
        if step % 10 == 0:
            with gc_timer.start_sub_timer():
                from warnings import warn
                warn("Running gc.collect() to work around memory growth issue "
                     "https://github.com/illinois-ceesd/mirgecom/issues/839")
                import gc
                gc.collect()

        # Filter *first* because this will be most straightfwd to
        # understand and move. For this to work, this routine
        # must pass back the filtered CV in the state.
        if check_step(step=step, interval=soln_nfilter):
            cv, tseed, wv = state
            cv = filter_cv_compiled(cv)
            state = make_obj_array([cv, tseed, wv])

        cv, tseed, wv = state
        fluid_state = create_fluid_state(cv=cv,
                                         temperature_seed=tseed,
                                         smoothness=no_smoothness)
        wdv = create_wall_dependent_vars_compiled(wv)

        try:

            if logmgr:
                logmgr.tick_before()

            # disable non-constant dt timestepping for now
            # re-enable when we're ready

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
            next_dump_number = step

            state = make_obj_array([cv, fluid_state.temperature, wv])

            if any([do_viz, do_restart, do_health, do_status]):

                if use_av:
                    # recompute the dv to have the correct smoothness
                    if do_viz:
                        # use the divergence to compute the smoothness field
                        grad_cv = grad_cv_operator_compiled(fluid_state,
                                                            time=t)
                        # limited cv here to compute smoothness
                        smoothness = compute_smoothness_compiled(
                            cv=cv, dv=fluid_state.dv,
                            grad_cv=grad_cv)

                        # unlimited cv here as that is what gets written
                        dv_new = update_dv_compiled(
                            cv=cv, temperature=fluid_state.temperature,
                            smoothness=smoothness)
                        tv_new = update_tv_compiled(cv=cv, dv=dv_new)
                        fluid_state = update_fluid_state_compiled(
                            cv=cv, dv=dv_new, tv=tv_new)

                #print(wv)
                #wv = force_evaluation(actx, wv)
                #print(wv)
                # pass through, removes a bunch of tagging to avoid recomplie
                wv = get_wv(wv)
                #print(wv)

                if not force_eval:
                    fluid_state = force_evaluation(actx, fluid_state)
                    wv = force_evaluation(actx, wv)

                dv = fluid_state.dv

                ts_field_fluid, cfl_fluid, dt_fluid = my_get_timestep(
                    dcoll=dcoll, fluid_state=fluid_state,
                    t=t, dt=dt, cfl=current_cfl, t_final=t_final,
                    constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)

                ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(
                    dcoll=dcoll, wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature, t=t, dt=dt,
                    cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
                    wall_dd=dd_vol_wall)

            """
            # adjust time for constant cfl, use the smallest timescale
            dt_const_cfl = 100.
            if constant_cfl:
                dt_const_cfl = np.minimum(dt_fluid, dt_wall)

            # adjust time to hit the final requested time
            t_remaining = max(0, t_final - t)

            if viz_interval_type == 0:
                dt = np.minimum(t_remaining, current_dt)
            else:
                dt = np.minimum(t_remaining, dt_const_cfl)

            # update our I/O quantities
            cfl_fluid = dt*cfl_fluid/dt_fluid
            cfl_wall = dt*cfl_wall/dt_wall
            ts_field_fluid = dt*ts_field_fluid/dt_fluid
            ts_field_wall = dt*ts_field_wall/dt_wall

            if viz_interval_type == 1:
                do_viz, dt, next_dump_number = check_time(
                    time=t, dt=dt, interval=t_viz_interval,
                    interval_type=viz_interval_type)
            elif viz_interval_type == 2:
                dt_sav = dt
                do_viz, dt, next_dump_number = check_time(
                    time=t, dt=dt, interval=t_viz_interval,
                    interval_type=viz_interval_type)

                # adjust cfl by dt
                cfl_fluid = dt*cfl_fluid/dt_sav
                cfl_wall = dt*cfl_wall/dt_sav
            else:
                do_viz = check_step(step=step, interval=nviz)
                next_dump_number = step
            """

            t_wall = t_wall_start + (step - first_step)*dt*wall_time_scale
            my_write_status_lite(step=step, t=t, t_wall=t_wall)

            # these status updates require global reductions on state data
            if do_status:
                my_write_status(cv=cv, dv=dv, wall_temperature=wdv.temperature,
                                dt=dt, cfl_fluid=cfl_fluid, cfl_wall=cfl_wall)

            if do_health:
                health_errors = global_reduce(
                    my_health_check(fluid_state, wall_temperature=wdv.temperature),
                    op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, t_wall=t_wall, state=state)

            if do_viz:
                my_write_viz(
                    step=step, t=t, t_wall=t_wall, fluid_state=fluid_state,
                    wv=wv, wall_kappa=wdv.thermal_conductivity,
                    wall_temperature=wdv.temperature, ts_field_fluid=ts_field_fluid,
                    ts_field_wall=ts_field_wall,
                    dump_number=next_dump_number)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")

            if viz_interval_type == 0:
                dump_number = step
            else:
                dump_number = (math.floor((t-t_start)/t_viz_interval) +
                    last_viz_interval)

            my_write_viz(
                step=step, t=t, t_wall=t_wall, fluid_state=fluid_state,
                wv=wv, wall_kappa=wdv.thermal_conductivity,
                wall_temperature=wdv.temperature, ts_field_fluid=ts_field_fluid,
                ts_field_wall=ts_field_wall,
                dump_number=dump_number)
            my_write_restart(step=step, t=t, t_wall=t_wall, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        cv, tseed, wv = state

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed,
                                       smoothness=no_smoothness,
                                       limiter_func=limiter_func,
                                       limiter_dd=dd_vol_fluid)

        if use_av:
            # use the divergence to compute the smoothness field
            grad_fluid_cv = grad_cv_operator(
                dcoll, gas_model, fluid_boundaries, fluid_state,
                dd=dd_vol_fluid,
                time=t, quadrature_tag=quadrature_tag,
                comm_tag=_SmoothnessCVGradCommTag)
            smoothness = compute_smoothness(cv=cv, dv=fluid_state.dv,
                                            grad_cv=grad_fluid_cv)

            dv_new = update_dv(cv=cv, temperature=fluid_state.temperature,
                               smoothness=smoothness)
            tv_new = update_tv(cv=cv, dv=dv_new)
            fluid_state = update_fluid_state(cv=cv, dv=dv_new, tv=tv_new)

        # update wall model
        wdv = wall_model.dependent_vars(wv)

        # Temperature seed RHS (keep tseed updated)
        tseed_rhs = fluid_state.temperature - tseed

        """
        # Steps common to NS and AV (and wall model needs grad(temperature))
        operator_fluid_states = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, boundaries, quadrature_tag)

        grad_fluid_cv = grad_cv_operator(
            dcoll, gas_model, boundaries, fluid_state,
            quadrature_tag=quadrature_tag,
            operator_states_quad=operator_fluid_states)
        """

        ns_rhs, wall_energy_rhs = coupled_ns_heat_operator(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=dd_vol_fluid, wall_dd=dd_vol_wall,
            fluid_boundaries=fluid_boundaries,
            wall_boundaries=wall_boundaries,
            interface_noslip=noslip,
            #interface_noslip=True,
            inviscid_numerical_flux_func=inviscid_numerical_flux_func,
            fluid_state=fluid_state,
            wall_kappa=wdv.thermal_conductivity,
            wall_temperature=wdv.temperature,
            time=t,
            wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag)

        chem_rhs = 0*cv
        if use_combustion:  # conditionals evaluated only once at compile time
            chem_rhs =  \
                eos.get_species_source_terms(cv, temperature=fluid_state.temperature)

        ignition_rhs = 0*cv
        if use_ignition > 0:
            ignition_rhs = ignition_source(x_vec=x_vec, state=fluid_state,
                                           eos=gas_model.eos, time=t)/current_dt

        sponge_rhs = 0*cv
        if use_sponge:
            sponge_rhs = _sponge_source(cv=cv)

        fluid_rhs = ns_rhs + chem_rhs + sponge_rhs + ignition_rhs

        #wall_mass_rhs = -wall_model.mass_loss_rate(wv)
        # wall mass loss
        wall_mass_rhs = 0.*wv.mass
        if use_wall_mass:
            wall_mass_rhs = -wall_model.mass_loss_rate(
                mass=wv.mass, ox_mass=wv.ox_mass,
                temperature=wdv.temperature)

        # wall oxygen diffusion
        #wall_ox_mass_rhs = 0.*wv.ox_mass
        wall_ox_mass_rhs = 0.*wv.mass
        if use_wall_ox:
            if nspecies == 0:
                fluid_ox_mass = cv.mass*0.
            elif nspecies > 3:
                fluid_ox_mass = cv.species_mass[i_ox]
            else:
                fluid_ox_mass = mf_o2*cv.species_mass[0]
            pairwise_ox = {
                (dd_vol_fluid, dd_vol_wall):
                    (fluid_ox_mass, wv.ox_mass)}
            pairwise_ox_tpairs = inter_volume_trace_pairs(
                dcoll, pairwise_ox, comm_tag=_OxCommTag)
            ox_tpairs = pairwise_ox_tpairs[dd_vol_fluid, dd_vol_wall]
            wall_ox_boundaries = {
                wall_ffld_bnd.domain_tag:  # pylint: disable=no-member
                DirichletDiffusionBoundary(0)}

            wall_ox_boundaries.update({
                tpair.dd.domain_tag:
                DirichletDiffusionBoundary(
                    op.project(dcoll, tpair.dd,
                               tpair.dd.with_discr_tag(quadrature_tag), tpair.ext))
                for tpair in ox_tpairs})

            wall_ox_mass_rhs = diffusion_operator(
                dcoll, wall_model.oxygen_diffusivity, wall_ox_boundaries, wv.ox_mass,
                penalty_amount=wall_penalty_amount,
                quadrature_tag=quadrature_tag, dd=dd_vol_wall,
                comm_tag=_WallOxDiffCommTag)

        wall_rhs = wall_time_scale * WallVars(
            mass=wall_mass_rhs,
            energy=wall_energy_rhs,
            ox_mass=wall_ox_mass_rhs)

        if use_wall_ox:
            # Solve a diffusion equation in the fluid too just to ensure all MPI
            # sends/recvs from inter_volume_trace_pairs are in DAG
            # FIXME: this is dumb
            reverse_ox_tpairs = pairwise_ox_tpairs[dd_vol_wall, dd_vol_fluid]
            fluid_ox_boundaries = {
                bdtag: DirichletDiffusionBoundary(0)
                for bdtag in fluid_boundaries}
            fluid_ox_boundaries.update({
                tpair.dd.domain_tag:
                DirichletDiffusionBoundary(
                    op.project(dcoll, tpair.dd,
                               tpair.dd.with_discr_tag(quadrature_tag), tpair.ext))
                for tpair in reverse_ox_tpairs})

            fluid_dummy_ox_mass_rhs = diffusion_operator(
                dcoll, 0, fluid_ox_boundaries, fluid_ox_mass,
                quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                comm_tag=_FluidOxDiffCommTag)

            fluid_rhs = fluid_rhs + 0*fluid_dummy_ox_mass_rhs

        # Use a spectral filter on the RHS
        if use_rhs_filter:
            fluid_rhs = filter_rhs(fluid_rhs)

        return make_obj_array([fluid_rhs, tseed_rhs, wall_rhs])

    """
    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)
    """

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      force_eval=force_eval,
                      state=stepper_state)
    current_cv, tseed, current_wv = stepper_state
    current_fluid_state = create_fluid_state(current_cv, tseed,
                                             no_smoothness)
    current_wdv = create_wall_dependent_vars_compiled(current_wv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    if use_av:
        # use the divergence to compute the smoothness field
        current_grad_cv = grad_cv_operator_compiled(
            fluid_state=current_fluid_state, time=current_t)
        smoothness = compute_smoothness_compiled(
            cv=current_cv, dv=current_fluid_state.dv, grad_cv=current_grad_cv)

        current_fluid_state = create_fluid_state(cv=current_cv,
                                           temperature_seed=tseed,
                                           smoothness=smoothness)

    final_dv = current_fluid_state.dv
    ts_field_fluid, cfl, dt = my_get_timestep(dcoll=dcoll,
        fluid_state=current_fluid_state,
        t=current_t, dt=current_dt, cfl=current_cfl,
        t_final=t_final, constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)
    ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(dcoll=dcoll,
        wv=current_wv, wall_kappa=current_wdv.thermal_conductivity,
        wall_temperature=current_wdv.temperature, t=current_t, dt=current_dt,
        cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
        wall_dd=dd_vol_wall)
    current_t_wall = t_wall_start + (current_step - first_step)*dt*wall_time_scale
    my_write_status_lite(step=current_step, t=current_t,
                         t_wall=current_t_wall)
    my_write_status(dv=final_dv, cv=current_cv,
                    wall_temperature=current_wdv.temperature,
                    dt=dt, cfl_fluid=cfl, cfl_wall=cfl_wall)

    if viz_interval_type == 0:
        dump_number = current_step
    else:
        dump_number = (math.floor((current_t - t_start)/t_viz_interval) +
            last_viz_interval)

    my_write_viz(
        step=current_step, t=current_t, t_wall=current_t_wall,
        fluid_state=current_fluid_state,
        wv=current_wv, wall_kappa=current_wdv.thermal_conductivity,
        wall_temperature=current_wdv.temperature,
        ts_field_fluid=ts_field_fluid,
        ts_field_wall=ts_field_wall,
        dump_number=dump_number)
    my_write_restart(step=current_step, t=current_t, t_wall=current_t_wall,
                     state=stepper_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 2*current_dt
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    #root_logger = logging.getLogger()

    #logging.debug("A DEBUG message")
    #logging.info("An INFO message")
    #logging.warning("A WARNING message")
    #logging.error("An ERROR message")
    #logging.critical("A CRITICAL message")

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-g", "--logpath", type=ascii, dest="log_path", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()

    # for writing output
    casename = "prediction"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")
    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    log_path = "log_data"
    if args.log_path:
        log_path = args.log_path.replace("'", "")

    print(f"Running {sys.argv[0]}\n")

    main(restart_filename=restart_filename, target_filename=target_filename,
         user_input_file=input_file, log_path=log_path,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
