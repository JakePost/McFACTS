from typing import Any

from mcfacts.inputs import ReadInputs

defaults = {
    # IO Options
    "verbose": False, # Print all debug messages
    "show_timeline_progress": False, # Shows a progress bar for the active timeline
    "override_files": False, # Override any output files that exist, throws errors otherwise
    "save_state": False, # Pickle and save the entire state of a galaxy after each population or timeline is run
    "save_each_timestep": False, # Pickle and save the state of a galaxy for each timestep during a simulation timeline

    # Black Hole and Disk Parameters
    "smbh_mass": 1.e8,  # Supermassive black hole mass (solar masses)
    "flag_use_pagn": True,  # Enable or disable the pAGN program
    "disk_model_name": "sirko_goodman",  # Disk model options: sirko_goodman, thompson_etal
    "disk_radius_trap": 700.0,  # Migration trap radius (gravitational radii)
    "disk_radius_outer": 50000.0,  # Outer radius of the disk (gravitational radii)
    "disk_alpha_viscosity": 0.01,  # Disk viscosity parameter
    "disk_aspect_ratio_avg": 0.03,  # Average disk height relative to its radius
    "disk_bh_torque_condition": 0.1,  # Fraction of mass accreted before spin alignment
    "disk_bh_eddington_ratio": 1.0,  # Eddington accretion ratio
    "disk_bh_orb_ecc_max_init": 0.3,  # Initial maximum orbital eccentricity
    "disk_radius_capture_outer": 2.e3,  # Outer radius for capture (gravitational radii)
    "disk_bh_pro_orb_ecc_crit": 0.01,  # Critical eccentricity for circularized orbits
    "inner_disk_outer_radius": 50.0,  # Outer radius of the inner disk (gravitational radii)
    "disk_radius_max_pc": 0.,  # Maximum disk size in parsecs (negative for off)
    "disk_inner_stable_circ_orb": 6.0,  # Innermost stable circular orbit (gravitational radii)
    "initial_binary_orbital_ecc": 0.01, # The initial common orbital eccentricity around the SMBH to be assumed of a binary when it forms
    "fraction_bin_retro": 0, # Fraction of formed binaries that should be retrograde orbiters
    "torque_prescription": "paardekooper", # Torque prescription ('old','paardekooper','jimenez_masset': Paaardekooper default; Jimenez-Masset option)
    "flag_phenom_turb": False, # Phenomenological turbulence
    "phenom_turb_centroid":  0., # Centroid of Gaussian w.r.t. to migrating BH. (default = 0.)
    "phenom_turb_std_dev": 1.0, # Variance of Gaussian around Centroid (default=0.1)
    "flag_use_surrogate": False, # TODO: Add description
    "mean_harden_energy_delta": 0.9,
    "var_harden_energy_delta": 0.025,

    # Star stuff
    "flag_add_stars": True,  # Enable or disable stars
    "disk_star_mass_min_init": 5.0,
    "disk_star_mass_max_init": 40,
    "rstar_rhill_exponent_ratio": 2.0,

    # Nuclear Star Cluster Parameters
    "nsc_radius_outer": 5.0,  # Outer radius of the Nuclear Star Cluster (pc)
    "nsc_mass": 3.e7,  # Mass of the Nuclear Star Cluster (solar masses)
    "nsc_radius_crit": 0.25,  # Critical radius of the Nuclear Star Cluster (pc)
    "nsc_ratio_bh_num_star_num": 1.e-3,  # Ratio of BH to stars in the Nuclear Star Cluster
    "nsc_ratio_bh_mass_star_mass": 10.0,  # Ratio of BH mass to star mass in the Nuclear Star Cluster
    "nsc_density_index_inner": 1.75,  # Inner density profile index
    "nsc_density_index_outer": 2.5,  # Outer density profile index
    "nsc_imf_star_powerlaw_index": 2.35,
    "nsc_imf_bh_mode": 10.0,  # Mode of the BH mass initial distribution (solar masses)
    "nsc_imf_bh_powerlaw_index": 2.0,  # Index of the BH mass initial distribution
    "nsc_imf_bh_mass_max": 40.0,  # Maximum BH mass (solar masses)
    "nsc_bh_spin_dist_mu": 0.0,  # Mean of the BH spin magnitude distribution
    "nsc_bh_spin_dist_sigma": 0.1,  # Standard deviation of the BH spin magnitude distribution
    "nsc_spheroid_normalization": 1.0,  # Sphericity normalization factor
    "nsc_star_spin_dist_mu": 100,
    "nsc_star_spin_dist_sigma": 20,
    "nsc_star_metallicity_x_init": 0.7274,
    "nsc_star_metallicity_y_init": 0.2638,
    "nsc_star_metallicity_z_init": 0.0088,
    "nsc_imf_bh_method": "default",

    # Simulation Parameters
    "timestep_duration_yr": 1.e4,  # Duration of each timestep (years)
    "timestep_num": 100,  # Number of timesteps
    "capture_time_yr": 1.e5,  # Time between disk captures (years)
    "galaxy_num": 1,  # DEPRECATED: Number of iterations of the simulation
    "save_snapshots": 0,  # Whether to save snapshots (0 for off)

    # Feedback and Interaction Switches
    "flag_thermal_feedback": 1,  # Enable thermal feedback for BH torque modifications
    "flag_orb_ecc_damping": 1,  # Enable orbital eccentricity damping
    "flag_dynamic_enc": 1,  # Enable dynamical interactions
    "delta_energy_strong": 0.1,  # Change in orbital energy/eccentricity post-interaction
    "mass_pile_up": 35.0,  # Mass pile-up term (solar masses)

    # Values that should not be changed (See static_settings below)
    "disk_bh_eddington_mass_growth_rate": 2.3e-8,
    "disk_bh_spin_resolution_min": 0.02,
    "min_bbh_gw_separation": 2.0,
    "agn_redshift": 0.1,

    # Filing cabinet array names
    "bh_array_name": "blackholes_unsort",
    "bh_inner_disk_array_name": "blackholes_inner_disk",
    "bh_prograde_array_name": "blackholes_prograde",
    "bh_retrograde_array_name": "blackholes_retrograde",
    "stars_prograde_array_name": "stars_prograde",
    "stars_retrograde_array_name": "stars_retrograde",
    "stars_merged_array_name": "stars_merged",
    "bbh_array_name": "blackholes_binary",
    "bbh_gw_array_name": "blackholes_binary_gw",
    "bbh_merged_array_name": "blackholes_merged"
}

# Static values that we do not want the user to change (Sorry user)
static_settings = [
    "disk_bh_eddington_mass_growth_rate",
    "disk_bh_spin_resolution_min",
    "min_bbh_gw_separation",
    "agn_redshift"
]


class SettingsManager:
    """
    Manages settings by providing a mechanism to override default values with user-defined values.

    Attributes
    ----------
    settings_overrides : dict[str, Any]
        A dictionary containing user-provided overrides for the default settings.
    """

    def __init__(self, settings_overrides: dict[str, Any] = None):
        """
        Initializes the SettingsManager with a dictionary of overrides.

        Parameters
        ----------
        settings_overrides : dict[str, Any]
            A dictionary containing user-provided overrides for specific settings.
        """
        self.settings_overrides: dict[str, Any] = dict() if settings_overrides is None else settings_overrides
        self.settings_defaults = defaults
        self.settings_finals = dict()

        for key, value in self.settings_defaults.items():
            if (key in self.settings_overrides) and (key not in static_settings):
                self.settings_finals[key] = self.settings_overrides[key]
            else:
                self.settings_finals[key] = value

    def __getattr__(self, item: str) -> Any:
        """
        Retrieves the value of a setting, checking for overrides first.
        If no override exists, the default value is returned. If the key
        is not found in either, an AttributeError is raised.

        Parameters
        ----------
        item : str
            The name of the setting to retrieve.

        Returns
        -------
        Any
            The value of the setting, either overridden or default.

        Raises
        ------
        AttributeError
            If the setting is not found in both the overrides and the defaults.
        """
        if item.startswith("__"):  # this allows for deepcopy
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(item)
            )

        try:
            return self.settings_finals[item]
        except KeyError as exception:
            raise AttributeError(f"SettingsManager has no key {item!r}") from exception


class AGNDisk:
    def __init__(self, settings: SettingsManager):
        # TODO: More advanced handling?

        (
            self.disk_surface_density,
            self.disk_aspect_ratio,
            self.disk_opacity,
            self.disk_sound_speed,
            self.disk_density,
            self.disk_pressure_grad,
            self.disk_omega,
            self.disk_surface_density_log,
            self.temp_func,
            self.disk_dlog10surfdens_dlog10R_func,
            self.disk_dlog10temp_dlog10R_func,
            self.disk_dlog10pressure_dlog10R_func
        ) = ReadInputs.construct_disk_interp(
            settings.smbh_mass,
            settings.disk_radius_outer,
            settings.disk_model_name,
            settings.disk_alpha_viscosity,
            settings.disk_bh_eddington_ratio,
            disk_radius_max_pc=settings.disk_radius_max_pc,
            flag_use_pagn=settings.flag_use_pagn,
            verbose= 1 if settings.verbose else 0
        )