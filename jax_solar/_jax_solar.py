# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Top of atmosphere incident solar radiation.

This code was inspired by pysolar, however that library is intended as a surface
of Earth solar radiation model that includes atmospheric scattering effects. Of
particular note, the constants used to compute solar irradiance are different.
References included below.

Another major difference is that this module represents time using radians,
which greatly reduces the number of conversions by factors of pi.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Self

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np


Array = np.ndarray | jnp.ndarray
Numeric = float | int | Array


DAYS_PER_YEAR = 365.2425  # per the Gregorian calendar
MINUTES_PER_DAY = 24 * 60
SECONDS_PER_DAY = 24 * 60 * 60

# TSI: Energy input to the top of the Earth's atmosphere
# https://www.ncei.noaa.gov/products/climate-data-records/total-solar-irradiance
TOTAL_SOLAR_IRRADIANCE = 1361  # W/m^2
# Seasonal variation in apparent solar irradiance due to Earth-Sun distance
SOLAR_IRRADIANCE_VARIATION = 47  # W/m^2
# Approximate perihelion, when Earth is closest to the sun (Jan 3rd)
PERIHELION = 3 * 2 * jnp.pi / DAYS_PER_YEAR  # radians
# Approximate equinox (March 20 on non-leap year), when dihedral is zero
SPRING_EQUINOX = 79 * 2 * jnp.pi / DAYS_PER_YEAR  # radians
# Angle between Earth's rotational axis and its orbital axis
EARTH_AXIS_INCLINATION = 23.45 * jnp.pi / 180  # radians


DatetimeLike = (
    jdt.Datetime | datetime.datetime | np.datetime64 | np.ndarray | str
)


@dataclasses.dataclass
class OrbitalTime:
  """Nondimensional time based on orbital dynamics.

  Attributes:
    orbital_phase: phase of the Earth's orbit around the Sun in radians. The
      values 0, 2pi correspond to January 1st, midnight UTC.
    synodic_phase: phase of the Earth's rotation around its axis in radians,
      relative to the Sun. The values 0, 2pi correspond to midnight UTC.
  """

  orbital_phase: float | Array
  synodic_phase: float | Array

  @classmethod
  def from_datetime(
      cls,
      when: DatetimeLike,
      origin: DatetimeLike = '1970-01-01T00:00:00',
      days_per_year: float = DAYS_PER_YEAR,
  ) -> Self:
    """Returns the OrbitalTime associated with the provided datetime.

    The calculation assumes all days are exactly 24 hours long (neglecting leap
    seconds).

    Specifying the precise origin and number of day per year is supported to
    enable exactly backwards compatibility with the original implementation in
    Dinosaur (https://github.com/neuralgcm/dinosaur), but makes a very small
    difference in practice (~0.01% relative difference in calculated radiation
    flux).

    Args:
      when: time for which to compute OrbitalTime.
      origin: time at which orbital and synodic phases are zero.
      days_per_year: number of days in a year.

    Returns:
      OrbitalTime corresponding to the provided datetime.
    """
    # TODO(shoyer): consider picking the default origin in a more principled
    # fashion, e.g., with the J2000 epoch used by astronomers, rather than the
    # Unix epoch:
    # https://en.wikipedia.org/wiki/Epoch_(astronomy)#J2000
    when = jdt.to_datetime(when)
    origin = jdt.to_datetime(origin)
    delta = when - origin
    fraction_of_day = delta.seconds / SECONDS_PER_DAY
    fraction_of_year = (delta.days + fraction_of_day) / days_per_year % 1
    return cls(
        orbital_phase=2 * jnp.pi * fraction_of_year,
        synodic_phase=2 * jnp.pi * fraction_of_day,
    )


jax.tree_util.register_dataclass(
    OrbitalTime,
    data_fields=['orbital_phase', 'synodic_phase'],
    meta_fields=[],
)


def direct_solar_irradiance(
    orbital_phase: Numeric,
    mean_irradiance: Numeric = TOTAL_SOLAR_IRRADIANCE,
    variation: Numeric = SOLAR_IRRADIANCE_VARIATION,
    perihelion: Numeric = PERIHELION,
) -> jnp.ndarray:
  """Returns solar radiation flux incident on the top of the atmosphere.

  Formula includes 6.9% seasonal variation due to Earth's elliptical orbit, but
  neglects the 0.1% variation of the 11-year solar cycle (Schwabe cycle).
  https://earth.gsfc.nasa.gov/climate/research/solar-radiation
  https://en.wikipedia.org/wiki/Solar_constant

  Note that the default values here are different from third_party/py/pysolar,
  which uses `1160 + (75 * math.sin(2 * math.pi / 365 * (day - 275)))`. This is
  presumably because pysolar is intended for modeling radiation incident on
  Earth's surface, including atmospheric scattering.

  Args:
    orbital_phase: phase of the Earth's orbit around the Sun in radians. The
      values 0, 2pi correspond to January 1st, midnight UTC.
    mean_irradiance: average annual solar flux just outside Earth's atmosphere.
      Default is the "total solar irradiance", or "solar consant", in W/m^2.
    variation: amplitude of fluctuation in solar flux due to Earth's elliptical
      orbit. Default is 0.5 * 6.9% of TSI. Units should match mean_irradiance.
    perihelion: orbital phase in radians where the Earth is closest to the Sun.
  """
  return mean_irradiance + variation * jnp.cos(orbital_phase - perihelion)


def get_declination(orbital_phase: Numeric) -> jnp.ndarray:
  """Returns angle between the Earth-Sun line and the Earth equitorial plane."""
  # https://en.wikipedia.org/wiki/Declination
  return EARTH_AXIS_INCLINATION * jnp.sin(orbital_phase - SPRING_EQUINOX)


def equation_of_time(orbital_phase: Numeric) -> jnp.ndarray:
  """Returns the value to add to mean solar time to get actual solar time."""
  # https://en.wikipedia.org/wiki/Equation_of_time
  b = orbital_phase - SPRING_EQUINOX
  added_minutes = 9.87 * jnp.sin(2 * b) - 7.53 * jnp.cos(b) - 1.5 * jnp.sin(b)
  # Output normalized as a correction to synodic_phase
  return 2 * jnp.pi * added_minutes / MINUTES_PER_DAY


def get_hour_angle(orbital_time: OrbitalTime, longitude: Array) -> jnp.ndarray:
  """Angular displacement of the sun east or west of the local meridian."""
  # https://en.wikipedia.org/wiki/Hour_angle
  solar_time = (
      orbital_time.synodic_phase
      + equation_of_time(orbital_time.orbital_phase)
      + jnp.deg2rad(longitude)
  )
  return solar_time - jnp.pi


def get_solar_sin_altitude(
    orbital_time: OrbitalTime,
    longitude: Array,
    latitude: Array,
) -> jnp.ndarray:
  """Returns sine of the solar altitude angle."""
  # Reference: https://en.wikipedia.org/wiki/Solar_zenith_angle
  # This function corresponds to get_altitude_fast() in pysolar:
  # https://github.com/pingswept/pysolar/blob/eabf85628640bbc4f8b2ac02c9b60a62830b0ac7/pysolar/solar.py#L134
  declination = get_declination(orbital_time.orbital_phase)
  hour_angle = get_hour_angle(orbital_time, longitude)
  lat_radians = jnp.deg2rad(latitude)
  first_term = jnp.cos(lat_radians) * jnp.cos(declination) * jnp.cos(hour_angle)
  second_term = jnp.sin(lat_radians) * jnp.sin(declination)
  return first_term + second_term


def radiation_flux(
    time: OrbitalTime | DatetimeLike,
    longitude: Array,
    latitude: Array,
    mean_irradiance: Numeric = TOTAL_SOLAR_IRRADIANCE,
    variation: Numeric = SOLAR_IRRADIANCE_VARIATION,
) -> jnp.ndarray:
  """Returns TOA incident radiation flux.

  `time`, `longitude`, and `latitude` must all have broadcast compatible shapes.

  Args:
    time: times for which to compute solar radiation flux, either as an
      OrbitalTime or value that can be converted to a jax_datetime.Datetime.
    longitude: array of longitudes in degrees.
    latitude: array of latitudes in degrees.
    mean_irradiance: average annual solar flux just outside Earth's atmosphere.
      Default is the "total solar irradiance", or "solar consant", in W/m^2.
    variation: amplitude of fluctuation in solar flux due to Earth's elliptical
      orbit. Default is 0.5 * 6.9% of TSI. Units should match mean_irradiance.

  Return:
    Array with normalized TOA incident solar radiation flux in the same units as
    `mean_irradiance` and `variation`, with the shape resulting from
    broadcasting `time`, `longitude`, and `latitude` against each other.
  """
  orbital_time = (
      time if isinstance(time, OrbitalTime) else OrbitalTime.from_datetime(time)
  )
  sin_altitude = get_solar_sin_altitude(
      orbital_time=orbital_time,
      longitude=longitude,
      latitude=latitude,
  )
  is_daytime = sin_altitude > 0
  flux = direct_solar_irradiance(
      orbital_phase=orbital_time.orbital_phase,
      mean_irradiance=mean_irradiance,
      variation=variation,
  )
  return flux * is_daytime * sin_altitude


def normalized_radiation_flux(
    time: OrbitalTime | DatetimeLike,
    longitude: Array,
    latitude: Array,
    mean_irradiance: Numeric = TOTAL_SOLAR_IRRADIANCE,
    variation: Numeric = SOLAR_IRRADIANCE_VARIATION,
) -> jnp.ndarray:
  """Like radiation_flux(), but normalized to between 0 and 1."""
  scale = mean_irradiance + variation
  return radiation_flux(
      time,
      longitude,
      latitude,
      mean_irradiance=mean_irradiance / scale,
      variation=variation / scale,
  )
