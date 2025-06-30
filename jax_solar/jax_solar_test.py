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
import datetime

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import jax_datetime as jdt
import jax_solar
import numpy as np


TWOPI = 2 * jnp.pi


def get_global_mesh():
  longitude = jnp.linspace(0, 360, num=360)
  latitude = jnp.linspace(-90, 90, num=180)
  return jnp.meshgrid(longitude, latitude, indexing='ij')


class RadiationTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          when='1970-01-01',
          expected_orbital_phase=0,
          expected_synodic_phase=0,
      ),
      dict(
          when='1970-01-01T12:00',
          expected_orbital_phase=0.5 * TWOPI / 365.2524,
          expected_synodic_phase=jnp.pi,
      ),
      dict(
          when='1970-01-01T23:59',
          expected_orbital_phase=(1439 / 1440) * TWOPI / 365.2524,
          expected_synodic_phase=(1439 / 1440) * TWOPI,
      ),
      dict(
          when='1970-05-01',
          days_per_year=365,
          expected_orbital_phase=(31 + 28 + 31 + 30) * TWOPI / 365,
          expected_synodic_phase=0,
      ),
      dict(
          when=jdt.Datetime(
              delta=jdt.Timedelta(days=1_000 * 365 + 50, seconds=6 * 60 * 60)
          ),
          days_per_year=365,
          expected_orbital_phase=50.25 / 365 * TWOPI,
          expected_synodic_phase=0.25 * TWOPI,
          orbital_phase_atol=2e-4,
      ),
  )
  def test_orbital_time_from_datetime(
      self,
      when,
      expected_orbital_phase,
      expected_synodic_phase,
      orbital_phase_atol=1e-6,
      synodic_phase_atol=1e-6,
      **kwargs,
  ):
    actual = jax_solar.OrbitalTime.from_datetime(when, **kwargs)
    np.testing.assert_allclose(
        actual.orbital_phase, expected_orbital_phase, atol=orbital_phase_atol
    )
    np.testing.assert_allclose(
        actual.synodic_phase, expected_synodic_phase, atol=synodic_phase_atol
    )

  def test_direct_solar_irradiance_no_units(self):
    flux = jax_solar.direct_solar_irradiance(
        orbital_phase=jnp.linspace(0, TWOPI, 4, endpoint=False),
        mean_irradiance=1.2,
        variation=0.3,
        perihelion=0,
    )
    np.testing.assert_allclose(flux, [1.5, 1.2, 0.9, 1.2])

  @parameterized.parameters(
      # days of the year when equation of time is nearly zero
      (datetime.datetime(2000, 4, 16),),
      (datetime.datetime(2000, 6, 14),),
      (datetime.datetime(2000, 8, 31),),
      (datetime.datetime(2000, 12, 25),),
  )
  def test_equation_of_time(self, when):
    ot = jax_solar.OrbitalTime.from_datetime(when)
    delta_phase = jax_solar.equation_of_time(ot.orbital_phase)
    np.testing.assert_allclose(delta_phase, 0, atol=0.006)

  def test_solar_hour_angle(self):
    # equation of time is approx 0
    orbital_time = jax_solar.OrbitalTime.from_datetime(
        datetime.datetime(1979, 4, 15, 0, 0)
    )
    longitudes = jnp.linspace(0, 360, num=360)
    actual = jax_solar.get_hour_angle(orbital_time, longitudes)
    # at UTC=0, when equation of time is 0, expect hour_angle=`longitude - 180`
    expected = np.deg2rad(longitudes - 180.0)
    np.testing.assert_allclose(actual, expected, atol=0.02)

  @parameterized.parameters(
      dict(time='1979-07-04', expected_max=1361 - 47),  # aphelion
      dict(time='1980-07-03', expected_max=1361 - 47),  # aphelion
      dict(time='2022-07-04', expected_max=1361 - 47),  # aphelion
      dict(time='1979-01-03', expected_max=1361 + 47),  # perihelion
  )
  def test_aphelion_and_perihelion(self, time, expected_max):
    longitude, latitude = get_global_mesh()
    actual_max = jax_solar.radiation_flux(time, longitude, latitude).max()
    np.testing.assert_allclose(actual_max, expected_max, rtol=5e-5)

    actual_normalized_max = jax_solar.normalized_radiation_flux(
        time, longitude, latitude
    ).max()
    expected_normalized_max = expected_max / (1361 + 47)
    np.testing.assert_allclose(
        actual_normalized_max, expected_normalized_max, rtol=5e-5
    )

  @parameterized.parameters(
      dict(time='1970-01-01T00:00'),
      dict(time='1986-05-31T12:00'),
      dict(time='2024-12-27T17:08'),
  )
  def test_earth_always_half_in_darkness(self, time):
    longitude, latitude = get_global_mesh()
    flux = jax_solar.radiation_flux(time, longitude, latitude)
    area_weight = np.cos(np.deg2rad(latitude))
    area_weight /= area_weight.mean()
    fraction_dark = ((flux <= 0) * area_weight).mean()
    np.testing.assert_allclose(fraction_dark, 0.5, atol=0.002)

  def test_eternal_night_in_artic_winter(self):
    time = jax_solar.OrbitalTime.from_datetime(
        jdt.to_datetime('2024-01-01T00')
        + jdt.Timedelta(seconds=60 * 60 * jnp.arange(24))
    )
    flux = jax_solar.radiation_flux(time, longitude=0.0, latitude=85.0)
    np.testing.assert_allclose(flux, np.zeros(24))

  def test_eternal_day_in_artic_summer(self):
    time = jax_solar.OrbitalTime.from_datetime(
        jdt.to_datetime('2024-06-21T00')
        + jdt.Timedelta(seconds=60 * 60 * jnp.arange(24))
    )
    flux = jax_solar.radiation_flux(time, longitude=0.0, latitude=85.0)
    self.assertTrue((flux > 0).all())


if __name__ == '__main__':
  absltest.main()
