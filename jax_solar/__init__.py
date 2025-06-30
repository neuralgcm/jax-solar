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
"""Public API for jax_solar."""
# pylint: disable=g-multiple-import,g-importing-member,useless-import-alias

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570
from jax_solar._jax_solar import (
    OrbitalTime as OrbitalTime,
    direct_solar_irradiance as direct_solar_irradiance,
    equation_of_time as equation_of_time,
    get_declination as get_declination,
    get_hour_angle as get_hour_angle,
    get_solar_sin_altitude as get_solar_sin_altitude,
    normalized_radiation_flux as normalized_radiation_flux,
    radiation_flux as radiation_flux,
)

__version__ = "0.1.0"  # keep sync with pyproject.toml
