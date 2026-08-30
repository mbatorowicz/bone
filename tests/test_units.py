"""Układ jednostek: czy stałe są tym, za co się podają.

Stała wpisana ręcznie jest najtańszym miejscem na literówkę w całym projekcie —
nic jej nie weryfikuje, a przesunięty przecinek przechodzi przez wszystkie testy
dynamiki, bo one sprawdzają spójność, nie skalę. Dlatego każda liczba z
``bone.units`` jest tutaj odtwarzana z definicji albo z niezależnie znanej
wartości literaturowej.
"""

from __future__ import annotations

import math

import pytest

from bone import units


def test_megaparsec_matches_its_definition_from_the_astronomical_unit():
    """1 pc to odległość, z której 1 au widać pod kątem jednej sekundy łuku."""
    au_km = 1.495978707e8
    arcsec_per_radian = 648000.0 / math.pi
    from_definition = 1e6 * au_km * arcsec_per_radian
    assert pytest.approx(from_definition, rel=1e-9) == units.MPC_KM


def test_gigayear_is_a_billion_julian_years():
    julian = 365.25 * 86400.0 * 1e9
    assert pytest.approx(julian, rel=1e-12) == units.GYR_S


def test_gravitational_constant_reproduces_the_cgs_value():
    """G w jednostkach kodu to ``G_cgs · M_jedn / (L_jedn · V_jedn²)``.

    Stałe wejściowe są te, których używa GADGET; patrz uwaga w ``units``
    o 0,03% różnicy względem dzisiejszego CODATA.
    """
    g_cgs = 6.672e-8
    mass_unit_g = 1e10 * 1.989e33
    length_unit_cm = units.MPC_KM * 1e5
    velocity_unit_cms = 1e5
    expected = g_cgs * mass_unit_g / (length_unit_cm * velocity_unit_cms**2)
    assert pytest.approx(expected, rel=1e-5) == units.G


def test_hubble_time_is_the_textbook_value():
    """``1/H₀`` musi wyjść 9,78 h⁻¹ Gyr — to sprawdza H100 i GYR_PER_CODE_TIME naraz."""
    hubble_time_gyr = units.GYR_PER_CODE_TIME / units.H100
    assert hubble_time_gyr == pytest.approx(9.778, rel=1e-4)


def test_critical_density_matches_the_reference_value():
    """Kryterium odbioru kroku 1.2: 2,7754·10¹ w jednostkach kodu."""
    assert units.critical_density_0(0.6736) == pytest.approx(27.754, rel=1e-4)


@pytest.mark.parametrize("h", [0.5, 0.6736, 1.0])
def test_critical_density_reproduces_the_physical_value_in_solar_masses(h):
    """Ta sama liczba w M☉/Mpc³ to znane ``2,775·10¹¹ h²``.

    Przelicznik: ``(10¹⁰ M☉/h)/(Mpc/h)³ = 10¹⁰ h² M☉/Mpc³``. Stała jest wolna
    od ``h`` tylko w jednostkach kodu — fizyczna gęstość rośnie jak ``h²`` i to
    właśnie sprawdza przebieg po kilku wartościach.
    """
    physical = units.critical_density_0(h) * 1e10 * h**2
    assert physical == pytest.approx(2.7754e11 * h**2, rel=1e-4)


def test_critical_density_does_not_depend_on_h():
    """W tych jednostkach h skraca się dokładnie — gdyby nie, coś jest nie tak."""
    assert units.critical_density_0(0.5) == units.critical_density_0(1.0)
    assert units.critical_density_0(0.6736) == units.critical_density_0(0.7)


def test_critical_density_follows_its_own_formula():
    expected = 3.0 * units.H100**2 / (8.0 * math.pi * units.G)
    assert units.critical_density_0() == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("bad_h", [0.0, -0.7, float("nan")])
def test_nonpositive_h_is_rejected(bad_h):
    with pytest.raises(ValueError):
        units.critical_density_0(bad_h)
