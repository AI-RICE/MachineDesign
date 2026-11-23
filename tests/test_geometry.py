import pytest

from machine_design.designs.geometry import GeometryBase


class _DummyGeometry(GeometryBase):
    def __init__(self, geom_params, wind_params, slot_params, mod_params):
        self._geom_params = geom_params
        self._wind_params = wind_params
        self._slot_params = slot_params
        self._mod_params = mod_params
        super().__init__()

    def set_iron(self):
        pass

    def set_geom_params(self):
        self.geom_params = self._geom_params

    def set_slot_params(self):
        self.slot_params = self._slot_params

    def set_winds_params(self):
        self.wind_params = self._wind_params

    def set_mod_params(self):
        self.mod_params = self._mod_params

    def set_rot_points(self):
        pass

    def set_derived_params(self):
        pass

    def set_udp_par_list_stator(self):
        pass

    def build_stator(self, m2d):
        pass


def test_no_reserved_names_does_not_raise():
    _DummyGeometry({"DiaStatorGap": "80mm"}, {"Nc": "10"}, {"Hs0": "1mm"}, {"Poles": "4"})


def test_reserved_name_in_geom_params_raises():
    with pytest.raises(ValueError, match="Distance"):
        _DummyGeometry({"Distance": "80mm"}, {}, {}, {})


def test_reserved_name_in_mod_params_raises():
    with pytest.raises(ValueError, match="Time"):
        _DummyGeometry({}, {}, {}, {"Time": "1s"})


def test_reserved_name_check_is_case_insensitive():
    with pytest.raises(ValueError, match="z"):
        _DummyGeometry({}, {}, {"z": "1mm"}, {})
