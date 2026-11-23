from types import SimpleNamespace

import pytest

from machine_design.designs.computation import ComputationBase


class _DummyComputation(ComputationBase):
    def __init__(self, geometry, oper_params, output_vars):
        self._oper_params = oper_params
        self._output_vars = output_vars
        super().__init__(geometry)

    def set_oper_params(self):
        self.oper_params = self._oper_params

    def set_solution_expressions(self):
        self.solution_expressions = "Moving1.Torque"

    def set_output_vars(self):
        self.output_vars = self._output_vars

    def set_post_params(self):
        self.post_params = {}

    def assign_stator_coils(self, m2d):
        pass

    def inductance_computation(self, m2d):
        pass

    def set_variables(self, m2d, *args):
        pass

    def extract_results(self, solutions):
        pass


def test_no_reserved_names_does_not_raise():
    _DummyComputation(SimpleNamespace(n_phases=3), {"Im": "1A"}, {"pos": "Moving1.Position"})


def test_reserved_name_in_oper_params_raises():
    with pytest.raises(ValueError, match="F"):
        _DummyComputation(SimpleNamespace(n_phases=3), {"F": "50Hz"}, {})


def test_reserved_name_in_output_vars_raises():
    with pytest.raises(ValueError, match="Time"):
        _DummyComputation(SimpleNamespace(n_phases=3), {}, {"Time": "0s"})


def test_reserved_name_check_is_case_insensitive():
    with pytest.raises(ValueError, match="f"):
        _DummyComputation(SimpleNamespace(n_phases=3), {"f": "50Hz"}, {})
