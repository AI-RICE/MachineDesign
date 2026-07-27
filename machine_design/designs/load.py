import os

from .computation import ComputationBase
from .design import Design
from .geometry import GeometryBase


def load_design(
    file_name_aedt,
    project_name,
    design_name,
    aedt_version,
    geometry: GeometryBase,
    computation: ComputationBase,
    new_desktop=False,
    non_graphical=False,
    close_on_exit=True,
    **kwargs,
):

    if not os.path.exists(file_name_aedt):
        return Design.create(
            project_name,
            design_name,
            file_name_aedt,
            geometry,
            computation,
            version=aedt_version,
            non_graphical=non_graphical,
            new_desktop=new_desktop,
            close_on_exit=close_on_exit,
            **kwargs,
        )
    else:
        return Design.load(
            file_name_aedt,
            geometry,
            computation,
            version=aedt_version,
            non_graphical=non_graphical,
            new_desktop=new_desktop,
            close_on_exit=close_on_exit,
            **kwargs,
        )
