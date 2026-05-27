import os

from .design import Design


def load_design(
    file_name_aedt,
    project_name,
    design_name,
    aedt_version,
    design_cls=Design,
    new_desktop=False,
    non_graphical=True,
    close_on_exit=True,
    **kwargs,
):

    if not os.path.exists(file_name_aedt):
        return design_cls.create(
            project_name,
            design_name,
            file_name_aedt,
            version=aedt_version,
            non_graphical=non_graphical,
            new_desktop=new_desktop,
            close_on_exit=close_on_exit,
            **kwargs,
        )
    else:
        return design_cls.load(
            file_name_aedt,
            design_name=design_name,
            version=aedt_version,
            non_graphical=non_graphical,
            new_desktop=new_desktop,
            close_on_exit=close_on_exit,
            **kwargs,
        )
