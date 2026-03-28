from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .adapter_library import get_adapter_library
from .macros import MACRO_LIBRARY


@dataclass(frozen=True)
class TutorialLibrary:
    name: str
    macro_names: tuple[str, ...]
    adapter_library_names: tuple[str, ...] = ()
    description: str = ""

    def build_macro(self, macro_name: str, **kwargs: Any):
        if macro_name not in self.macro_names:
            available = ", ".join(self.macro_names)
            raise KeyError(
                f"Macro '{macro_name}' is not part of tutorial library '{self.name}'. "
                f"Available: {available}"
            )
        return MACRO_LIBRARY[macro_name](**kwargs)

    def macro_builders(self) -> Dict[str, Callable[..., Any]]:
        return {name: MACRO_LIBRARY[name] for name in self.macro_names}


FOUNDATIONS_TUTORIAL_LIBRARY = TutorialLibrary(
    name="foundations",
    macro_names=("ket", "completion", "structured_lm_duality", "db_square", "gt_neighborhood"),
    description="Core aggregation, commutativity, and geometric neighborhood blocks.",
)

PLANNING_TUTORIAL_LIBRARY = TutorialLibrary(
    name="planning",
    macro_names=("basket_workflow", "rocket_repair", "basket_rocket_pipeline"),
    adapter_library_names=("standard",),
    description="Workflow drafting and repair blocks for BASKET/ROCKET style tutorials.",
)

UNIFIED_TUTORIAL_LIBRARY = TutorialLibrary(
    name="unified",
    macro_names=tuple(sorted(MACRO_LIBRARY)),
    adapter_library_names=("standard",),
    description="Full FunctorFlow tutorial surface over the current macro and adapter packs.",
)


TUTORIAL_LIBRARIES: Dict[str, TutorialLibrary] = {
    FOUNDATIONS_TUTORIAL_LIBRARY.name: FOUNDATIONS_TUTORIAL_LIBRARY,
    PLANNING_TUTORIAL_LIBRARY.name: PLANNING_TUTORIAL_LIBRARY,
    UNIFIED_TUTORIAL_LIBRARY.name: UNIFIED_TUTORIAL_LIBRARY,
}


def get_tutorial_library(name: str) -> TutorialLibrary:
    try:
        return TUTORIAL_LIBRARIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(TUTORIAL_LIBRARIES))
        raise KeyError(f"Unknown FunctorFlow tutorial library '{name}'. Available: {available}") from exc


def install_tutorial_library(diagram, library: str | TutorialLibrary) -> TutorialLibrary:
    tutorial_library = get_tutorial_library(library) if isinstance(library, str) else library
    for adapter_library_name in tutorial_library.adapter_library_names:
        diagram.use_adapter_library(get_adapter_library(adapter_library_name))
    return tutorial_library
