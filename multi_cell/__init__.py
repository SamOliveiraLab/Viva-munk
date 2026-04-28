"""
Registration-related functions
"""

__version__ = "0.0.1"

from process_bigraph import register_types as pb_register_types
from process_bigraph.types.process import register_types as pb_types_register
from process_bigraph.composite import Composite
from process_bigraph.emitter import RAMEmitter
try:
    from process_bigraph.emitter import SQLiteEmitter
except ImportError:
    SQLiteEmitter = None
from bigraph_schema.core import Core, BASE_TYPES
from bigraph_viz import register_types as viz_register_types

from multi_cell.processes.multibody import PymunkProcess
from multi_cell.processes.grow_divide import GrowDivide, AdderGrowDivide
from multi_cell.processes.remove_crossing import RemoveCrossing
from multi_cell.processes.secrete_eps import SecreteEPS
from multi_cell.processes.pressure import Pressure
from multi_cell.processes.diffusion_advection import DiffusionAdvection
from multi_cell.processes.cell_field_exchange import CellFieldExchange
from multi_cell.processes.chemotaxis import Chemotaxis
from multi_cell.processes.inclusion_body import InclusionBody, IBColony
from multi_cell.processes.quorum_sensing import QuorumSensing
from multi_cell.processes.field_decay import FieldDecay
from multi_cell.pymunk_agent_type import PymunkAgent, register_pymunk_agent_dispatches
from multi_cell.types import positive_types

# Register custom dispatches once at module import
register_pymunk_agent_dispatches()


def _patch_composite_realize_on_sentinels():
    """Ensure Composite re-realizes newly-added subtrees on `_add`/`_remove`.

    `Map.apply` drops `_add` entries straight into state without calling
    the value type's realize path, and `Composite.apply_updates` only
    triggers `core.realize` on schema-level merges — not on structural
    sentinels. So embedded process specs (e.g. `grow_divide` on a freshly
    divided daughter) are never instantiated: the spec is visible in
    state but its `'instance'` slot stays unset and the Process class
    never runs on that cell. Result: lineages freeze after the first
    division in every experiment that uses per-cell embedded processes.

    The fix is a one-line amend: when structural sentinels are detected,
    run `core.realize` (which follows `Link`/process specs and creates
    instances via `realize_link`) before `find_instance_paths` scans
    state for them.
    """
    _original_apply_updates = Composite.apply_updates

    def apply_updates_with_realize(self, updates):
        update_paths = _original_apply_updates(self, updates)
        if getattr(self, '_last_apply_structural', False):
            self.schema, self.state = self.core.realize(self.schema, self.state)
            self.find_instance_paths(self.state)
            self._build_view_project_cache()
        return update_paths

    Composite.apply_updates = apply_updates_with_realize


_patch_composite_realize_on_sentinels()


def register_pymunk_types(core):
    # Use the optimized PymunkAgent Node subclass instead of a dict schema.
    # This eliminates per-field dispatch overhead in apply/reconcile/realize.
    core.register_type('pymunk_agent', PymunkAgent())
    # Concentration / field types used by DiffusionAdvection + CellFieldExchange
    for name, schema in positive_types.items():
        core.register_type(name, schema)


def register_processes(core):
    core.register_link('PymunkProcess', PymunkProcess)
    core.register_link('GrowDivide', GrowDivide)
    core.register_link('AdderGrowDivide', AdderGrowDivide)
    core.register_link('RemoveCrossing', RemoveCrossing)
    core.register_link('SecreteEPS', SecreteEPS)
    core.register_link('Pressure', Pressure)
    core.register_link('DiffusionAdvection', DiffusionAdvection)
    core.register_link('CellFieldExchange', CellFieldExchange)
    core.register_link('Chemotaxis', Chemotaxis)
    core.register_link('InclusionBody', InclusionBody)
    core.register_link('IBColony', IBColony)
    core.register_link('QuorumSensing', QuorumSensing)
    core.register_link('FieldDecay', FieldDecay)
    core.register_link('Composite', Composite)
    core.register_link('RAMEmitter', RAMEmitter)
    if SQLiteEmitter is not None:
        core.register_link('SQLiteEmitter', SQLiteEmitter)


def core_import(core=None, config=None):
    if not core:
        core = Core(BASE_TYPES)
    pb_types_register(core)
    pb_register_types(core)
    viz_register_types(core)
    register_pymunk_types(core)
    register_processes(core)
    return core
