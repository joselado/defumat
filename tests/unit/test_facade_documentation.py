"""Every ``Calculator`` method is in the README and in the user guide.

``CLAUDE.md`` makes two claims that have to stay true together. **The front door
is ``Calculator``**: it is "what the README, the user guide and new notebooks
use", and a feature "adds a ``get_*`` method in the same pass that adds its entry
point". And **the audit that catches drift is a set difference, not a
read-through** -- which is exactly what a read-through missed for months.

It missed a specific thing, and the shape of it is worth stating because it is
what this file is built against. Nothing was *undocumented*: every quantity had
its section and its table row. What they named was the **functional entry
point** -- ``run_absorption``, ``elastic_constants``, ``raman_tensors`` -- and
never the bound method beside it. So a reader who knew the physics and grepped
for ``get_absorption`` found nothing, in a document whose own introduction says
that is the way to call it. Fifteen of the forty-six were missing from
``docs/features.tex`` and thirteen from ``README.md``, and a human reading either
document end to end would not have noticed, because each page is individually
complete.

The fix is mechanical and so is the check: a set difference against
``dir(Calculator)``, run in the fast gate. It cannot say whether an entry is any
*good* -- that is what the executed snippet and the amber box are for -- but it
can say the name is there, and that is the half that drifts silently.
"""

import re
from pathlib import Path

import pytest

from defumat import Calculator

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "features.tex"

#: Methods with no row of their own in the README's feature table, each for a
#: stated reason. **This set does not grow without one**: the table "names
#: quantities, not routines" and "the rows are physics, not knobs"
#: (``CLAUDE.md``), so the bar is that the method is machinery a physics row
#: already covers, not that writing a row is inconvenient.
NOT_A_FEATURE_ROW = {
    # Diagonalising at a fixed density on a grid is not a quantity anyone sets
    # out to compute; it is what the band-structure and density-of-states rows
    # are built on, and both name it.
    "get_nscf",
}


def _methods():
    return sorted(name for name in dir(Calculator) if name.startswith("get_"))


def test_there_are_methods_to_check():
    """A guard on the guard: an empty set difference proves nothing.

    ``CLAUDE.md``'s own list of recurring traps has "a check whose null result
    cannot be told from a pass" on it. If ``dir(Calculator)`` ever stopped
    returning bound methods -- a renamed prefix, a refactor into a mixin that
    this import does not reach -- every assertion below would pass on an empty
    set and read as agreement.
    """
    assert len(_methods()) > 30


def test_every_calculator_method_is_named_in_the_user_guide():
    """``docs/features.tex``, which the "every method in one place" table covers.

    LaTeX escapes the underscore, so ``get_dos`` is written ``get\\_dos``; both
    spellings count, because the bare one appears inside ``pycode`` blocks.
    """
    text = GUIDE.read_text()
    missing = [m for m in _methods()
               if m not in text and m.replace("_", r"\_") not in text]
    assert not missing, (
        f"{len(missing)} Calculator methods are absent from docs/features.tex: "
        f"{missing}. Add each to the 'Every method, in one place' table, and to "
        "the section where its physics is described"
    )


def test_the_guide_names_no_method_that_does_not_exist():
    """The other direction, which is the one that goes stale after a rename."""
    text = GUIDE.read_text()
    named = {m.replace(r"\_", "_")
             for m in re.findall(r"\\code\{(get(?:\\_\w+)+)\}", text)}
    unknown = sorted(named - set(_methods()))
    assert not unknown, (
        f"docs/features.tex names methods Calculator does not have: {unknown}"
    )


def test_every_calculator_method_is_named_in_the_readme_feature_table():
    """``README.md``'s entry-point column, which is a claim about *this* code.

    ``CLAUDE.md``: "The entry-point column is a claim about this code. A row
    naming something nothing parses is worse than no row." The converse holds
    too -- a method no row names is a feature a reader cannot find from the
    table that is supposed to list them all.
    """
    text = README.read_text()
    missing = [m for m in _methods()
               if m not in text and m not in NOT_A_FEATURE_ROW]
    assert not missing, (
        f"{len(missing)} Calculator methods are absent from README.md: "
        f"{missing}. Add each to the entry-point column of the row whose "
        "quantity it computes -- not a new row, unless the quantity is new"
    )


def test_the_readme_exemptions_are_real_methods():
    """An exemption for a method that no longer exists is a silent hole."""
    stale = sorted(NOT_A_FEATURE_ROW - set(_methods()))
    assert not stale, (
        f"NOT_A_FEATURE_ROW exempts methods that do not exist: {stale}"
    )
