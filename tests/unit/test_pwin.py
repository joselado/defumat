"""The input parser's job is Fortran syntax, and Fortran syntax has traps."""

import pytest

from pypresso.io.pwin import parse_pw_input

pytestmark = pytest.mark.unit


def test_namelist_scalars_and_types():
    inp = parse_pw_input(
        """
        &CONTROL
           calculation = 'scf'
           tstress = .true., tprnfor = .FALSE.
           nstep = 50
        /
        &electrons
           conv_thr = 1.d-8   ! Fortran double exponent
           mixing_beta = 0.7
        /
        """
    )
    assert inp.namelists["control"]["calculation"] == "scf"
    assert inp.namelists["control"]["tstress"] is True
    assert inp.namelists["control"]["tprnfor"] is False
    assert inp.namelists["control"]["nstep"] == 50
    assert isinstance(inp.namelists["control"]["nstep"], int)
    assert inp.namelists["electrons"]["conv_thr"] == pytest.approx(1e-8)


def test_uppercase_namelist_name_is_lowered():
    """&SYSTEM and &System must land in the same place as &system."""
    inp = parse_pw_input("&SYSTEM\n ibrav = 2\n/\n")
    assert inp.get("system", "ibrav") == 2


def test_indexed_variables():
    inp = parse_pw_input("&system\n celldm(1)=10.2, celldm(3) = 1.5\n/\n")
    assert inp.namelists["system"]["celldm"] == {(1,): 10.2, (3,): 1.5}
    assert inp.indexed("system", "celldm", 6) == [10.2, 0.0, 1.5, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="out of range"):
        inp.indexed("system", "celldm", 2)


def test_comments_are_stripped_but_not_inside_strings():
    inp = parse_pw_input(
        "&control\n"
        "  prefix = 'a!b#c'   ! this is a comment\n"
        "  outdir = './out/'  # so is this\n"
        "/\n"
    )
    assert inp.get("control", "prefix") == "a!b#c"
    assert inp.get("control", "outdir") == "./out/"


@pytest.mark.parametrize("header", ["K_POINTS {crystal}", "K_POINTS (crystal)", "K_POINTS crystal"])
def test_card_option_syntaxes(header):
    inp = parse_pw_input(f"{header}\n 1\n 0.0 0.0 0.0 1.0\n")
    card = inp.require_card("K_POINTS")
    assert card.option == "crystal"
    assert card.floats() == [[1.0], [0.0, 0.0, 0.0, 1.0]]


def test_card_without_option():
    inp = parse_pw_input("ATOMIC_SPECIES\n Si 28.086 Si.pz-vbc.UPF\n")
    assert inp.require_card("ATOMIC_SPECIES").option is None


def test_cards_end_at_the_next_card():
    inp = parse_pw_input(
        "ATOMIC_SPECIES\n Si 28.086 Si.upf\n"
        "ATOMIC_POSITIONS alat\n Si 0 0 0\n Si 0.25 0.25 0.25\n"
        "K_POINTS gamma\n"
    )
    assert len(inp.require_card("ATOMIC_SPECIES").lines) == 1
    assert len(inp.require_card("ATOMIC_POSITIONS").lines) == 2
    assert inp.require_card("K_POINTS").lines == ()


def test_missing_card_and_unknown_input_are_errors():
    inp = parse_pw_input("&system\n ibrav=0\n/\n")
    assert inp.card("CELL_PARAMETERS") is None
    with pytest.raises(ValueError, match="missing required card"):
        inp.require_card("CELL_PARAMETERS")
    with pytest.raises(ValueError, match="unrecognised input"):
        parse_pw_input("this is not a card\n")


def test_unterminated_namelist_is_an_error():
    with pytest.raises(ValueError, match="not terminated"):
        parse_pw_input("&system\n ibrav = 2\n")


def test_get_returns_default_without_inventing_physics():
    inp = parse_pw_input("&system\n ibrav=2\n/\n")
    assert inp.get("system", "ecutrho") is None
    assert inp.get("system", "ecutrho", 48.0) == 48.0
