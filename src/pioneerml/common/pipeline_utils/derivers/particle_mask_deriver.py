class ParticleMaskDeriver:
    """Derive compact particle mask bits from PDG code."""

    def mask_from_pdg(self, pdg: int) -> int:
        if pdg == 211:
            return 0b00001
        if pdg == -13:
            return 0b00010
        if pdg == -11:
            return 0b00100
        if pdg == 11:
            return 0b01000
        return 0b10000
